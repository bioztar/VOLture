"""Interactive Brokers client wrapper using ib_async."""

import asyncio
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from ib_async import IB, Contract, Option, Stock, Ticker, util

from volture.config import IBConfig
from volture.types import Candle, VolEstimate

logger = logging.getLogger(__name__)


class IBClient:
    """Async wrapper around ib_async for options-focused operations."""

    def __init__(self, config: IBConfig):
        self._config = config
        self._ib = IB()

    async def connect(self) -> None:
        await self._ib.connectAsync(
            host=self._config.host,
            port=self._config.port,
            clientId=self._config.client_id,
        )
        mode = "PAPER" if self._config.is_paper else "LIVE"
        logger.info(f"Connected to IB ({mode}) at {self._config.host}:{self._config.port}")

    async def disconnect(self) -> None:
        self._ib.disconnect()
        logger.info("Disconnected from IB")

    async def get_stock_price(self, symbol: str) -> float:
        """Get current price for a stock/ETF."""
        contract = Stock(symbol, "SMART", "USD")
        await self._ib.qualifyContractsAsync(contract)
        [ticker] = await self._ib.reqTickersAsync(contract)
        price = ticker.marketPrice()
        if np.isnan(price):
            price = ticker.close
        return float(price)

    async def get_historical_candles(
        self,
        symbol: str,
        duration: str = "30 D",
        bar_size: str = "1 hour",
    ) -> tuple[Candle, ...]:
        """Fetch historical OHLCV bars from IB."""
        contract = Stock(symbol, "SMART", "USD")
        await self._ib.qualifyContractsAsync(contract)

        bars = await self._ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )

        candles = tuple(
            Candle(
                timestamp=bar.date if isinstance(bar.date, datetime) else datetime.fromisoformat(str(bar.date)),
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
                amount=float(bar.volume * bar.close),
            )
            for bar in bars
        )
        logger.info(f"Fetched {len(candles)} candles for {symbol}")
        return candles

    async def get_option_chains(
        self,
        symbol: str,
        max_expiries: int = 4,
    ) -> list[dict]:
        """Get available option chain parameters for a symbol."""
        contract = Stock(symbol, "SMART", "USD")
        await self._ib.qualifyContractsAsync(contract)

        chains = await self._ib.reqSecDefOptParamsAsync(
            underlyingSymbol=symbol,
            futFopExchange="",
            underlyingSecType="STK",
            underlyingConId=contract.conId,
        )

        results = []
        for chain in chains:
            if chain.exchange == "SMART":
                results.append({
                    "exchange": chain.exchange,
                    "trading_class": chain.tradingClass,
                    "expirations": sorted(chain.expirations)[:max_expiries],
                    "strikes": sorted(chain.strikes),
                })
        return results

    async def get_option_greeks(
        self,
        symbol: str,
        expiry: str,
        strikes: list[float],
        rights: list[str] | None = None,
    ) -> list[dict]:
        """Get Greeks for specific option contracts."""
        if rights is None:
            rights = ["C", "P"]

        contracts = []
        for strike in strikes:
            for right in rights:
                opt = Option(symbol, expiry, strike, right, "SMART")
                contracts.append(opt)

        qualified = await self._ib.qualifyContractsAsync(*contracts)
        valid_contracts = [c for c in qualified if c.conId > 0]

        if not valid_contracts:
            logger.warning(f"No valid option contracts found for {symbol} {expiry}")
            return []

        tickers = await self._ib.reqTickersAsync(*valid_contracts)
        await asyncio.sleep(2)  # allow Greeks to populate

        results = []
        for ticker in tickers:
            greeks = ticker.modelGreeks
            if greeks is None:
                continue

            results.append({
                "symbol": ticker.contract.symbol,
                "expiry": ticker.contract.lastTradeDateOrContractMonth,
                "strike": ticker.contract.strike,
                "right": ticker.contract.right,
                "bid": ticker.bid if not np.isnan(ticker.bid) else 0.0,
                "ask": ticker.ask if not np.isnan(ticker.ask) else 0.0,
                "last": ticker.last if not np.isnan(ticker.last) else 0.0,
                "iv": greeks.impliedVol if greeks.impliedVol else 0.0,
                "delta": greeks.delta if greeks.delta else 0.0,
                "gamma": greeks.gamma if greeks.gamma else 0.0,
                "theta": greeks.theta if greeks.theta else 0.0,
                "vega": greeks.vega if greeks.vega else 0.0,
                "underlying_price": greeks.undPrice if greeks.undPrice else 0.0,
            })

        logger.info(f"Got Greeks for {len(results)} contracts ({symbol} {expiry})")
        return results

    async def get_atm_iv(self, symbol: str, expiry: str) -> VolEstimate:
        """Get ATM implied volatility for a symbol and expiry."""
        price = await self.get_stock_price(symbol)

        chain_data = await self.get_option_chains(symbol, max_expiries=8)
        if not chain_data:
            raise ValueError(f"No option chains found for {symbol}")

        strikes = chain_data[0]["strikes"]
        atm_strike = min(strikes, key=lambda s: abs(s - price))

        nearby_strikes = [
            s for s in strikes
            if abs(s - price) / price < 0.03  # within 3% of ATM
        ]
        if not nearby_strikes:
            nearby_strikes = [atm_strike]

        greeks = await self.get_option_greeks(symbol, expiry, nearby_strikes)
        if not greeks:
            raise ValueError(f"No Greeks available for {symbol} {expiry}")

        ivs = [g["iv"] for g in greeks if g["iv"] > 0]
        if not ivs:
            raise ValueError(f"No valid IVs for {symbol} {expiry}")

        avg_iv = float(np.mean(ivs))

        return VolEstimate(
            value=avg_iv,
            method="iv",
            window=len(nearby_strikes),
            timestamp=datetime.now(),
        )

    async def get_nearest_expiry(self, symbol: str, min_dte: int = 5) -> str:
        """Get the nearest option expiry with at least min_dte days to expiration."""
        chains = await self.get_option_chains(symbol)
        if not chains:
            raise ValueError(f"No option chains for {symbol}")

        today = datetime.now().date()
        for exp_str in chains[0]["expirations"]:
            exp_date = datetime.strptime(exp_str, "%Y%m%d").date()
            dte = (exp_date - today).days
            if dte >= min_dte:
                return exp_str

        raise ValueError(f"No expiry with >= {min_dte} DTE for {symbol}")

    def candles_to_dataframe(self, candles: tuple[Candle, ...]) -> pd.DataFrame:
        """Convert candle tuples to DataFrame for Kronos compatibility."""
        return pd.DataFrame(
            {
                "timestamps": pd.to_datetime([c.timestamp for c in candles]),
                "open": [c.open for c in candles],
                "high": [c.high for c in candles],
                "low": [c.low for c in candles],
                "close": [c.close for c in candles],
                "volume": [c.volume for c in candles],
                "amount": [c.amount for c in candles],
            }
        )
