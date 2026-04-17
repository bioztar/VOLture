"""Main scanner loop: scan tickers -> predict vol -> compare IV -> generate signals."""

import asyncio
import logging
from datetime import datetime

from volture.config import Config
from volture.data.ib_client import IBClient
from volture.data.kronos_client import KronosClient
from volture.execution.orders import place_trade
from volture.execution.risk import build_trade_signal
from volture.strategy.strike_selector import select_strategy
from volture.strategy.vol_analyzer import compare_vol, predicted_rv_from_mc
from volture.types import TradeSignal

logger = logging.getLogger(__name__)


async def scan_ticker(
    ticker: str,
    ib: IBClient,
    kronos: KronosClient,
    config: Config,
) -> TradeSignal | None:
    """Run full analysis pipeline for a single ticker."""
    try:
        logger.info(f"--- Scanning {ticker} ---")

        # 1. Get current price and historical candles from IB
        price = await ib.get_stock_price(ticker)
        candles = await ib.get_historical_candles(
            ticker,
            duration="30 D",
            bar_size=_interval_to_ib_bar(config.scanner.candle_interval),
        )

        if len(candles) < 100:
            logger.warning(f"{ticker}: only {len(candles)} candles — skipping")
            return None

        # 2. Run Kronos MC prediction
        prediction = kronos.predict_from_candles(
            ticker=ticker,
            candles=candles,
            pred_len=config.kronos.prediction_candles,
            mc_samples=config.kronos.mc_samples,
        )

        is_crypto = ticker.endswith("-USD") and not ticker.startswith("^")

        # 3. Calculate predicted RV from MC paths
        predicted_rv = predicted_rv_from_mc(
            prediction,
            interval=config.scanner.candle_interval,
            is_crypto=is_crypto,
        )

        # 4. Get market IV from IB options chain
        expiry = await ib.get_nearest_expiry(ticker, min_dte=5)
        market_iv = await ib.get_atm_iv(ticker, expiry)

        # 5. Compare RV vs IV
        vol_comp = compare_vol(
            ticker=ticker,
            current_price=price,
            predicted_rv=predicted_rv,
            market_iv=market_iv,
            prediction=prediction,
            min_edge_pct=config.risk.min_vol_edge_pct,
        )

        logger.info(
            f"{ticker}: price=${price:,.2f} | "
            f"predicted RV={predicted_rv.pct:.1f}% | "
            f"market IV={market_iv.pct:.1f}% | "
            f"ratio={vol_comp.rv_iv_ratio:.2f} | "
            f"signal={vol_comp.signal.name} | "
            f"direction={vol_comp.direction.name}"
        )

        if vol_comp.signal.name == "NO_SIGNAL":
            return None

        # 6. Select strategy and strikes
        chains = await ib.get_option_chains(ticker)
        if not chains:
            logger.warning(f"{ticker}: no option chains available")
            return None

        available_strikes = chains[0]["strikes"]
        strategy = select_strategy(vol_comp, available_strikes, expiry)

        if strategy is None:
            return None

        # 7. Size and score
        # TODO: get actual account value from IB
        account_value = 100_000.0  # placeholder
        signal = build_trade_signal(vol_comp, strategy, account_value, config.risk)

        return signal

    except Exception as e:
        logger.error(f"{ticker}: scan failed — {e}", exc_info=True)
        return None


async def run_scan(
    config: Config,
    dry_run: bool = True,
    single_pass: bool = False,
) -> list[TradeSignal]:
    """Run the scanner loop across all configured tickers."""
    ib = IBClient(config.ib)
    kronos = KronosClient(config.kronos)

    await ib.connect()

    try:
        while True:
            logger.info(f"=== Scan started at {datetime.now().isoformat()} ===")
            signals: list[TradeSignal] = []

            for ticker in config.scanner.tickers:
                signal = await scan_ticker(ticker, ib, kronos, config)
                if signal is not None:
                    signals.append(signal)

            # Sort by score descending
            signals.sort(key=lambda s: s.score, reverse=True)

            if signals:
                logger.info(f"\n=== {len(signals)} signals found ===")
                for sig in signals:
                    logger.info(
                        f"  {sig.vol_comparison.ticker}: "
                        f"{sig.strategy.name.name} | "
                        f"score={sig.score} | "
                        f"edge={sig.vol_comparison.vol_edge_pct:+.1f}% | "
                        f"contracts={sig.position_size.contracts}"
                    )

                    await place_trade(ib._ib, sig, dry_run=dry_run)
            else:
                logger.info("No actionable signals this scan")

            if single_pass:
                return signals

            logger.info(
                f"Next scan in {config.scanner.scan_interval_seconds}s..."
            )
            await asyncio.sleep(config.scanner.scan_interval_seconds)

    finally:
        await ib.disconnect()


async def analyze_ticker(
    ticker: str,
    config: Config,
    expiry: str | None = None,
) -> TradeSignal | None:
    """One-shot analysis of a single ticker (for CLI 'analyze' command)."""
    ib = IBClient(config.ib)
    kronos = KronosClient(config.kronos)

    await ib.connect()

    try:
        signal = await scan_ticker(ticker, ib, kronos, config)
        return signal
    finally:
        await ib.disconnect()


def _interval_to_ib_bar(interval: str) -> str:
    """Convert config interval string to IB bar size."""
    mapping = {
        "1min": "1 min",
        "5min": "5 mins",
        "15min": "15 mins",
        "1h": "1 hour",
        "4h": "4 hours",
        "1d": "1 day",
    }
    return mapping.get(interval, "1 hour")
