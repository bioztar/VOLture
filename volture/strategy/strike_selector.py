"""MC-based option strike selection using Kronos predictions."""

import logging
import math
from datetime import datetime

import numpy as np

from volture.types import (
    Direction,
    OptionLeg,
    OptionStrategy,
    OrderAction,
    Right,
    StrategyType,
    VolComparison,
    VolSignalType,
)

logger = logging.getLogger(__name__)


def select_strategy(
    vol_comp: VolComparison,
    available_strikes: list[float],
    expiry: str,
) -> OptionStrategy | None:
    """Select the best options strategy based on vol comparison signal."""
    if vol_comp.signal == VolSignalType.NO_SIGNAL:
        return None

    if vol_comp.signal == VolSignalType.VOL_OVERPRICED:
        # IV > predicted RV → sell premium
        if vol_comp.direction == Direction.NEUTRAL:
            return build_iron_condor(vol_comp, available_strikes, expiry)
        elif vol_comp.direction == Direction.BULLISH:
            return build_bull_put_spread(vol_comp, available_strikes, expiry)
        else:
            return build_bear_call_spread(vol_comp, available_strikes, expiry)

    if vol_comp.signal == VolSignalType.VOL_UNDERPRICED:
        # predicted RV > IV → buy premium
        if vol_comp.direction == Direction.NEUTRAL:
            return build_long_straddle(vol_comp, available_strikes, expiry)
        elif vol_comp.direction == Direction.BULLISH:
            return build_bull_call_spread_debit(vol_comp, available_strikes, expiry)
        else:
            return build_bear_put_spread_debit(vol_comp, available_strikes, expiry)

    return None


def build_iron_condor(
    vol_comp: VolComparison,
    strikes: list[float],
    expiry: str,
    wing_width_pct: float = 0.02,
) -> OptionStrategy:
    """Build iron condor with strikes based on predicted range.

    Short strikes at ~1 std dev from current price.
    Long strikes 1 strike further out for protection.
    """
    price = vol_comp.current_price
    ticker = vol_comp.ticker

    # Use predicted RV to estimate 1-sigma move
    # Annualized vol → period vol (rough approximation)
    period_vol = vol_comp.predicted_rv.value * 0.1  # ~1 week in annualized terms
    short_put_target = price * (1 - period_vol)
    short_call_target = price * (1 + period_vol)

    short_put = _nearest_strike(strikes, short_put_target)
    short_call = _nearest_strike(strikes, short_call_target)

    # Ensure short strikes are OTM
    if short_put >= price:
        short_put = _nearest_strike_below(strikes, price)
    if short_call <= price:
        short_call = _nearest_strike_above(strikes, price)

    # Wing strikes
    wing_distance = price * wing_width_pct
    long_put = _nearest_strike(strikes, short_put - wing_distance)
    long_call = _nearest_strike(strikes, short_call + wing_distance)

    # Ensure ordering: long_put < short_put < short_call < long_call
    if not (long_put < short_put < short_call < long_call):
        long_put = _nearest_strike_below(strikes, short_put)
        long_call = _nearest_strike_above(strikes, short_call)

    put_width = short_put - long_put
    call_width = long_call - short_call
    max_width = max(put_width, call_width)

    # Estimate credit (rough — actual comes from market data)
    est_credit = max_width * 0.3  # ~30% of width as credit estimate

    legs = (
        OptionLeg(ticker, expiry, long_put, Right.PUT, OrderAction.BUY, 1),
        OptionLeg(ticker, expiry, short_put, Right.PUT, OrderAction.SELL, 1),
        OptionLeg(ticker, expiry, short_call, Right.CALL, OrderAction.SELL, 1),
        OptionLeg(ticker, expiry, long_call, Right.CALL, OrderAction.BUY, 1),
    )

    return OptionStrategy(
        name=StrategyType.IRON_CONDOR,
        ticker=ticker,
        legs=legs,
        max_profit=est_credit * 100,
        max_loss=(max_width - est_credit) * 100,
        breakeven_low=short_put - est_credit,
        breakeven_high=short_call + est_credit,
        net_credit=est_credit,
        vol_comparison=vol_comp,
    )


def build_long_straddle(
    vol_comp: VolComparison,
    strikes: list[float],
    expiry: str,
) -> OptionStrategy:
    """Buy ATM straddle when predicted RV > IV."""
    price = vol_comp.current_price
    ticker = vol_comp.ticker
    atm = _nearest_strike(strikes, price)

    # Rough debit estimate from IV
    dte_years = 7 / 365  # assume ~1 week
    est_premium = price * vol_comp.market_iv.value * math.sqrt(dte_years) * 0.4
    total_debit = est_premium * 2  # call + put

    legs = (
        OptionLeg(ticker, expiry, atm, Right.CALL, OrderAction.BUY, 1),
        OptionLeg(ticker, expiry, atm, Right.PUT, OrderAction.BUY, 1),
    )

    return OptionStrategy(
        name=StrategyType.LONG_STRADDLE,
        ticker=ticker,
        legs=legs,
        max_profit=float("inf"),
        max_loss=total_debit * 100,
        breakeven_low=atm - total_debit,
        breakeven_high=atm + total_debit,
        net_credit=-total_debit,
        vol_comparison=vol_comp,
    )


def build_bull_put_spread(
    vol_comp: VolComparison,
    strikes: list[float],
    expiry: str,
) -> OptionStrategy:
    """Sell put spread (bullish, vol overpriced)."""
    price = vol_comp.current_price
    ticker = vol_comp.ticker

    period_vol = vol_comp.predicted_rv.value * 0.1
    short_put = _nearest_strike(strikes, price * (1 - period_vol * 0.5))
    long_put = _nearest_strike_below(strikes, short_put)

    if short_put >= price:
        short_put = _nearest_strike_below(strikes, price)
        long_put = _nearest_strike_below(strikes, short_put)

    width = short_put - long_put
    est_credit = width * 0.35

    legs = (
        OptionLeg(ticker, expiry, long_put, Right.PUT, OrderAction.BUY, 1),
        OptionLeg(ticker, expiry, short_put, Right.PUT, OrderAction.SELL, 1),
    )

    return OptionStrategy(
        name=StrategyType.BULL_CALL_SPREAD,
        ticker=ticker,
        legs=legs,
        max_profit=est_credit * 100,
        max_loss=(width - est_credit) * 100,
        breakeven_low=short_put - est_credit,
        breakeven_high=None,
        net_credit=est_credit,
        vol_comparison=vol_comp,
    )


def build_bear_call_spread(
    vol_comp: VolComparison,
    strikes: list[float],
    expiry: str,
) -> OptionStrategy:
    """Sell call spread (bearish, vol overpriced)."""
    price = vol_comp.current_price
    ticker = vol_comp.ticker

    period_vol = vol_comp.predicted_rv.value * 0.1
    short_call = _nearest_strike(strikes, price * (1 + period_vol * 0.5))
    long_call = _nearest_strike_above(strikes, short_call)

    if short_call <= price:
        short_call = _nearest_strike_above(strikes, price)
        long_call = _nearest_strike_above(strikes, short_call)

    width = long_call - short_call
    est_credit = width * 0.35

    legs = (
        OptionLeg(ticker, expiry, short_call, Right.CALL, OrderAction.SELL, 1),
        OptionLeg(ticker, expiry, long_call, Right.CALL, OrderAction.BUY, 1),
    )

    return OptionStrategy(
        name=StrategyType.BEAR_PUT_SPREAD,
        ticker=ticker,
        legs=legs,
        max_profit=est_credit * 100,
        max_loss=(width - est_credit) * 100,
        breakeven_low=None,
        breakeven_high=short_call + est_credit,
        net_credit=est_credit,
        vol_comparison=vol_comp,
    )


def build_bull_call_spread_debit(
    vol_comp: VolComparison,
    strikes: list[float],
    expiry: str,
) -> OptionStrategy:
    """Buy call spread (bullish, vol underpriced)."""
    price = vol_comp.current_price
    ticker = vol_comp.ticker

    long_call = _nearest_strike(strikes, price)
    short_call = _nearest_strike_above(strikes, long_call)

    width = short_call - long_call
    est_debit = width * 0.55

    legs = (
        OptionLeg(ticker, expiry, long_call, Right.CALL, OrderAction.BUY, 1),
        OptionLeg(ticker, expiry, short_call, Right.CALL, OrderAction.SELL, 1),
    )

    return OptionStrategy(
        name=StrategyType.BULL_CALL_SPREAD,
        ticker=ticker,
        legs=legs,
        max_profit=(width - est_debit) * 100,
        max_loss=est_debit * 100,
        breakeven_low=long_call + est_debit,
        breakeven_high=None,
        net_credit=-est_debit,
        vol_comparison=vol_comp,
    )


def build_bear_put_spread_debit(
    vol_comp: VolComparison,
    strikes: list[float],
    expiry: str,
) -> OptionStrategy:
    """Buy put spread (bearish, vol underpriced)."""
    price = vol_comp.current_price
    ticker = vol_comp.ticker

    long_put = _nearest_strike(strikes, price)
    short_put = _nearest_strike_below(strikes, long_put)

    width = long_put - short_put
    est_debit = width * 0.55

    legs = (
        OptionLeg(ticker, expiry, long_put, Right.PUT, OrderAction.BUY, 1),
        OptionLeg(ticker, expiry, short_put, Right.PUT, OrderAction.SELL, 1),
    )

    return OptionStrategy(
        name=StrategyType.BEAR_PUT_SPREAD,
        ticker=ticker,
        legs=legs,
        max_profit=(width - est_debit) * 100,
        max_loss=est_debit * 100,
        breakeven_low=None,
        breakeven_high=long_put - est_debit,
        net_credit=-est_debit,
        vol_comparison=vol_comp,
    )


# --- Strike helpers ---

def _nearest_strike(strikes: list[float], target: float) -> float:
    return min(strikes, key=lambda s: abs(s - target))


def _nearest_strike_above(strikes: list[float], target: float) -> float:
    above = [s for s in strikes if s > target]
    if not above:
        return max(strikes)
    return min(above)


def _nearest_strike_below(strikes: list[float], target: float) -> float:
    below = [s for s in strikes if s < target]
    if not below:
        return min(strikes)
    return max(below)
