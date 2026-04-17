"""Volatility analysis: Parkinson/Garman-Klass RV estimation and IV comparison."""

import math
from datetime import datetime

import numpy as np

from volture.types import (
    Candle,
    Direction,
    PredictionResult,
    VolComparison,
    VolEstimate,
    VolSignalType,
)

# Annualization factors by candle interval
ANNUALIZATION_FACTORS = {
    "1min": 252 * 6.5 * 60,    # ~98,280 trading minutes/year
    "5min": 252 * 6.5 * 12,    # ~19,656
    "15min": 252 * 6.5 * 4,    # ~6,552
    "1h": 252 * 6.5,           # ~1,638 trading hours/year
    "4h": 252 * 6.5 / 4,       # ~409.5
    "1d": 252,                  # trading days/year
}

# For 24/7 crypto markets
ANNUALIZATION_FACTORS_CRYPTO = {
    "1min": 365 * 24 * 60,
    "5min": 365 * 24 * 12,
    "15min": 365 * 24 * 4,
    "1h": 365 * 24,
    "4h": 365 * 6,
    "1d": 365,
}


def parkinson_vol(
    candles: tuple[Candle, ...],
    annualization: float = 252 * 6.5,
) -> float:
    """Parkinson volatility estimator using high-low range.

    ~5x more efficient than close-to-close for estimating true vol.
    """
    n = len(candles)
    if n < 2:
        return 0.0

    sum_sq = 0.0
    for c in candles:
        if c.low > 0 and c.high > 0:
            log_hl = math.log(c.high / c.low)
            sum_sq += log_hl * log_hl

    variance = sum_sq / (4 * n * math.log(2))
    return math.sqrt(variance * annualization)


def garman_klass_vol(
    candles: tuple[Candle, ...],
    annualization: float = 252 * 6.5,
) -> float:
    """Garman-Klass volatility estimator using OHLC data.

    More efficient than Parkinson, uses open and close as well.
    """
    n = len(candles)
    if n < 2:
        return 0.0

    total = 0.0
    for c in candles:
        if c.low > 0 and c.high > 0 and c.open > 0 and c.close > 0:
            log_hl = math.log(c.high / c.low)
            log_co = math.log(c.close / c.open)
            total += 0.5 * log_hl * log_hl - (2 * math.log(2) - 1) * log_co * log_co

    variance = total / n
    return math.sqrt(max(0.0, variance) * annualization)


def close_to_close_vol(
    candles: tuple[Candle, ...],
    annualization: float = 252 * 6.5,
) -> float:
    """Standard close-to-close realized volatility."""
    if len(candles) < 3:
        return 0.0

    log_returns = []
    for i in range(1, len(candles)):
        if candles[i - 1].close > 0 and candles[i].close > 0:
            log_returns.append(math.log(candles[i].close / candles[i - 1].close))

    if len(log_returns) < 2:
        return 0.0

    arr = np.array(log_returns)
    return float(np.std(arr, ddof=1) * math.sqrt(annualization))


def predicted_rv_from_mc(
    prediction: PredictionResult,
    interval: str = "1h",
    is_crypto: bool = False,
) -> VolEstimate:
    """Calculate predicted realized vol from Kronos MC paths.

    Uses Parkinson estimator on each MC path, then averages.
    This gives a forward-looking RV estimate.
    """
    factors = ANNUALIZATION_FACTORS_CRYPTO if is_crypto else ANNUALIZATION_FACTORS
    ann_factor = factors.get(interval, 252 * 6.5)

    path_vols = []
    for path in prediction.mc_paths:
        vol = parkinson_vol(path, annualization=ann_factor)
        if vol > 0:
            path_vols.append(vol)

    if not path_vols:
        avg_candles_vol = parkinson_vol(prediction.candles, annualization=ann_factor)
        return VolEstimate(
            value=avg_candles_vol,
            method="parkinson_single",
            window=prediction.prediction_candles,
        )

    mean_vol = float(np.mean(path_vols))

    return VolEstimate(
        value=mean_vol,
        method="parkinson_mc",
        window=prediction.mc_samples,
    )


def compare_vol(
    ticker: str,
    current_price: float,
    predicted_rv: VolEstimate,
    market_iv: VolEstimate,
    prediction: PredictionResult,
    min_edge_pct: float = 3.0,
) -> VolComparison:
    """Compare predicted RV against market IV to generate a vol signal."""
    if market_iv.value <= 0:
        return VolComparison(
            ticker=ticker,
            current_price=current_price,
            predicted_rv=predicted_rv,
            market_iv=market_iv,
            rv_iv_ratio=0.0,
            signal=VolSignalType.NO_SIGNAL,
            confidence=0.0,
            direction=Direction.NEUTRAL,
        )

    rv_iv_ratio = predicted_rv.value / market_iv.value
    edge_pct = abs(predicted_rv.value - market_iv.value) * 100

    # Determine signal
    if edge_pct < min_edge_pct:
        signal = VolSignalType.NO_SIGNAL
    elif rv_iv_ratio < 1.0:
        signal = VolSignalType.VOL_OVERPRICED  # IV > RV → sell premium
    else:
        signal = VolSignalType.VOL_UNDERPRICED  # RV > IV → buy premium

    # Confidence based on edge size and MC consistency
    confidence = _calc_confidence(prediction, rv_iv_ratio, edge_pct)

    # Direction from predicted price movement
    direction = _infer_direction(prediction, current_price)

    return VolComparison(
        ticker=ticker,
        current_price=current_price,
        predicted_rv=predicted_rv,
        market_iv=market_iv,
        rv_iv_ratio=rv_iv_ratio,
        signal=signal,
        confidence=confidence,
        direction=direction,
    )


def _calc_confidence(
    prediction: PredictionResult,
    rv_iv_ratio: float,
    edge_pct: float,
) -> float:
    """Calculate confidence score (0-1) for the vol signal.

    Based on:
    - Size of RV-IV gap (larger = more confident, up to a point)
    - Consistency of MC paths (low variance = more confident)
    """
    # Edge component: sigmoid-like, maxes out around 10% edge
    edge_score = min(1.0, edge_pct / 10.0)

    # MC consistency: std of path vols relative to mean
    if prediction.mc_paths:
        path_vols = []
        for path in prediction.mc_paths:
            vol = parkinson_vol(path)
            if vol > 0:
                path_vols.append(vol)

        if len(path_vols) >= 2:
            cv = float(np.std(path_vols) / np.mean(path_vols))
            consistency_score = max(0.0, 1.0 - cv)
        else:
            consistency_score = 0.3
    else:
        consistency_score = 0.3

    return round(0.6 * edge_score + 0.4 * consistency_score, 3)


def _infer_direction(prediction: PredictionResult, current_price: float) -> Direction:
    """Infer directional bias from predicted price path."""
    if not prediction.candles:
        return Direction.NEUTRAL

    pred_close = prediction.predicted_close
    pct_change = (pred_close - current_price) / current_price * 100

    if pct_change > 0.5:
        return Direction.BULLISH
    elif pct_change < -0.5:
        return Direction.BEARISH
    return Direction.NEUTRAL
