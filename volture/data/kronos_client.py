"""Kronos model client for volatility prediction via MC sampling."""

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from volture.config import KronosConfig
from volture.types import Candle, PredictionResult

logger = logging.getLogger(__name__)

_kronos_loaded = False


def _ensure_kronos_path(repo_path: str) -> None:
    """Add Kronos repo to sys.path if not already present."""
    global _kronos_loaded
    if _kronos_loaded:
        return

    kronos_dir = str(Path(repo_path).resolve())
    if kronos_dir not in sys.path:
        sys.path.insert(0, kronos_dir)
    _kronos_loaded = True


class KronosClient:
    """Interface to Kronos TSFM for Monte Carlo volatility prediction."""

    def __init__(self, config: KronosConfig):
        self._config = config
        self._predictor = None

    def _load_model(self) -> None:
        """Lazy-load Kronos model on first use."""
        if self._predictor is not None:
            return

        _ensure_kronos_path(self._config.repo_path)
        from model import Kronos, KronosPredictor, KronosTokenizer

        logger.info(f"Loading Kronos model: {self._config.model_name}")
        tokenizer = KronosTokenizer.from_pretrained(self._config.tokenizer_name)
        model = Kronos.from_pretrained(self._config.model_name)
        self._predictor = KronosPredictor(
            model, tokenizer, max_context=self._config.max_context
        )
        logger.info(f"Kronos loaded on device: {self._predictor.device}")

    def predict(
        self,
        df: pd.DataFrame,
        pred_len: int | None = None,
        mc_samples: int | None = None,
    ) -> PredictionResult:
        """Run Kronos prediction with MC sampling, returning individual paths.

        Args:
            df: DataFrame with columns [timestamps, open, high, low, close, volume, amount]
            pred_len: Number of candles to predict (default from config)
            mc_samples: Number of MC samples (default from config)
        """
        self._load_model()

        pred_len = pred_len or self._config.prediction_candles
        mc_samples = mc_samples or self._config.mc_samples
        lookback = min(self._config.lookback, len(df) - pred_len)

        if lookback < 50:
            raise ValueError(
                f"Not enough data: {len(df)} candles, need at least {pred_len + 50}"
            )

        x_df = df.iloc[:lookback][["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
        x_timestamp = df.iloc[:lookback]["timestamps"].reset_index(drop=True)

        last_ts = x_timestamp.iloc[-1]
        if len(df) > lookback + pred_len:
            y_timestamp = df.iloc[lookback : lookback + pred_len]["timestamps"].reset_index(drop=True)
        else:
            freq = _infer_freq(x_timestamp)
            y_timestamp = pd.Series(
                [last_ts + freq * (i + 1) for i in range(pred_len)]
            )

        ticker = "unknown"

        logger.info(
            f"Kronos prediction: {lookback} lookback, {pred_len} ahead, "
            f"{mc_samples} MC samples"
        )

        start = time.time()
        mc_paths = []
        all_preds = []

        for i in range(mc_samples):
            pred_df = self._predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=self._config.temperature,
                top_p=self._config.top_p,
                sample_count=1,
                verbose=False,
            )

            path_candles = _df_to_candles(pred_df, y_timestamp)
            mc_paths.append(path_candles)
            all_preds.append(pred_df)

        elapsed = time.time() - start

        avg_df = _average_predictions(all_preds)
        avg_candles = _df_to_candles(avg_df, y_timestamp)

        logger.info(f"Kronos inference: {elapsed:.1f}s ({mc_samples} samples)")

        return PredictionResult(
            ticker=ticker,
            candles=avg_candles,
            mc_paths=tuple(mc_paths),
            lookback_candles=lookback,
            prediction_candles=pred_len,
            mc_samples=mc_samples,
            inference_time_s=elapsed,
        )

    def predict_from_candles(
        self,
        ticker: str,
        candles: tuple[Candle, ...],
        pred_len: int | None = None,
        mc_samples: int | None = None,
    ) -> PredictionResult:
        """Predict from a tuple of Candle objects."""
        df = pd.DataFrame(
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

        result = self.predict(df, pred_len, mc_samples)

        return PredictionResult(
            ticker=ticker,
            candles=result.candles,
            mc_paths=result.mc_paths,
            lookback_candles=result.lookback_candles,
            prediction_candles=result.prediction_candles,
            mc_samples=result.mc_samples,
            inference_time_s=result.inference_time_s,
        )


def _infer_freq(timestamps: pd.Series) -> timedelta:
    """Infer candle frequency from timestamp series."""
    diffs = timestamps.diff().dropna()
    median_diff = diffs.median()
    return median_diff


def _df_to_candles(df: pd.DataFrame, timestamps: pd.Series) -> tuple[Candle, ...]:
    """Convert prediction DataFrame to Candle tuple."""
    candles = []
    for i in range(len(df)):
        ts = timestamps.iloc[i] if i < len(timestamps) else datetime.now()
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        candles.append(
            Candle(
                timestamp=ts,
                open=float(df["open"].iloc[i]),
                high=float(df["high"].iloc[i]),
                low=float(df["low"].iloc[i]),
                close=float(df["close"].iloc[i]),
                volume=float(df.get("volume", pd.Series([0.0] * len(df))).iloc[i]),
                amount=float(df.get("amount", pd.Series([0.0] * len(df))).iloc[i]),
            )
        )
    return tuple(candles)


def _average_predictions(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Average multiple MC prediction DataFrames."""
    cols = ["open", "high", "low", "close"]
    arrays = {col: np.stack([df[col].values for df in dfs]) for col in cols}
    avg = {col: np.mean(arrays[col], axis=0) for col in cols}

    result = pd.DataFrame(avg)

    if "volume" in dfs[0].columns:
        vol_arr = np.stack([df["volume"].values for df in dfs])
        result["volume"] = np.mean(vol_arr, axis=0)

    if "amount" in dfs[0].columns:
        amt_arr = np.stack([df["amount"].values for df in dfs])
        result["amount"] = np.mean(amt_arr, axis=0)

    return result
