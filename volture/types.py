"""Immutable data types for VOLture."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto


class Direction(Enum):
    BULLISH = auto()
    BEARISH = auto()
    NEUTRAL = auto()


class VolSignalType(Enum):
    VOL_OVERPRICED = auto()   # IV > predicted RV → sell premium
    VOL_UNDERPRICED = auto()  # IV < predicted RV → buy premium
    NO_SIGNAL = auto()


class StrategyType(Enum):
    IRON_CONDOR = auto()
    SHORT_STRANGLE = auto()
    LONG_STRADDLE = auto()
    LONG_STRANGLE = auto()
    BULL_CALL_SPREAD = auto()
    BEAR_PUT_SPREAD = auto()
    COVERED_CALL = auto()
    CASH_SECURED_PUT = auto()
    BUTTERFLY = auto()


class OrderAction(Enum):
    BUY = "BUY"
    SELL = "SELL"


class Right(Enum):
    CALL = "C"
    PUT = "P"


@dataclass(frozen=True)
class Candle:
    """Single OHLCV candle."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float = 0.0


@dataclass(frozen=True)
class PredictionResult:
    """Result from Kronos MC prediction."""

    ticker: str
    candles: tuple[Candle, ...]
    mc_paths: tuple[tuple[Candle, ...], ...]  # individual MC sample paths
    lookback_candles: int
    prediction_candles: int
    mc_samples: int
    inference_time_s: float

    @property
    def predicted_high(self) -> float:
        return max(c.high for c in self.candles)

    @property
    def predicted_low(self) -> float:
        return min(c.low for c in self.candles)

    @property
    def predicted_close(self) -> float:
        return self.candles[-1].close if self.candles else 0.0

    @property
    def predicted_range_pct(self) -> float:
        mid = (self.predicted_high + self.predicted_low) / 2
        if mid == 0:
            return 0.0
        return (self.predicted_high - self.predicted_low) / mid * 100


@dataclass(frozen=True)
class VolEstimate:
    """Volatility estimate from a specific method."""

    value: float  # annualized vol as decimal (e.g. 0.25 = 25%)
    method: str   # "parkinson", "garman_klass", "close_to_close", "iv"
    window: int   # number of candles used
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def pct(self) -> float:
        return self.value * 100


@dataclass(frozen=True)
class VolComparison:
    """Comparison between predicted RV and market IV."""

    ticker: str
    current_price: float
    predicted_rv: VolEstimate
    market_iv: VolEstimate
    rv_iv_ratio: float  # predicted_rv / market_iv
    signal: VolSignalType
    confidence: float   # 0.0 to 1.0
    direction: Direction
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def vol_edge_pct(self) -> float:
        """How much RV differs from IV in percentage points."""
        return (self.predicted_rv.value - self.market_iv.value) * 100


@dataclass(frozen=True)
class OptionLeg:
    """Single option contract leg."""

    ticker: str
    expiry: str        # YYYYMMDD
    strike: float
    right: Right
    action: OrderAction
    quantity: int


@dataclass(frozen=True)
class OptionStrategy:
    """Complete options strategy with legs."""

    name: StrategyType
    ticker: str
    legs: tuple[OptionLeg, ...]
    max_profit: float
    max_loss: float
    breakeven_low: float | None
    breakeven_high: float | None
    net_credit: float  # positive = credit, negative = debit
    vol_comparison: VolComparison

    @property
    def risk_reward_ratio(self) -> float:
        if self.max_loss == 0:
            return float("inf")
        return abs(self.max_profit / self.max_loss)


@dataclass(frozen=True)
class PositionSize:
    """Calculated position sizing."""

    contracts: int
    capital_required: float
    max_loss: float
    portfolio_risk_pct: float
    reason: str


@dataclass(frozen=True)
class TradeSignal:
    """Complete trade signal with strategy and sizing."""

    vol_comparison: VolComparison
    strategy: OptionStrategy
    position_size: PositionSize
    score: float  # composite score 0-100
    timestamp: datetime = field(default_factory=datetime.now)
