"""Configuration management for VOLture."""

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True)
class IBConfig:
    """Interactive Brokers connection settings."""

    host: str = "127.0.0.1"
    port: int = 7497  # 7497=paper, 7496=live
    client_id: int = 1

    @property
    def is_paper(self) -> bool:
        return self.port == 7497


@dataclass(frozen=True)
class KronosConfig:
    """Kronos model settings."""

    repo_path: str = "../kronos"
    model_name: str = "NeoQuasar/Kronos-base"
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base"
    max_context: int = 512
    mc_samples: int = 50
    prediction_candles: int = 24
    temperature: float = 1.0
    top_p: float = 0.9
    lookback: int = 400


@dataclass(frozen=True)
class RiskConfig:
    """Risk management settings."""

    max_position_size: int = 1000
    max_portfolio_risk_pct: float = 2.0
    max_single_trade_risk_pct: float = 0.5
    min_vol_edge_pct: float = 3.0       # minimum RV-IV difference to trigger signal
    min_confidence: float = 0.6
    min_score: float = 50.0


@dataclass(frozen=True)
class ScannerConfig:
    """Scanner loop settings."""

    tickers: tuple[str, ...] = ("SPY", "QQQ", "IWM")
    scan_interval_seconds: int = 300
    candle_interval: str = "1h"
    data_period: str = "30d"
    dry_run: bool = True


@dataclass(frozen=True)
class Config:
    """Root configuration."""

    ib: IBConfig = field(default_factory=IBConfig)
    kronos: KronosConfig = field(default_factory=KronosConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)


def load_config(env_path: Path | None = None) -> Config:
    """Load configuration from .env file and defaults."""
    env = dotenv_values(env_path) if env_path else dotenv_values()

    tickers_str = env.get("DEFAULT_TICKERS", "SPY,QQQ,IWM")
    tickers = tuple(t.strip() for t in tickers_str.split(","))

    return Config(
        ib=IBConfig(
            host=env.get("IB_HOST", "127.0.0.1"),
            port=int(env.get("IB_PORT", "7497")),
            client_id=int(env.get("IB_CLIENT_ID", "1")),
        ),
        kronos=KronosConfig(
            repo_path=env.get("KRONOS_REPO_PATH", "../kronos"),
            mc_samples=int(env.get("MC_SAMPLES", "50")),
            prediction_candles=int(env.get("PREDICTION_CANDLES", "24")),
        ),
        risk=RiskConfig(
            max_position_size=int(env.get("MAX_POSITION_SIZE", "1000")),
            max_portfolio_risk_pct=float(env.get("MAX_PORTFOLIO_RISK_PCT", "2.0")),
            max_single_trade_risk_pct=float(env.get("MAX_SINGLE_TRADE_RISK_PCT", "0.5")),
        ),
        scanner=ScannerConfig(
            tickers=tickers,
            scan_interval_seconds=int(env.get("SCAN_INTERVAL_SECONDS", "300")),
            candle_interval=env.get("CANDLE_INTERVAL", "1h"),
        ),
    )
