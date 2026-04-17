# VOLture

**Volatility arbitrage trading agent powered by [Kronos TSFM](https://github.com/bioztar/kronos) and Interactive Brokers.**

VOLture uses a Time Series Foundation Model (TSFM) to predict realized volatility (RV) via Monte Carlo simulation, then compares it against market implied volatility (IV) from IB options chains — exploiting the RV-IV mispricing with targeted options strategies.

---

## How It Works

```
  IB TWS/Gateway
       │
       ▼
 Historical OHLCV candles (30D lookback)
       │
       ▼
 ┌─────────────────────────────┐
 │  Kronos TSFM (MC sampling)  │  ← 50 independent predictions
 │  Predicts full OHLCV ahead  │     each generates an OHLCV path
 └─────────────────────────────┘
       │
       ▼
 Parkinson/GK vol on each MC path
 → average → predicted RV (annualized)
       │
       ▼
 IB Options Chain
 → nearest expiry ATM IV
       │
       ▼
 ┌─────────────────────────────┐
 │   RV vs IV comparison       │
 │   IV >> RV  → sell premium  │
 │   RV >> IV  → buy premium   │
 └─────────────────────────────┘
       │
       ▼
 Strategy selection + strike sizing
 (strikes anchored to Kronos-predicted range)
       │
       ▼
 Risk sizing + signal scoring (0–100)
       │
       ▼
 IB BAG/combo order → paper or live execution
```

Kronos's ability to predict **full OHLCV candles** (not just close prices) makes it uniquely suited for volatility estimation — Parkinson's estimator using the high/low range is ~5× more efficient than close-to-close vol, extracting maximum signal from each prediction.

---

## Strategies

| Market Condition | Vol Signal | Direction | Strategy |
|---|---|---|---|
| Vol overpriced | IV >> RV | Neutral | **Iron Condor** |
| Vol overpriced | IV >> RV | Bullish | **Bull Put Spread** |
| Vol overpriced | IV >> RV | Bearish | **Bear Call Spread** |
| Vol underpriced | RV >> IV | Neutral | **Long Straddle** |
| Vol underpriced | RV >> IV | Bullish | **Bull Call Spread** |
| Vol underpriced | RV >> IV | Bearish | **Bear Put Spread** |

Iron condor short strikes are set at Kronos-predicted ±1σ range, sized to the MC path distribution rather than arbitrary percentages.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12+ | Required |
| [Kronos TSFM](https://github.com/bioztar/kronos) | Cloned as sibling dir `../kronos` |
| IB TWS or IB Gateway | Paper: port `7497` · Live: port `7496` |
| IB market data subscriptions | US Securities Snapshot + Options Add-On (~$15/mo) |

---

## Installation

```bash
# 1. Clone alongside Kronos
git clone https://github.com/bioztar/VOLture
cd VOLture

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install
pip install -e ".[dev]"

# 4. Configure
cp .env.example .env
# Edit .env with your IB connection details and risk limits
```

### IB TWS Setup

1. Open TWS or IB Gateway and log in to your paper account
2. **Edit → Global Configuration → API → Settings**
   - Enable *Active X and Socket Clients*
   - Set *Socket port* to `7497` (paper) / `7496` (live)
   - Uncheck *Read-Only API*
3. Optionally add `127.0.0.1` to *Trusted IPs* to skip manual confirmation prompts

---

## Configuration

Copy `.env.example` to `.env` and adjust:

```env
# IB connection
IB_HOST=127.0.0.1
IB_PORT=7497          # 7497 paper, 7496 live
IB_CLIENT_ID=1

# Kronos repo path (default: sibling directory)
KRONOS_REPO_PATH=../kronos

# Risk limits
MAX_PORTFOLIO_RISK_PCT=2.0        # max % of account at risk across all open positions
MAX_SINGLE_TRADE_RISK_PCT=0.5     # max % of account at risk per trade
MIN_VOL_EDGE_PCT=3.0              # minimum RV-IV gap to generate a signal (percentage points)
MIN_CONFIDENCE=0.6                # minimum signal confidence (0–1)

# Scanner defaults
DEFAULT_TICKERS=SPY,QQQ,IWM,AAPL,MSFT,NVDA,AMZN,TSLA
SCAN_INTERVAL_SECONDS=300
MC_SAMPLES=50
PREDICTION_CANDLES=24
CANDLE_INTERVAL=1h
```

### Recommended candle intervals

| Interval | Use case |
|---|---|
| `5min` | Intraday scalping, high-frequency scanning |
| `15min` | Intraday with more stable signals |
| `1h` | **Default — best balance of signal quality and data availability** |
| `4h` | Swing trading, multi-day vol plays |
| `1d` | Weekly/monthly options, lower noise |

Kronos was trained with temporal embeddings at minute, hour, weekday, day, and month resolution — intervals below `5min` are not recommended.

---

## Usage

### Scan — continuous loop across all configured tickers

```bash
# Paper trading, dry run (default — safe to run anytime)
volture scan

# Custom tickers and interval
volture scan --tickers SPY QQQ NVDA --interval 1h --samples 100

# Single pass, then print results and exit
volture scan --once

# Live trading with execution (requires confirmation prompt)
volture scan --live --execute
```

### Analyze — single ticker deep dive

```bash
volture analyze SPY
volture analyze TSLA --samples 100 --interval 4h --pred 48
volture analyze AAPL --expiry 20250530
```

### Info — show current configuration

```bash
volture info
```

### All options

```
volture scan [OPTIONS]
  -t, --tickers TEXT     Tickers to scan (repeatable, default: SPY QQQ IWM)
  --interval TEXT        Candle interval: 1min 5min 15min 1h 4h 1d (default: 1h)
  --samples INT          Monte Carlo samples (default: 50)
  --pred INT             Prediction candles ahead (default: 24)
  --paper / --live       Paper vs live trading (default: paper)
  --dry-run / --execute  Dry run vs execute orders (default: dry-run)
  --once                 Single scan pass then exit
  --port INT             IB port override

volture analyze TICKER [OPTIONS]
  --expiry TEXT          Option expiry YYYYMMDD (default: nearest)
  --samples INT          Monte Carlo samples (default: 50)
  --interval TEXT        Candle interval (default: 1h)
  --pred INT             Prediction candles ahead (default: 24)
  --port INT             IB port (default: 7497)
```

---

## Architecture

```
volture/
├── types.py              # Immutable frozen dataclasses (Candle, PredictionResult,
│                         #   VolEstimate, VolComparison, OptionStrategy, TradeSignal…)
├── config.py             # IBConfig, KronosConfig, RiskConfig, ScannerConfig
│
├── data/
│   ├── ib_client.py      # Async IB wrapper (prices, candles, option chains, greeks)
│   └── kronos_client.py  # Kronos MC prediction interface (lazy model loading)
│
├── strategy/
│   ├── vol_analyzer.py   # Parkinson/GK vol estimators, RV-IV comparison, confidence
│   └── strike_selector.py# Strategy builder — selects strategy + strikes from VolComparison
│
├── execution/
│   ├── risk.py           # Position sizing, signal scoring (0–100), min-score filter
│   └── orders.py         # IB BAG/combo order construction, place_trade()
│
├── scanner.py            # Main async pipeline: scan_ticker(), run_scan(), analyze_ticker()
├── cli.py                # Click CLI entry point with Rich output tables
└── __main__.py           # python -m volture
```

### Data flow

```
IBClient.get_historical_candles()
    → KronosClient.predict_from_candles()         # 50 MC paths
    → vol_analyzer.predicted_rv_from_mc()         # Parkinson on each path → mean RV
    → IBClient.get_atm_iv()                       # live IV from options chain
    → vol_analyzer.compare_vol()                  # VolComparison with signal + confidence
    → strike_selector.select_strategy()           # OptionStrategy with legs + P&L
    → risk.build_trade_signal()                   # PositionSize + score filter
    → orders.place_trade()                        # IB BAG order (or dry-run log)
```

---

## Signal Scoring

Each trade signal receives a composite score (0–100):

| Component | Weight | Description |
|---|---|---|
| Vol edge magnitude | 40% | Size of RV-IV gap (capped at 10 pp) |
| Confidence | 30% | MC path consistency + edge size |
| Risk/reward ratio | 20% | Strategy max profit / max loss |
| Liquidity | 10% | Placeholder (open interest data coming) |

Signals below `MIN_CONFIDENCE` (default 0.6) or `MIN_SCORE` (default 50) are discarded.

---

## Development

```bash
# Lint
ruff check volture/

# Format
ruff format volture/

# Tests
pytest

# Tests with coverage
pytest --cov=volture --cov-report=term-missing
```

---

## Stack

| Component | Library |
|---|---|
| IB connectivity | [ib_async](https://github.com/ib-api-reloaded/ib_async) — maintained fork of ib_insync |
| TSFM predictions | [Kronos](https://github.com/bioztar/kronos) |
| Numerical / vol math | numpy, pandas |
| Deep learning | PyTorch |
| Options math | py_vollib |
| CLI | Click + Rich |

---

## Disclaimer

**This software is for educational and research purposes only. It is not financial advice. Options trading involves substantial risk of loss and is not suitable for all investors. Always paper trade first. Never risk capital you cannot afford to lose. The authors are not responsible for any trading losses.**

---

## License

MIT — see [LICENSE](LICENSE)
