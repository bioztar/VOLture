# VOLture - Kronos RV-IV Trading Strategy Harness

## Project Overview
VOLture uses the Kronos time-series foundation model to predict realized volatility
and compares it against market implied volatility to generate options trading signals.
Executes via Interactive Brokers API.

## Architecture
- `volture/data/` - Data layer (IB client, Kronos predictions)
- `volture/strategy/` - Strategy logic (vol analysis, strike selection, signals)
- `volture/execution/` - Order execution and risk management
- `volture/scanner.py` - Main async scan loop
- `volture/cli.py` - CLI entry point

## Stack
- Python 3.12, async throughout
- `ib_async` for Interactive Brokers (fork of ib_insync)
- Kronos-base model (from sibling /DEV/kronos directory)
- Immutable frozen dataclasses for all data types

## Commands
```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Run scanner (paper trading)
python -m volture scan --tickers SPY QQQ AAPL --paper

# Single ticker analysis
python -m volture analyze --ticker SPY --expiry 2026-04-24

# Dry run (no orders)
python -m volture scan --tickers SPY --dry-run
```

## Conventions
- All data types are frozen dataclasses (immutable)
- Async/await for all IB operations
- No mutation of existing objects
- Config via .env file or CLI args
- Paper trading mode by default (must explicitly enable live)
