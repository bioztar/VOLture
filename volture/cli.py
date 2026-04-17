"""CLI entry point for VOLture."""

import asyncio
import logging
import sys

import click
from rich.console import Console
from rich.table import Table

from volture.config import Config, IBConfig, KronosConfig, RiskConfig, ScannerConfig, load_config
from volture.scanner import analyze_ticker, run_scan

console = Console()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy loggers
    logging.getLogger("ib_async").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """VOLture - Kronos RV-IV volatility arbitrage trading agent."""
    setup_logging(verbose)


@cli.command()
@click.option("--tickers", "-t", multiple=True, default=["SPY", "QQQ", "IWM"])
@click.option("--interval", default="1h", help="Candle interval (1min, 5min, 15min, 1h, 4h, 1d)")
@click.option("--samples", default=50, help="Monte Carlo samples")
@click.option("--pred", default=24, help="Prediction candles ahead")
@click.option("--paper/--live", default=True, help="Paper trading (default) or live")
@click.option("--dry-run/--execute", default=True, help="Dry run (default) or execute trades")
@click.option("--once", is_flag=True, help="Single scan pass, then exit")
@click.option("--port", default=None, type=int, help="IB port override")
def scan(
    tickers: tuple[str, ...],
    interval: str,
    samples: int,
    pred: int,
    paper: bool,
    dry_run: bool,
    once: bool,
    port: int | None,
) -> None:
    """Scan tickers for RV-IV arbitrage opportunities."""
    ib_port = port or (7497 if paper else 7496)

    config = Config(
        ib=IBConfig(port=ib_port),
        kronos=KronosConfig(mc_samples=samples, prediction_candles=pred),
        scanner=ScannerConfig(tickers=tickers, candle_interval=interval),
    )

    mode = "PAPER" if paper else "LIVE"
    action = "DRY RUN" if dry_run else "EXECUTING"

    console.print(f"\n[bold]VOLture Scanner[/bold]")
    console.print(f"  Mode:     {mode} ({action})")
    console.print(f"  Tickers:  {', '.join(tickers)}")
    console.print(f"  Interval: {interval}")
    console.print(f"  MC:       {samples} samples, {pred} candles ahead")
    console.print(f"  IB Port:  {ib_port}\n")

    if not paper and not dry_run:
        if not click.confirm(
            "WARNING: Live trading with order execution enabled. Continue?",
            default=False,
        ):
            console.print("[red]Aborted.[/red]")
            return

    signals = asyncio.run(run_scan(config, dry_run=dry_run, single_pass=once))

    if once and signals:
        _print_signals_table(signals)


@cli.command()
@click.argument("ticker")
@click.option("--expiry", default=None, help="Option expiry (YYYYMMDD)")
@click.option("--samples", default=50, help="Monte Carlo samples")
@click.option("--interval", default="1h", help="Candle interval")
@click.option("--pred", default=24, help="Prediction candles ahead")
@click.option("--port", default=7497, type=int, help="IB port")
def analyze(
    ticker: str,
    expiry: str | None,
    samples: int,
    interval: str,
    pred: int,
    port: int,
) -> None:
    """Analyze a single ticker for vol opportunities."""
    config = Config(
        ib=IBConfig(port=port),
        kronos=KronosConfig(mc_samples=samples, prediction_candles=pred),
        scanner=ScannerConfig(tickers=(ticker,), candle_interval=interval),
    )

    console.print(f"\n[bold]VOLture Analysis: {ticker}[/bold]")
    console.print(f"  MC samples:  {samples}")
    console.print(f"  Interval:    {interval}")
    console.print(f"  Pred length: {pred} candles\n")

    signal = asyncio.run(analyze_ticker(ticker, config, expiry))

    if signal:
        _print_signals_table([signal])
    else:
        console.print(f"[yellow]No actionable signal for {ticker}[/yellow]")


@cli.command()
def info() -> None:
    """Show VOLture configuration and status."""
    config = load_config()

    console.print("\n[bold]VOLture Configuration[/bold]\n")

    table = Table(show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("IB Host", f"{config.ib.host}:{config.ib.port}")
    table.add_row("IB Mode", "PAPER" if config.ib.is_paper else "LIVE")
    table.add_row("Kronos Model", config.kronos.model_name)
    table.add_row("Kronos Path", config.kronos.repo_path)
    table.add_row("MC Samples", str(config.kronos.mc_samples))
    table.add_row("Prediction", f"{config.kronos.prediction_candles} candles")
    table.add_row("Tickers", ", ".join(config.scanner.tickers))
    table.add_row("Interval", config.scanner.candle_interval)
    table.add_row("Max Position", str(config.risk.max_position_size))
    table.add_row("Max Risk/Trade", f"{config.risk.max_single_trade_risk_pct}%")
    table.add_row("Min Vol Edge", f"{config.risk.min_vol_edge_pct}%")
    table.add_row("Min Confidence", str(config.risk.min_confidence))

    console.print(table)


def _print_signals_table(signals: list) -> None:
    """Pretty-print trade signals as a table."""
    table = Table(title="Trade Signals")
    table.add_column("Ticker", style="cyan")
    table.add_column("Strategy", style="green")
    table.add_column("Score", justify="right")
    table.add_column("RV%", justify="right")
    table.add_column("IV%", justify="right")
    table.add_column("Edge%", justify="right")
    table.add_column("Direction")
    table.add_column("Contracts", justify="right")
    table.add_column("Max Loss", justify="right", style="red")

    for sig in signals:
        vc = sig.vol_comparison
        edge_style = "green" if vc.vol_edge_pct > 0 else "red"
        table.add_row(
            vc.ticker,
            sig.strategy.name.name,
            f"{sig.score:.0f}",
            f"{vc.predicted_rv.pct:.1f}",
            f"{vc.market_iv.pct:.1f}",
            f"[{edge_style}]{vc.vol_edge_pct:+.1f}[/{edge_style}]",
            vc.direction.name,
            str(sig.position_size.contracts),
            f"${sig.position_size.max_loss:,.0f}",
        )

    console.print(table)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
