"""Position sizing and risk management."""

import logging

from volture.config import RiskConfig
from volture.types import OptionStrategy, PositionSize, TradeSignal, VolComparison

logger = logging.getLogger(__name__)


def calculate_position_size(
    strategy: OptionStrategy,
    account_value: float,
    config: RiskConfig,
) -> PositionSize:
    """Calculate position size based on risk limits."""
    if strategy.max_loss <= 0:
        return PositionSize(
            contracts=0,
            capital_required=0.0,
            max_loss=0.0,
            portfolio_risk_pct=0.0,
            reason="Strategy has zero or negative max loss — skipping",
        )

    max_risk_dollars = account_value * (config.max_single_trade_risk_pct / 100)
    max_contracts_by_risk = int(max_risk_dollars / strategy.max_loss)
    contracts = min(max_contracts_by_risk, config.max_position_size)
    contracts = max(contracts, 0)

    actual_max_loss = contracts * strategy.max_loss
    portfolio_risk_pct = (actual_max_loss / account_value * 100) if account_value > 0 else 0.0

    if portfolio_risk_pct > config.max_portfolio_risk_pct:
        contracts = int(
            (account_value * config.max_portfolio_risk_pct / 100) / strategy.max_loss
        )
        actual_max_loss = contracts * strategy.max_loss
        portfolio_risk_pct = (actual_max_loss / account_value * 100) if account_value > 0 else 0.0

    capital_required = actual_max_loss  # margin requirement approximation

    reason = (
        f"{contracts} contracts, "
        f"max loss ${actual_max_loss:,.0f} "
        f"({portfolio_risk_pct:.1f}% of ${account_value:,.0f})"
    )

    return PositionSize(
        contracts=contracts,
        capital_required=capital_required,
        max_loss=actual_max_loss,
        portfolio_risk_pct=portfolio_risk_pct,
        reason=reason,
    )


def score_signal(vol_comp: VolComparison, strategy: OptionStrategy) -> float:
    """Score a trade signal from 0-100 for ranking.

    Components:
    - Vol edge magnitude (40%)
    - Confidence from MC consistency (30%)
    - Risk/reward ratio (20%)
    - Liquidity proxy from bid-ask (10%, placeholder)
    """
    # Vol edge: bigger gap = better, capped at 15%
    edge = abs(vol_comp.vol_edge_pct)
    edge_score = min(edge / 15.0, 1.0) * 40

    # Confidence from vol comparison
    confidence_score = vol_comp.confidence * 30

    # Risk/reward
    rr = strategy.risk_reward_ratio
    rr_score = min(rr / 2.0, 1.0) * 20  # 2:1 R:R gets full marks

    # Liquidity placeholder (would use bid-ask spread in production)
    liquidity_score = 5.0  # default to 50% of max

    total = edge_score + confidence_score + rr_score + liquidity_score
    return round(min(total, 100.0), 1)


def build_trade_signal(
    vol_comp: VolComparison,
    strategy: OptionStrategy,
    account_value: float,
    config: RiskConfig,
) -> TradeSignal | None:
    """Build a complete trade signal with sizing, or None if it fails filters."""
    if vol_comp.confidence < config.min_confidence:
        logger.info(
            f"{vol_comp.ticker}: confidence {vol_comp.confidence:.2f} "
            f"< min {config.min_confidence} — skipping"
        )
        return None

    score = score_signal(vol_comp, strategy)
    if score < config.min_score:
        logger.info(
            f"{vol_comp.ticker}: score {score:.1f} < min {config.min_score} — skipping"
        )
        return None

    position_size = calculate_position_size(strategy, account_value, config)
    if position_size.contracts == 0:
        logger.info(f"{vol_comp.ticker}: position size is 0 — skipping")
        return None

    return TradeSignal(
        vol_comparison=vol_comp,
        strategy=strategy,
        position_size=position_size,
        score=score,
    )
