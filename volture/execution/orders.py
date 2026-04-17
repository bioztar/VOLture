"""Order construction and execution via IB."""

import logging
from datetime import datetime

from ib_async import (
    ComboLeg,
    Contract,
    IB,
    LimitOrder,
    Option,
    Order,
    Trade,
)

from volture.types import OptionLeg, OptionStrategy, OrderAction, Right, TradeSignal

logger = logging.getLogger(__name__)


def build_combo_contract(strategy: OptionStrategy) -> tuple[Contract, list[Contract]]:
    """Build IB combo (BAG) contract from strategy legs."""
    leg_contracts = []
    for leg in strategy.legs:
        opt = Option(
            symbol=leg.ticker,
            lastTradeDateOrContractMonth=leg.expiry,
            strike=leg.strike,
            right=leg.right.value,
            exchange="SMART",
            currency="USD",
        )
        leg_contracts.append(opt)

    return leg_contracts


async def qualify_and_build_combo(
    ib: IB,
    strategy: OptionStrategy,
) -> tuple[Contract, Order] | None:
    """Qualify option contracts and build a combo order.

    Returns (combo_contract, order) or None if qualification fails.
    """
    leg_contracts = build_combo_contract(strategy)

    qualified = await ib.qualifyContractsAsync(*leg_contracts)
    valid = [c for c in qualified if c.conId > 0]

    if len(valid) != len(strategy.legs):
        logger.error(
            f"Failed to qualify all legs: {len(valid)}/{len(strategy.legs)} valid"
        )
        return None

    # Build BAG contract
    bag = Contract()
    bag.symbol = strategy.ticker
    bag.secType = "BAG"
    bag.exchange = "SMART"
    bag.currency = "USD"

    combo_legs = []
    for i, (leg, contract) in enumerate(zip(strategy.legs, valid)):
        cl = ComboLeg()
        cl.conId = contract.conId
        cl.ratio = leg.quantity
        cl.action = leg.action.value
        cl.exchange = "SMART"
        combo_legs.append(cl)

    bag.comboLegs = combo_legs

    # Limit order at the net credit/debit
    action = "BUY" if strategy.net_credit < 0 else "SELL"
    limit_price = abs(strategy.net_credit)

    order = LimitOrder(
        action=action,
        totalQuantity=1,  # will be multiplied by position size
        lmtPrice=round(limit_price, 2),
        tif="DAY",
        transmit=False,  # safety: don't auto-transmit
    )

    return bag, order


async def place_trade(
    ib: IB,
    signal: TradeSignal,
    dry_run: bool = True,
) -> Trade | None:
    """Place a trade from a signal, or log it if dry_run."""
    strategy = signal.strategy

    logger.info(
        f"{'[DRY RUN] ' if dry_run else ''}"
        f"Trade: {strategy.name.name} on {strategy.ticker} | "
        f"Score: {signal.score} | "
        f"Contracts: {signal.position_size.contracts} | "
        f"Max Loss: ${signal.position_size.max_loss:,.0f}"
    )

    for leg in strategy.legs:
        logger.info(
            f"  {leg.action.value} {leg.quantity} "
            f"{leg.ticker} {leg.expiry} "
            f"${leg.strike} {leg.right.value}"
        )

    if dry_run:
        logger.info("[DRY RUN] Order not submitted")
        return None

    result = await qualify_and_build_combo(ib, strategy)
    if result is None:
        logger.error("Failed to build combo order — not placing trade")
        return None

    bag, order = result
    order.totalQuantity = signal.position_size.contracts

    trade = ib.placeOrder(bag, order)
    logger.info(
        f"Order placed: {trade.order.orderId} | "
        f"Status: {trade.orderStatus.status}"
    )

    return trade
