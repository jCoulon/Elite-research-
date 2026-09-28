"""Vectorised daily portfolio backtester.

Conventions (chosen to be conservative):

* ``target`` weights at date t are computed with information available at the
  close of t.
* With ``lag=1`` (default) the trade is executed at the close of t+1, so the
  position first earns the return of t+2. ``lag=0`` assumes a market-on-close
  order filled at the same close the signal was computed on (optimistic).
* Any weight not allocated to risky assets (1 - sum(w)) earns the T-bill rate.
  Gross exposure above 1 is financed at T-bill + ``borrow_spread``.
* Costs are charged on one-way turnover: ``cost_bps`` per unit of traded
  notional (covers commission, half spread and slippage).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series          # net daily returns
    gross_returns: pd.Series    # before costs
    weights: pd.DataFrame       # held weights (post-lag)
    turnover: pd.Series         # daily one-way turnover
    costs: pd.Series            # daily cost drag
    trades: pd.DataFrame | None = None  # round trips for single-asset timing strategies

    @property
    def equity(self) -> pd.Series:
        return (1 + self.returns).cumprod()


def run_backtest(
    name: str,
    target: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cash: pd.Series,
    cost_bps: float = 5.0,
    lag: int = 1,
    borrow_spread: float = 0.01,
    start=None,
    end=None,
) -> BacktestResult:
    target = target.reindex(asset_returns.index).ffill().fillna(0.0)
    rets = asset_returns[target.columns].fillna(0.0)
    held = target.shift(1 + lag).fillna(0.0)

    if start is not None or end is not None:
        sl = slice(start, end)
        held, rets, cash_ = held.loc[sl], rets.loc[sl], cash.loc[sl]
    else:
        cash_ = cash

    gross_exposure = held.sum(axis=1)
    cash_w = 1.0 - gross_exposure
    borrow = np.where(cash_w < 0, cash_w * (cash_ + borrow_spread / 252), cash_w * cash_)
    gross = (held * rets).sum(axis=1) + borrow

    # Turnover: today's held weights vs yesterday's weights after drifting with
    # yesterday's returns (i.e. the pre-trade book at yesterday's close).
    drifted = held.shift(1) * (1 + rets.shift(1))
    drifted = drifted.div((1 + gross.shift(1)).replace(0, np.nan), axis=0).fillna(0.0)
    turnover = (held - drifted).abs().sum(axis=1)
    # the first row trades in from cash
    turnover.iloc[0] = held.iloc[0].abs().sum()
    costs = turnover * cost_bps / 1e4
    net = gross - costs
    return BacktestResult(name, net.rename(name), gross.rename(name), held, turnover, costs)


def round_trips(position: pd.Series, asset_ret: pd.Series, cost_bps: float) -> pd.DataFrame:
    """Round-trip trade list for a single-asset 0/x position series (held weights)."""
    pos = position.fillna(0.0)
    in_trade = pos > 0
    trade_id = (in_trade & ~in_trade.shift(1, fill_value=False)).cumsum().where(in_trade)
    rows = []
    for tid, grp in asset_ret[in_trade].groupby(trade_id[in_trade]):
        w = pos.loc[grp.index]
        r = float(np.prod(1 + w * grp) - 1) - 2 * cost_bps / 1e4 * float(w.mean())
        rows.append({"entry": grp.index[0], "exit": grp.index[-1], "days": len(grp), "return": r})
    return pd.DataFrame(rows)
