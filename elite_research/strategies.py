"""Strategy signal generators.

Every function returns a DataFrame of *target* weights (dates x assets) using
only information available at each date's close. The backtester applies the
execution lag, costs and cash accrual. Weights may sum to < 1 (rest in T-bills).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CROSS_ASSET, SECTORS


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)


def _cash_growth(cash: pd.Series) -> pd.Series:
    return (1 + cash).cumprod()


# --------------------------------------------------------------------------- #
# S1. Cross-asset time-series trend following, inverse-vol weighted
# --------------------------------------------------------------------------- #
def cross_asset_trend(close: pd.DataFrame, cash: pd.Series,
                      lookbacks=(21, 63, 126, 252), vol_window: int = 63,
                      universe=tuple(CROSS_ASSET)) -> pd.DataFrame:
    """Hold each asset in proportion to the fraction of lookbacks on which its
    total return beat T-bills; size positions by inverse realised volatility
    (risk-parity style) across the eligible universe. Monthly rebalance.

    Thesis: time-series momentum (Moskowitz, Ooi & Pedersen 2012; Hurst, Ooi &
    Pedersen 2017) - slow-moving capital, under-reaction to news and hedging
    flows create persistent trends that have held across a century of data.
    """
    px = close[list(universe)]
    rets = px.pct_change(fill_method=None)
    cg = _cash_growth(cash.reindex(px.index).fillna(0))
    vol = rets.rolling(vol_window).std() * np.sqrt(252)
    score = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for lb in lookbacks:
        excess = px.pct_change(lb, fill_method=None).sub(cg.pct_change(lb), axis=0)
        score += (excess > 0).astype(float)
    score /= len(lookbacks)
    eligible = px.notna() & px.shift(max(lookbacks)).notna() & vol.notna()
    inv = (1 / vol).where(eligible)
    risk_w = inv.div(inv.sum(axis=1), axis=0)
    w = (risk_w * score).fillna(0.0)
    me = month_ends(px.index)
    return w.loc[me].reindex(px.index).ffill().fillna(0.0)


# --------------------------------------------------------------------------- #
# S2. Sector relative-strength rotation
# --------------------------------------------------------------------------- #
def sector_momentum(close: pd.DataFrame, cash: pd.Series, lookback: int = 252,
                    skip: int = 21, top_n: int = 3, trend_filter: bool = False,
                    universe=tuple(SECTORS)) -> pd.DataFrame:
    """Equal-weight the top-N sector ETFs by (lookback, skip) momentum, monthly.

    Thesis: industry momentum (Moskowitz & Grinblatt 1999) - information about
    industry-wide demand shocks diffuses slowly and institutional flows chase
    winning sectors. Optional filter moves to T-bills when SPY < 10-month SMA.
    """
    px = close[list(universe)]
    mom = px.shift(skip) / px.shift(lookback) - 1
    mom = mom.where(px.notna() & px.shift(lookback).notna())
    rank = mom.rank(axis=1, ascending=False)
    w = (rank <= top_n).astype(float)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if trend_filter:
        spy = close["SPY"]
        on = (spy > spy.rolling(210).mean()).astype(float)
        w = w.mul(on, axis=0)
    me = month_ends(px.index)
    return w.loc[me].reindex(px.index).ffill().fillna(0.0)


# --------------------------------------------------------------------------- #
# S3. Short-term mean reversion (RSI-2) in an uptrend
# --------------------------------------------------------------------------- #
def rsi(series: pd.Series, n: int) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def rsi_mean_reversion(close: pd.DataFrame, cash: pd.Series, ticker: str = "SPY",
                       rsi_n: int = 2, entry: float = 10, exit_sma: int = 5,
                       regime_sma: int = 200, max_hold: int = 10) -> pd.DataFrame:
    """Buy after a sharp short-term sell-off (RSI(n) < entry) while the index is
    above its long-term average; exit when price closes above its short SMA or
    after ``max_hold`` days.

    Thesis: liquidity provision. Short-horizon index overreaction to forced /
    uninformed selling reverts as dealers and value buyers step in; the regime
    filter avoids catching falling knives in bear markets.
    """
    px = close[ticker].dropna()
    r = rsi(px, rsi_n)
    sma_exit = px.rolling(exit_sma).mean()
    sma_reg = px.rolling(regime_sma).mean()
    pos = np.zeros(len(px))
    held = 0
    in_pos = False
    for i in range(regime_sma, len(px)):
        if in_pos:
            held += 1
            if px.iloc[i] > sma_exit.iloc[i] or held >= max_hold:
                in_pos = False
        else:
            if r.iloc[i] < entry and px.iloc[i] > sma_reg.iloc[i]:
                in_pos, held = True, 0
        pos[i] = 1.0 if in_pos else 0.0
    return pd.DataFrame({ticker: pos}, index=px.index).reindex(close.index).fillna(0.0)


# --------------------------------------------------------------------------- #
# S4. Global dual momentum (relative + absolute)
# --------------------------------------------------------------------------- #
def dual_momentum(close: pd.DataFrame, cash: pd.Series, lookback: int = 252,
                  risky=("SPY", "EFA"), safe: str = "IEF") -> pd.DataFrame:
    """If the best risky asset's lookback return beats T-bills, hold it;
    otherwise hold intermediate Treasuries (Antonacci 2014). Monthly."""
    cols = list(risky) + [safe]
    px = close[cols]
    cg = _cash_growth(cash.reindex(px.index).fillna(0))
    mom = px.pct_change(lookback, fill_method=None)
    cash_mom = cg.pct_change(lookback)
    rk = mom[list(risky)]
    best_ret = rk.max(axis=1)
    best = rk.fillna(-np.inf).idxmax(axis=1).where(best_ret.notna())
    w = pd.DataFrame(0.0, index=px.index, columns=cols)
    risk_on = best_ret > cash_mom
    for a in risky:
        w.loc[risk_on & (best == a), a] = 1.0
    safe_ok = px[safe].notna() & ~risk_on & best_ret.notna()
    w.loc[safe_ok, safe] = 1.0
    me = month_ends(px.index)
    return w.loc[me].reindex(px.index).ffill().fillna(0.0)


# --------------------------------------------------------------------------- #
# S5. Turn-of-the-month seasonality
# --------------------------------------------------------------------------- #
def turn_of_month(close: pd.DataFrame, cash: pd.Series, ticker: str = "SPY",
                  days_before: int = 1, days_after: int = 3) -> pd.DataFrame:
    """Long the index over the last ``days_before`` and first ``days_after``
    trading days of each month; T-bills otherwise.

    Thesis: month-end pension/401(k) contributions, fund window dressing and
    payroll-driven inflows concentrate buying around the turn of the month
    (Ariel 1987; Lakonishok & Smidt 1988; Etula et al. 2020).

    NOTE: the schedule is known in advance, so weights here are *return-day*
    indicators and must be run with ``lag=0`` after shifting by one day (done
    below): target[t] = 1 means "hold during day t+1".
    """
    idx = close[ticker].dropna().index
    s = pd.Series(idx, index=idx)
    ym = idx.year * 12 + idx.month
    pos_in_month = s.groupby(ym).cumcount() + 1
    from_end = s.groupby(ym).cumcount(ascending=False) + 1
    hold_day = (pos_in_month <= days_after) | (from_end <= days_before)
    target = hold_day.shift(-1, fill_value=False).astype(float)
    return pd.DataFrame({ticker: target.values}, index=idx).reindex(close.index).fillna(0.0)


# --------------------------------------------------------------------------- #
# S6. Volatility-managed equity exposure
# --------------------------------------------------------------------------- #
def vol_managed(close: pd.DataFrame, cash: pd.Series, ticker: str = "SPY",
                target_vol: float = 0.15, window: int = 20, cap: float = 1.0,
                rebalance_band: float = 0.10) -> pd.DataFrame:
    """Scale equity exposure by target_vol / realised vol (Moreira & Muir 2017).

    Thesis: volatility clusters and is only weakly related to subsequent
    returns, so cutting exposure when vol is high improves the risk/return
    trade-off and truncates crash drawdowns. A no-trade band limits churn.
    """
    px = close[ticker].dropna()
    rv = px.pct_change().rolling(window).std() * np.sqrt(252)
    raw = (target_vol / rv).clip(upper=cap)
    w = np.full(len(raw), np.nan)
    cur = np.nan
    for i, x in enumerate(raw.values):
        if np.isnan(x):
            continue
        if np.isnan(cur) or abs(x - cur) > rebalance_band:
            cur = x
        w[i] = cur
    return pd.DataFrame({ticker: w}, index=px.index).reindex(close.index).fillna(0.0)


# --------------------------------------------------------------------------- #
# S7. Overnight drift (negative control - evaluated separately, see analysis)
# --------------------------------------------------------------------------- #
def overnight_returns(open_: pd.Series, close: pd.Series, cost_bps: float) -> pd.Series:
    """Buy at the close, sell at the next open, every day (Cliff, Cooper & Gulen
    2008; Kelly & Clark 2011). Round-trip cost is charged daily."""
    on = open_ / close.shift(1) - 1
    return (on - 2 * cost_bps / 1e4).dropna()


# --------------------------------------------------------------------------- #
# Benchmarks
# --------------------------------------------------------------------------- #
def buy_hold(close: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Static weights, rebalanced back to target daily in the engine (the drift
    between rebalances is charged as turnover, which is conservative)."""
    avail = close[list(weights)].notna()
    w = pd.DataFrame({k: v for k, v in weights.items()}, index=close.index) * avail
    return w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * sum(weights.values())


def equal_weight(close: pd.DataFrame, universe) -> pd.DataFrame:
    px = close[list(universe)]
    avail = px.notna().astype(float)
    w = avail.div(avail.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    me = month_ends(px.index)
    return w.loc[me].reindex(px.index).ffill().fillna(0.0)


# --------------------------------------------------------------------------- #
# Iteration: portfolio-level volatility targeting overlay
# --------------------------------------------------------------------------- #
def vol_target_overlay(target: pd.DataFrame, asset_returns: pd.DataFrame,
                       target_vol: float = 0.10, window: int = 63,
                       cap: float = 2.0) -> pd.DataFrame:
    """Scale a strategy's target weights so its trailing realised volatility
    (computed from the *unscaled* book held at each date) matches
    ``target_vol``. The scale is refreshed at month-ends only and uses data up
    to that close, so it inherits the backtester's execution lag.
    Gross exposure > 1 is financed at T-bill + spread by the engine."""
    w = target.reindex(asset_returns.index).fillna(0.0)
    book = (w.shift(1) * asset_returns[w.columns].fillna(0.0)).sum(axis=1)
    rv = book.rolling(window).std() * np.sqrt(252)
    scale = (target_vol / rv).clip(upper=cap).replace([np.inf], cap)
    me = month_ends(w.index)
    scale = scale.loc[me].reindex(w.index).ffill()
    return w.mul(scale, axis=0).fillna(0.0)
