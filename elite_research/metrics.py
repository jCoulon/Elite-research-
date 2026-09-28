"""Performance and robustness statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TDAYS = 252


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def drawdown_series(r: pd.Series) -> pd.Series:
    eq = (1 + r).cumprod()
    return eq / eq.cummax() - 1


def sharpe(r: pd.Series, cash: pd.Series | None = None) -> float:
    ex = r - (cash.reindex(r.index).fillna(0) if cash is not None else 0)
    sd = ex.std()
    return float(ex.mean() / sd * np.sqrt(TDAYS)) if sd > 0 else np.nan


def sortino(r: pd.Series, cash: pd.Series | None = None) -> float:
    ex = r - (cash.reindex(r.index).fillna(0) if cash is not None else 0)
    dd = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return float(ex.mean() / dd * np.sqrt(TDAYS)) if dd > 0 else np.nan


def summary(r: pd.Series, cash: pd.Series, bench: pd.Series | None = None,
            turnover: pd.Series | None = None, exposure: pd.Series | None = None,
            trades: pd.DataFrame | None = None) -> dict:
    r = r.dropna()
    n = len(r)
    years = n / TDAYS
    total = float((1 + r).prod() - 1)
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    vol = float(r.std() * np.sqrt(TDAYS))
    mdd = max_drawdown(r)
    monthly = (1 + r).resample("ME").prod() - 1
    pos, neg = monthly[monthly > 0].sum(), -monthly[monthly < 0].sum()
    out = {
        "start": r.index[0].date(), "end": r.index[-1].date(),
        "total_return": total, "cagr": cagr, "vol": vol,
        "sharpe": sharpe(r, cash), "sortino": sortino(r, cash),
        "max_dd": mdd, "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "monthly_win_rate": float((monthly > 0).mean()),
        "profit_factor": float(pos / neg) if neg > 0 else np.nan,
        "worst_month": float(monthly.min()),
        "skew_monthly": float(stats.skew(monthly)),
    }
    if bench is not None:
        b = bench.reindex(r.index).fillna(0)
        c = cash.reindex(r.index).fillna(0)
        x, y = b - c, r - c
        beta = float(np.cov(y, x)[0, 1] / x.var()) if x.var() > 0 else np.nan
        alpha = float((y.mean() - beta * x.mean()) * TDAYS)
        resid = y - beta * x
        out.update({"beta": beta, "alpha_ann": alpha,
                    "alpha_t": float(resid.mean() / resid.std() * np.sqrt(n)),
                    "corr_bench": float(np.corrcoef(r, b)[0, 1])})
    if turnover is not None:
        out["turnover_ann"] = float(turnover.reindex(r.index).sum() / years)
    if exposure is not None:
        out["avg_exposure"] = float(exposure.reindex(r.index).mean())
    if trades is not None and len(trades):
        t = trades
        wins, losses = t.loc[t["return"] > 0, "return"], t.loc[t["return"] <= 0, "return"]
        out.update({"n_trades": len(t), "trade_win_rate": float((t["return"] > 0).mean()),
                    "avg_trade": float(t["return"].mean()),
                    "trade_profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else np.nan,
                    "avg_hold_days": float(t["days"].mean())})
    return out


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """Bailey & Lopez de Prado PSR on daily data; sr_benchmark is annualised."""
    r = r.dropna()
    n = len(r)
    sr = r.mean() / r.std()
    sk, ku = stats.skew(r), stats.kurtosis(r, fisher=False)
    srb = sr_benchmark / np.sqrt(TDAYS)
    z = (sr - srb) * np.sqrt(n - 1) / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf(z))


def deflated_sharpe(r: pd.Series, trial_sharpes_ann: list[float]) -> float:
    """Deflated Sharpe ratio: PSR against the expected max Sharpe of N trials."""
    n_trials = max(len(trial_sharpes_ann), 2)
    var_sr = np.var(np.asarray(trial_sharpes_ann) / np.sqrt(TDAYS), ddof=1)
    emc = 0.5772156649
    sr0 = np.sqrt(var_sr) * ((1 - emc) * stats.norm.ppf(1 - 1 / n_trials)
                             + emc * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    return probabilistic_sharpe(r, sr0 * np.sqrt(TDAYS))


def bootstrap_sharpe_ci(r: pd.Series, cash: pd.Series, n_boot: int = 2000,
                        block: int = 21, seed: int = 7) -> tuple[float, float]:
    """Stationary-ish block bootstrap 90% CI for annualised excess Sharpe."""
    ex = (r - cash.reindex(r.index).fillna(0)).dropna().values
    n = len(ex)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block, n_blocks)
        s = np.concatenate([ex[k:k + block] for k in starts])[:n]
        out[i] = s.mean() / s.std() * np.sqrt(TDAYS)
    return float(np.percentile(out, 5)), float(np.percentile(out, 95))
