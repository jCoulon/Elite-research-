"""End-to-end research run: market overview, strategy backtests, robustness,
walk-forward, stress tests, ranking. Writes tables to results/ and charts to
results/charts/. Usage:  python run_research.py [--refresh]
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from elite_research import data, engine, metrics as M, strategies as S
from elite_research.plotting import PALETTE, NEUTRAL, INK2, plt, save, setup

# ----------------------------------------------------------------------------- config
END = "2026-09-25"          # last complete session when the study was run
START = "2001-01-01"        # first evaluation date (1y warm-up after sector ETF launch)
IS_END = "2014-12-31"       # in-sample: 2001-2014 (dot-com bust, GFC, recovery)
OOS_START = "2015-01-01"    # out-of-sample: 2015-2026 (2018, COVID, 2022 rate shock)
COST_BPS = 5.0              # per unit one-way turnover (commission + half-spread + slippage)
RES = Path("results")
CH = RES / "charts"

STRATEGIES = {
    # name: (function, default params, sensitivity grid, execution lag)
    "Cross-Asset Trend": (
        S.cross_asset_trend, {},
        {"lookbacks": [(252,), (126,), (63, 126, 252), (21, 63, 126, 252)],
         "vol_window": [21, 63, 126]}, 1),
    "Sector Momentum": (
        S.sector_momentum, {},
        {"lookback": [63, 126, 189, 252], "skip": [0, 21], "top_n": [2, 3, 4, 5],
         "trend_filter": [False, True]}, 1),
    "RSI-2 Mean Reversion": (
        S.rsi_mean_reversion, {},
        {"entry": [5, 10, 15, 20, 25], "rsi_n": [2, 3, 4], "exit_sma": [3, 5, 10]}, 1),
    "Dual Momentum": (
        S.dual_momentum, {},
        {"lookback": [63, 126, 189, 252, 315], "safe": ["IEF", "TLT"]}, 1),
    "Turn of Month": (
        S.turn_of_month, {},
        {"days_before": [1, 2, 3], "days_after": [2, 3, 4, 5]}, 0),
    "Vol-Managed Equity": (
        S.vol_managed, {},
        {"target_vol": [0.10, 0.12, 0.15, 0.20], "window": [10, 20, 42, 63],
         "cap": [1.0, 1.5]}, 1),
}
SINGLE_ASSET = {"RSI-2 Mean Reversion", "Turn of Month"}  # discrete in/out trades

CRISES = {
    "Dot-com bust": ("2001-01-01", "2002-10-09"),
    "GFC": ("2007-10-09", "2009-03-09"),
    "2011 euro crisis": ("2011-04-29", "2011-10-03"),
    "2015-16 selloff": ("2015-07-20", "2016-02-11"),
    "Q4 2018": ("2018-09-20", "2018-12-24"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 rate shock": ("2022-01-03", "2022-10-12"),
    "2025 tariff shock": ("2025-02-19", "2025-04-08"),
}


def fmt_pct(x):
    return "" if pd.isna(x) else f"{x*100:.1f}%"


def fmt_num(x, d=2):
    return "" if pd.isna(x) else f"{x:.{d}f}"


# ----------------------------------------------------------------------------- setup
def load():
    refresh = "--refresh" in sys.argv
    close = data.load_panel(refresh=refresh).loc[:END]
    # Anchor the calendar to NYSE sessions: index series (e.g. ^VIX) occasionally
    # carry prints on exchange holidays, which would insert empty ETF rows and
    # poison rolling windows.
    close = close[close["SPY"].notna()]
    etfs = close.drop(columns=["^VIX", "^IRX"])
    gaps = (etfs.isna() & etfs.ffill().notna() & etfs.bfill().notna()).sum()
    assert gaps.sum() == 0, f"interior price gaps: {gaps[gaps > 0].to_dict()}"
    open_ = data.load_panel(["SPY"], field="Open").reindex(close.index)
    rets = close.pct_change(fill_method=None)
    asset_rets = rets.drop(columns=["^VIX", "^IRX"])
    cash = data.cash_returns(close.index)
    return close, open_, asset_rets, cash


def backtest(name, fn, params, lag, close, asset_rets, cash, cost=COST_BPS, start=START):
    target = fn(close, cash, **params)
    bt = engine.run_backtest(name, target, asset_rets, cash, cost_bps=cost, lag=lag,
                             start=start, end=END)
    if name in SINGLE_ASSET:
        col = target.columns[0]
        bt.trades = engine.round_trips(bt.weights[col], asset_rets[col].loc[bt.returns.index], cost)
    return bt


def stats_for(bt_or_r, cash, spy, sl=slice(None)):
    if isinstance(bt_or_r, engine.BacktestResult):
        r = bt_or_r.returns.loc[sl]
        tr = bt_or_r.trades
        if tr is not None and len(tr):
            tr = tr[(tr["entry"] >= r.index[0]) & (tr["entry"] <= r.index[-1])]
        return M.summary(r, cash, spy, bt_or_r.turnover, bt_or_r.weights.sum(axis=1), tr)
    return M.summary(bt_or_r.loc[sl], cash, spy)


# ----------------------------------------------------------------------------- 1. market overview
def market_overview(close, cash):
    out = {}
    px = close.drop(columns=["^VIX", "^IRX"])
    last = px.index[-1]
    rows = []
    for t in px.columns:
        s = px[t].dropna()
        if len(s) < 260:
            continue
        r = lambda n: s.iloc[-1] / s.iloc[-1 - n] - 1  # noqa: E731
        rows.append({"ticker": t, "1m": r(21), "3m": r(63), "6m": r(126), "12m": r(252),
                     "vs_200d": s.iloc[-1] / s.iloc[-200:].mean() - 1,
                     "dd_from_high": s.iloc[-1] / s.max() - 1,
                     "vol_1m": s.pct_change().iloc[-21:].std() * np.sqrt(252),
                     "vol_1y": s.pct_change().iloc[-252:].std() * np.sqrt(252)})
    ov = pd.DataFrame(rows).set_index("ticker")
    spy12 = ov.loc["SPY", "12m"]
    ov["rs_vs_spy_6m"] = ov["6m"] - ov.loc["SPY", "6m"]
    ov.to_csv(RES / "market_overview.csv")

    fred = {s: data.load_fred(s) for s in data.FRED_SERIES}
    vix = close["^VIX"].dropna()
    macro = {
        "as_of": str(last.date()),
        "SPY_close": float(px["SPY"].iloc[-1]),
        "SPY_12m": float(spy12),
        "SPY_vs_200d": float(ov.loc["SPY", "vs_200d"]),
        "SPY_dd_from_high": float(ov.loc["SPY", "dd_from_high"]),
        "VIX": float(vix.iloc[-1]),
        "VIX_1y_pctile": float((vix.iloc[-252:] < vix.iloc[-1]).mean()),
        "VIX_20y_median": float(vix.loc["2006":].median()),
        "UST10Y": float(fred["DGS10"].iloc[-1]),
        "UST2Y": float(fred["DGS2"].iloc[-1]),
        "UST3M": float(fred["DGS3MO"].iloc[-1]),
        "curve_10y_3m": float(fred["DGS10"].iloc[-1] - fred["DGS3MO"].iloc[-1]),
        "curve_10y_2y": float(fred["DGS10"].iloc[-1] - fred["DGS2"].iloc[-1]),
        "UST10Y_chg_12m": float(fred["DGS10"].iloc[-1] - fred["DGS10"].loc[:last - pd.DateOffset(years=1)].iloc[-1]),
        "breakeven_10y": float(fred["T10YIE"].iloc[-1]),
        "HY_OAS": float(fred["BAMLH0A0HYM2"].iloc[-1]),
        "HY_OAS_10y_pctile": float((fred["BAMLH0A0HYM2"].loc["2016":] < fred["BAMLH0A0HYM2"].iloc[-1]).mean()),
        "unemployment": float(fred["UNRATE"].iloc[-1]),
        "cpi_yoy": float(fred["CPIAUCSL"].iloc[-1] / fred["CPIAUCSL"].iloc[-13] - 1),
        "breadth_sectors_above_200d": float((ov.loc[[s for s in data.SECTORS if s in ov.index], "vs_200d"] > 0).mean()),
        "RSP_minus_SPY_12m": float(ov.loc["RSP", "12m"] - spy12),
        "IWM_minus_SPY_12m": float(ov.loc["IWM", "12m"] - spy12),
    }
    out["macro"] = macro
    out["table"] = ov
    (RES / "market_macro.json").write_text(json.dumps(macro, indent=2))

    # chart: sector 6m relative strength + SPY vs 200d
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    spx = px["SPY"].loc["2024-01-01":]
    ax.plot(spx.index, spx, color=PALETTE[0], label="SPY (total return)")
    ax.plot(spx.index, px["SPY"].rolling(200).mean().loc[spx.index], color=NEUTRAL,
            lw=1.2, ls="--", label="200-day average")
    ax.set_title("SPY vs its 200-day average")
    ax.legend(loc="upper left")
    ax = axes[1]
    secs = [s for s in data.SECTORS + ["SPY", "EFA", "EEM", "TLT", "GLD", "DBC"] if s in ov.index]
    rs = ov.loc[secs, "6m"].sort_values()
    colors = [PALETTE[0] if v >= 0 else PALETTE[1] for v in rs.values]
    ax.barh(rs.index, rs.values * 100, color=colors, height=0.6)
    for i, v in enumerate(rs.values):
        ax.text(v * 100 + (0.5 if v >= 0 else -0.5), i, f"{v*100:.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=7, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(rs.min() * 100 - 6, rs.max() * 100 + 8)
    ax.set_title("6-month total return by ETF")
    ax.set_xlabel("%")
    save(fig, CH / "01_market_overview.png")
    return out


# ----------------------------------------------------------------------------- 2. backtests
def main():
    RES.mkdir(exist_ok=True)
    CH.mkdir(exist_ok=True)
    setup()
    close, open_, asset_rets, cash = load()
    spy = asset_rets["SPY"]
    ov = market_overview(close, cash)

    # Benchmarks
    benches = {
        "SPY Buy & Hold": engine.run_backtest("SPY Buy & Hold", S.buy_hold(close, {"SPY": 1.0}),
                                              asset_rets, cash, COST_BPS, 0, start=START, end=END),
        "60/40 SPY/IEF": engine.run_backtest("60/40 SPY/IEF", S.buy_hold(close, {"SPY": 0.6, "IEF": 0.4}),
                                             asset_rets, cash, COST_BPS, 0, start=START, end=END),
        "EW Sectors": engine.run_backtest("EW Sectors", S.equal_weight(close, data.SECTORS),
                                          asset_rets, cash, COST_BPS, 0, start=START, end=END),
    }

    results, grids, grid_returns = {}, {}, {}
    all_trial_sharpes_is = []
    for name, (fn, defaults, grid, lag) in STRATEGIES.items():
        print("running", name)
        results[name] = backtest(name, fn, defaults, lag, close, asset_rets, cash)
        keys = list(grid)
        rows, rets_cfg = [], {}
        for combo in itertools.product(*grid.values()):
            params = dict(zip(keys, combo))
            bt = backtest(name, fn, params, lag, close, asset_rets, cash)
            label = ", ".join(f"{k}={v}" for k, v in params.items())
            r = bt.returns
            rets_cfg[label] = r
            rows.append({**{k: str(v) for k, v in params.items()}, "label": label,
                         "sharpe_is": M.sharpe(r.loc[:IS_END], cash),
                         "sharpe_oos": M.sharpe(r.loc[OOS_START:], cash),
                         "cagr_is": stats_for(r, cash, spy, slice(None, IS_END))["cagr"],
                         "cagr_oos": stats_for(r, cash, spy, slice(OOS_START, None))["cagr"],
                         "mdd_full": M.max_drawdown(r)})
        g = pd.DataFrame(rows)
        grids[name] = g
        grid_returns[name] = pd.DataFrame(rets_cfg)
        all_trial_sharpes_is += g["sharpe_is"].tolist()
        g.to_csv(RES / f"grid_{name.replace(' ', '_').replace('/', '')}.csv", index=False)

    # Overnight drift negative control (close -> next open)
    on = S.overnight_returns(open_["SPY"], close["SPY"], COST_BPS).loc[START:END]
    on_nocost = S.overnight_returns(open_["SPY"], close["SPY"], 0.0).loc[START:END]
    on_1bp = S.overnight_returns(open_["SPY"], close["SPY"], 1.0).loc[START:END]
    overnight = {
        "gross (0 bps)": M.summary(on_nocost, cash * 0, spy),
        "1 bp/side": M.summary(on_1bp, cash * 0, spy),
        f"{COST_BPS:.0f} bps/side": M.summary(on, cash * 0, spy),
    }
    pd.DataFrame(overnight).T.to_csv(RES / "overnight_negative_control.csv")

    # ------------------------------------------------------------------ summary tables
    periods = {"Full": slice(START, END), "In-sample": slice(START, IS_END), "Out-of-sample": slice(OOS_START, END)}
    summ = []
    for pname, sl in periods.items():
        for name, bt in {**results, **benches}.items():
            s = stats_for(bt, cash, spy, sl)
            summ.append({"period": pname, "strategy": name, **s})
    summ = pd.DataFrame(summ)
    summ.to_csv(RES / "summary_metrics.csv", index=False)

    # Deflated Sharpe & bootstrap CI on the default configs (full period)
    extra = {}
    for name, bt in results.items():
        r = bt.returns
        ex = r - cash.reindex(r.index)
        lo, hi = M.bootstrap_sharpe_ci(r, cash)
        extra[name] = {"psr_vs_0": M.probabilistic_sharpe(ex),
                       "dsr": M.deflated_sharpe(ex, all_trial_sharpes_is),
                       "sharpe_ci90_lo": lo, "sharpe_ci90_hi": hi}
    for name, bt in benches.items():
        lo, hi = M.bootstrap_sharpe_ci(bt.returns, cash)
        extra[name] = {"sharpe_ci90_lo": lo, "sharpe_ci90_hi": hi,
                       "psr_vs_0": M.probabilistic_sharpe(bt.returns - cash.reindex(bt.returns.index))}
    extra = pd.DataFrame(extra).T
    extra.to_csv(RES / "significance.csv")

    # ------------------------------------------------------------------ robustness: grid IS vs OOS
    robust = {}
    for name, g in grids.items():
        best_is = g.loc[g["sharpe_is"].idxmax()]
        robust[name] = {
            "n_configs": len(g),
            "median_sharpe_is": g["sharpe_is"].median(),
            "median_sharpe_oos": g["sharpe_oos"].median(),
            "pct_configs_oos_sharpe_gt0": (g["sharpe_oos"] > 0).mean(),
            "pct_configs_oos_sharpe_gt_0.5": (g["sharpe_oos"] > 0.5).mean(),
            "best_is_config": best_is["label"],
            "best_is_sharpe_is": best_is["sharpe_is"],
            "best_is_sharpe_oos": best_is["sharpe_oos"],
            "rank_corr_is_oos": g["sharpe_is"].rank().corr(g["sharpe_oos"].rank()),
        }

    # Walk-forward: each Jan, pick the config with the best trailing-5y Sharpe
    wf_returns = {}
    for name, R in grid_returns.items():
        pieces = []
        for yr in range(2006, int(END[:4]) + 1):
            trail = R.loc[f"{yr-5}-01-01":f"{yr-1}-12-31"]
            ex = trail.sub(cash.reindex(trail.index), axis=0)
            sh = ex.mean() / ex.std()
            pick = sh.idxmax()
            pieces.append(R.loc[f"{yr}-01-01":f"{yr}-12-31", pick])
        wf = pd.concat(pieces)
        wf_returns[name] = wf
        robust[name]["walkforward_sharpe_2006_on"] = M.sharpe(wf, cash)
        robust[name]["walkforward_sharpe_oos"] = M.sharpe(wf.loc[OOS_START:], cash)
        robust[name]["default_sharpe_2006_on"] = M.sharpe(results[name].returns.loc["2006":], cash)
    robust = pd.DataFrame(robust).T
    robust.to_csv(RES / "robustness.csv")

    # ------------------------------------------------------------------ stress: costs and execution lag
    stress = []
    for name, (fn, defaults, grid, lag) in STRATEGIES.items():
        for cost in [0, 5, 10, 20, 30]:
            bt = backtest(name, fn, defaults, lag, close, asset_rets, cash, cost=cost)
            stress.append({"strategy": name, "test": f"cost {cost} bps", "sharpe_full": M.sharpe(bt.returns, cash),
                           "sharpe_oos": M.sharpe(bt.returns.loc[OOS_START:], cash),
                           "cagr_full": stats_for(bt.returns, cash, spy)["cagr"]})
        for lg in [0, 1, 2]:
            bt = backtest(name, fn, defaults, lg, close, asset_rets, cash)
            stress.append({"strategy": name, "test": f"lag {lg}d", "sharpe_full": M.sharpe(bt.returns, cash),
                           "sharpe_oos": M.sharpe(bt.returns.loc[OOS_START:], cash),
                           "cagr_full": stats_for(bt.returns, cash, spy)["cagr"]})
    stress = pd.DataFrame(stress)
    stress.to_csv(RES / "stress_tests.csv", index=False)

    # ------------------------------------------------------------------ regimes
    all_r = pd.DataFrame({n: bt.returns for n, bt in {**results, **benches}.items()})
    yearly = (1 + all_r).groupby(all_r.index.year).prod() - 1
    yearly.to_csv(RES / "calendar_year_returns.csv")
    crisis = pd.DataFrame({c: (1 + all_r.loc[a:b]).prod() - 1 for c, (a, b) in CRISES.items()}).T
    crisis.to_csv(RES / "crisis_returns.csv")
    vix_prev = close["^VIX"].shift(1).reindex(all_r.index)
    regime = pd.cut(vix_prev, [0, 15, 25, 200], labels=["VIX<15", "VIX 15-25", "VIX>25"])
    vix_tbl = all_r.groupby(regime, observed=True).mean() * 252
    vix_tbl = all_r.groupby(regime, observed=True).mean() * 252
    vix_tbl.to_csv(RES / "vix_regime_ann_returns.csv")
    spy_trend = (close["SPY"] > close["SPY"].rolling(200).mean()).shift(1).reindex(all_r.index)
    trend_tbl = all_r.groupby(spy_trend.map({True: "SPY>200d", False: "SPY<200d"})).mean() * 252
    trend_tbl.to_csv(RES / "trend_regime_ann_returns.csv")

    # ------------------------------------------------------------------ correlation & ensemble
    strat_r = pd.DataFrame({n: bt.returns for n, bt in results.items()})
    monthly = (1 + strat_r).resample("ME").prod() - 1
    corr = monthly.corr()
    corr.to_csv(RES / "strategy_correlation_monthly.csv")

    # Pre-registered ensemble rule (uses in-sample information only):
    # include a strategy if IS excess Sharpe > 0.3 and IS max drawdown > -35%;
    # weight by inverse IS volatility; rebalance monthly.
    is_s = summ[(summ.period == "In-sample") & summ.strategy.isin(results)].set_index("strategy")
    members = is_s[(is_s.sharpe > 0.3) & (is_s.max_dd > -0.35)].index.tolist()
    inv = 1 / is_s.loc[members, "vol"]
    ens_w = (inv / inv.sum()).to_dict()
    # daily approximation: constant weights on daily sleeve returns (monthly rebalanced ~ daily)
    ens = (strat_r[members] * pd.Series(ens_w)).sum(axis=1).rename("Ensemble")
    ens_turn = sum(results[m].turnover * w for m, w in ens_w.items())
    ens_stats = {p: M.summary(ens.loc[sl], cash, spy) for p, sl in periods.items()}
    ens_extra = {"members": ens_w, "psr_vs_0": M.probabilistic_sharpe(ens - cash.reindex(ens.index)),
                 "sharpe_ci90": M.bootstrap_sharpe_ci(ens, cash),
                 "turnover_ann": float(ens_turn.sum() / (len(ens) / 252))}
    pd.DataFrame(ens_stats).T.to_csv(RES / "ensemble_metrics.csv")
    (RES / "ensemble_members.json").write_text(json.dumps(ens_extra, indent=2, default=str))
    all_r["Ensemble"] = ens
    yearly["Ensemble"] = (1 + ens).groupby(ens.index.year).prod() - 1
    yearly.to_csv(RES / "calendar_year_returns.csv")
    crisis["Ensemble"] = [(1 + ens.loc[a:b]).prod() - 1 for a, b in CRISES.values()]
    crisis.to_csv(RES / "crisis_returns.csv")

    # ------------------------------------------------------------------ iteration round
    # Motivated by round-1 evidence (not by OOS returns): trend runs at ~5% vol
    # with 58% average exposure; sector momentum's weakness is crash drawdowns
    # (-45%), which vol-scaling is documented to reduce (Barroso & Santa-Clara 2015).
    iters = {}
    t_trend = S.cross_asset_trend(close, cash)
    iters["Trend @10% vol (≤2x)"] = engine.run_backtest(
        "Trend @10% vol", S.vol_target_overlay(t_trend, asset_rets, 0.10, 63, 2.0),
        asset_rets, cash, COST_BPS, 1, start=START, end=END)
    t_sec = S.sector_momentum(close, cash)
    iters["Sector Mom. vol-managed (15%, ≤1x)"] = engine.run_backtest(
        "Sector Mom VM", S.vol_target_overlay(t_sec, asset_rets, 0.15, 63, 1.0),
        asset_rets, cash, COST_BPS, 1, start=START, end=END)
    iter_rows = []
    for n, bt in iters.items():
        for pname, sl in periods.items():
            iter_rows.append({"period": pname, "strategy": n, **stats_for(bt, cash, spy, sl)})
        all_trial_sharpes_is.append(M.sharpe(bt.returns.loc[:IS_END], cash))
    # Post-hoc blend (flagged: chosen after seeing round-1 results incl. OOS)
    blend = (0.5 * iters["Trend @10% vol (≤2x)"].returns
             + 0.5 * iters["Sector Mom. vol-managed (15%, ≤1x)"].returns).rename("Blend")
    for pname, sl in periods.items():
        iter_rows.append({"period": pname, "strategy": "Blend: 50% Trend@10% + 50% SecMom-VM (post-hoc)",
                          **M.summary(blend.loc[sl], cash, spy)})
    iter_df = pd.DataFrame(iter_rows)
    iter_df.to_csv(RES / "iteration_metrics.csv", index=False)
    iter_extra = {}
    for n, r in {**{k: v.returns for k, v in iters.items()}, "Blend": blend}.items():
        ex = r - cash.reindex(r.index)
        iter_extra[n] = {"dsr": M.deflated_sharpe(ex, all_trial_sharpes_is),
                         "sharpe_ci90": M.bootstrap_sharpe_ci(r, cash)}
        crisis[n] = [(1 + r.loc[a:b]).prod() - 1 for a, b in CRISES.values()]
        yearly[n] = (1 + r).groupby(r.index.year).prod() - 1
        all_r[n] = r
    (RES / "iteration_significance.json").write_text(json.dumps(iter_extra, indent=2, default=float))
    crisis.to_csv(RES / "crisis_returns.csv")
    yearly.to_csv(RES / "calendar_year_returns.csv")
    (RES / "n_trials.txt").write_text(str(len(all_trial_sharpes_is)))

    # ------------------------------------------------------------------ ranking
    oos = summ[summ.period == "Out-of-sample"].set_index("strategy")
    full = summ[summ.period == "Full"].set_index("strategy")
    st20 = stress[stress.test == "cost 20 bps"].set_index("strategy")
    rank_df = pd.DataFrame({
        "sharpe_full": full.loc[list(results), "sharpe"],
        "sharpe_oos": oos.loc[list(results), "sharpe"],
        "wf_sharpe_oos": robust["walkforward_sharpe_oos"].astype(float),
        "pct_grid_oos_pos": robust["pct_configs_oos_sharpe_gt0"].astype(float),
        "sharpe_oos_20bps": st20["sharpe_oos"],
        "max_dd_full": full.loc[list(results), "max_dd"],
        "alpha_t_full": full.loc[list(results), "alpha_t"],
        "dsr": extra.loc[list(results), "dsr"].astype(float),
    })
    score = rank_df.rank(ascending=True).mean(axis=1)  # higher is better for every column
    rank_df["composite_rank_score"] = score
    rank_df = rank_df.sort_values("composite_rank_score", ascending=False)
    rank_df.to_csv(RES / "ranking.csv")

    # ------------------------------------------------------------------ recommended portfolio A
    # 70% Vol-Managed Equity + 30% Cross-Asset Trend (post-hoc recommendation; see REPORT §6)
    port_a = (0.7 * results["Vol-Managed Equity"].returns + 0.3 * results["Cross-Asset Trend"].returns)
    pa = {p: M.summary(port_a.loc[sl], cash, spy) for p, sl in periods.items()}
    pa_df = pd.DataFrame(pa).T
    pa_df.to_csv(RES / "portfolio_a_metrics.csv")
    pd.Series({c: (1 + port_a.loc[a:b]).prod() - 1 for c, (a, b) in CRISES.items()}).to_csv(RES / "portfolio_a_crises.csv")

    # ------------------------------------------------------------------ current positioning
    live = {}
    for name, (fn, defaults, grid, lag) in STRATEGIES.items():
        w = fn(close, cash, **defaults).iloc[-1]
        live[name] = {k: round(float(v), 4) for k, v in w[w.abs() > 1e-6].items()}
    (RES / "current_positions.json").write_text(json.dumps(live, indent=2))

    # ------------------------------------------------------------------ charts
    charts(results, benches, ens, grids, stress, corr, all_r, cash, ov)
    print("done")


def charts(results, benches, ens, grids, stress, corr, all_r, cash, ov):
    # equity curves (log) + drawdowns
    names = list(results) + ["Ensemble"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    for i, n in enumerate(names):
        eq = (1 + all_r[n]).cumprod()
        ax.plot(eq.index, eq, color=PALETTE[i % 8], lw=1.3, label=n)
    eq = (1 + all_r["SPY Buy & Hold"]).cumprod()
    ax.plot(eq.index, eq, color="#0b0b0b", lw=1.0, ls="--", label="SPY Buy & Hold")
    ax.set_yscale("log")
    ax.axvline(pd.Timestamp(OOS_START), color=NEUTRAL, lw=0.8)
    ax.text(pd.Timestamp(OOS_START), ax.get_ylim()[1] * 0.9, "  out-of-sample →", color=INK2, fontsize=8)
    ax.set_title(f"Growth of $1, net of {COST_BPS:.0f} bps costs (log scale)")
    ax.legend(ncol=4, loc="upper left")
    ax = axes[1]
    for i, n in enumerate(names):
        ax.plot(all_r.index, M.drawdown_series(all_r[n]) * 100, color=PALETTE[i % 8], lw=0.9)
    ax.plot(all_r.index, M.drawdown_series(all_r["SPY Buy & Hold"]) * 100, color="#0b0b0b", lw=0.8, ls="--")
    ax.set_title("Drawdown (%)")
    save(fig, CH / "02_equity_drawdowns.png")

    # rolling 3y sharpe
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, n in enumerate(names):
        ex = all_r[n] - cash.reindex(all_r.index)
        rs = ex.rolling(756).mean() / ex.rolling(756).std() * np.sqrt(252)
        ax.plot(rs.index, rs, color=PALETTE[i % 8], lw=1.0, label=n)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title("Rolling 3-year Sharpe ratio (excess of T-bills)")
    ax.legend(ncol=4, loc="upper left")
    save(fig, CH / "03_rolling_sharpe.png")

    # IS vs OOS sharpe scatter per grid (small multiples, one colour)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (n, g) in zip(axes.flat, grids.items()):
        ax.scatter(g["sharpe_is"], g["sharpe_oos"], s=18, color=PALETTE[0], alpha=0.75, edgecolor="white", lw=0.5)
        lim = [min(g[["sharpe_is", "sharpe_oos"]].min().min(), 0) - 0.1, g[["sharpe_is", "sharpe_oos"]].max().max() + 0.1]
        ax.plot(lim, lim, color=NEUTRAL, lw=0.8, ls="--")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(n, fontsize=10)
        ax.set_xlabel("In-sample Sharpe (2001-14)")
        ax.set_ylabel("Out-of-sample Sharpe (2015-26)")
    fig.suptitle("Parameter robustness: every grid configuration, IS vs OOS", x=0.01, ha="left", fontweight="bold")
    save(fig, CH / "04_param_robustness.png")

    # cost sensitivity
    fig, ax = plt.subplots(figsize=(8, 4))
    cs = stress[stress.test.str.startswith("cost")].copy()
    cs["bps"] = cs.test.str.extract(r"(\d+)").astype(int)
    for i, n in enumerate(results):
        d = cs[cs.strategy == n]
        ax.plot(d.bps, d.sharpe_full, marker="o", ms=5, color=PALETTE[i], label=n)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("One-way cost (bps per unit turnover)")
    ax.set_ylabel("Sharpe, full period")
    ax.set_title("Cost sensitivity")
    ax.legend(loc="lower left")
    save(fig, CH / "05_cost_sensitivity.png")

    # correlation heatmap (diverging: blue neg, orange pos, gray mid)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#0b0b0b")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Monthly return correlation between strategies")
    save(fig, CH / "06_correlation.png")


if __name__ == "__main__":
    main()
