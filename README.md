# Elite Research: systematic strategy research

A reproducible pipeline that finds, backtests and stress-tests systematic
trading strategies on liquid US equity sector ETFs and cross-asset ETFs.

**Read the findings in [REPORT.md](REPORT.md).**

```bash
pip install -r requirements.txt
python run_research.py            # full run, ~30 s once data is cached
python run_research.py --refresh  # re-download Yahoo / FRED data
```

What the pipeline does:

- **Market regime scan:** trend, volatility, the rates curve, credit spreads, breadth, and sector relative strength.
- **Six strategy families plus a negative control:** cross-asset trend, sector momentum, RSI-2 mean reversion, dual momentum, turn-of-month, vol-managed equity, and overnight drift.
- **Realistic backtesting:** next-close execution lag, turnover-based costs, cash earning T-bills, and leverage financing.
- **In-sample / out-of-sample split** (2001–14 / 2015–26), 175-config parameter grids, and walk-forward selection.
- **Overfitting controls:** Deflated Sharpe ratio across all trials, block-bootstrap confidence intervals, and cost/lag stress tests.
- **Regime analysis:** crisis windows, VIX regimes, calendar years, strategy correlation, and a pre-registered ensemble.

Outputs go to `results/` (CSV/JSON tables) and `results/charts/` (PNG figures).
