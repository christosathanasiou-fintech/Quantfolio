# Trend-Enhanced Fama-French Factor Portfolio
### A Systematic Long-Short Equity Strategy with CTA-Style Factor Timing & Volatility Targeting

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Research-orange.svg)

---

## 1. Project Overview

This project implements an institutional-grade **multi-layer systematic equity strategy** that combines two of the most robust ideas in quantitative finance:

- **Layer 1 — Cross-Sectional Factor Engine:** Ranks a large-cap US equity universe on rolling Fama-French 5-Factor (FF5) exposures and constructs monthly-rebalanced long-short factor portfolios (decile spreads).
- **Layer 2 — CTA Trend Overlay (Factor Momentum):** Applies EWMA crossover trend signals *to the factor return streams themselves*, dynamically scaling each factor sleeve up or down based on whether the factor is currently "working".
- **Layer 3 — Volatility Targeting:** Scales the combined portfolio to a constant ex-ante annualized volatility target (10%), producing a smoother, more allocable return stream.

The result is a **Trend-Enhanced Factor Portfolio** — the same architecture used in AQR's "Factor Momentum Everywhere" research and Man AHL's trend-overlay products.

## 2. Real-World Finance Use Case

| Institution | How they use this exact architecture |
|---|---|
| **AQR Capital** | "Factor Momentum Everywhere" (Gupta & Kelly, 2019) — timing factors with their own momentum |
| **Man AHL** | Trend overlays on systematic equity books |
| **Robeco / BlackRock** | Volatility-targeted multi-factor UCITS products |
| **Multi-strat pods (Millennium, Balyasny)** | Factor-neutral books with dynamic factor risk budgeting |

Factor returns are cyclical: value can underperform for years, momentum crashes violently after market bottoms. Static factor allocation suffers through these regimes. Factor momentum — trending each factor's own return stream — is one of the few documented ways to time factors that survives out-of-sample testing.

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                             │
│  yfinance (prices)  │  Ken French Library (FF5 factors)     │
│  FRED (risk-free)   │                                       │
└──────────────┬──────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────┐
│              LAYER 1: FACTOR ENGINE                         │
│  Rolling 252d OLS: rᵢ = α + β·FF5 + ε                       │
│  Monthly ranks → decile long-short portfolios per factor    │
└──────────────┬──────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────┐
│              LAYER 2: TREND OVERLAY (CTA)                   │
│  Signal = EWMA(fast) − EWMA(slow) of factor returns         │
│  Normalized by ex-ante vol → factor weight ∈ [−1, +1]       │
└──────────────┬──────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────┐
│              LAYER 3: VOLATILITY TARGETING                  │
│  Portfolio scaled to 10% ex-ante annualized volatility      │
│  Leverage cap at 2.0x                                       │
└──────────────┬──────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────┐
│         BACKTEST + ANALYTICS + TEARSHEET                    │
│  Transaction costs │ Sharpe/Sortino/Calmar │ Attribution    │
└─────────────────────────────────────────────────────────────┘
```

## 4. Required APIs and Data Sources

| Source | Data | Access | Cost |
|---|---|---|---|
| **Yahoo Finance** (`yfinance`) | Daily OHLCV for equity universe | No key needed | Free |
| **Ken French Data Library** (`pandas_datareader`) | FF5 daily factors + RF rate | No key needed | Free |
| **FRED** (optional) | 3M T-Bill for robustness check | Free API key | Free |

## 5. Required Python Libraries

```
yfinance>=0.2.40
pandas>=2.0
numpy>=1.24
pandas_datareader>=0.10
statsmodels>=0.14
scipy>=1.10
matplotlib>=3.7
seaborn>=0.12
plotly>=5.18
quantstats>=0.0.62
```

## 6. Folder / File Structure

Although built in a single Colab notebook, the code is organized as if it were a package (mirrors GitHub layout):

```
trend-enhanced-factor-portfolio/
│
├── README.md                     ← this file
├── requirements.txt
├── notebook/
│   └── factor_trend_strategy.ipynb
└── src/                          (conceptual sections inside the notebook)
    ├── config.py                 ← Cell 2: all parameters in one place
    ├── data_loader.py            ← Cells 3–4: prices + factors download
    ├── factor_engine.py          ← Cells 5–6: rolling betas, decile portfolios
    ├── trend_overlay.py          ← Cell 7: EWMA factor momentum signals
    ├── vol_targeting.py          ← Cell 8: risk scaling
    ├── backtest.py               ← Cell 9: combined engine + costs
    ├── analytics.py              ← Cell 10: performance metrics
    └── visualization.py          ← Cells 11–13: all charts
```

## 7. Step-by-Step Build Guide

1. **Setup** — install libraries, set the configuration block (universe, windows, targets).
2. **Data collection** — download FF5 daily factors (Ken French) and daily prices for ~100 liquid large-caps.
3. **Cleaning** — align calendars, forward-fill limits, compute excess returns, survivorship note.
4. **Factor engine** — rolling 252-day OLS per stock → factor betas → monthly decile long-short portfolios per factor.
5. **Trend overlay** — EWMA(21) − EWMA(126) crossover on each factor sleeve's cumulative return, normalized by ex-ante vol, clipped to [−1, +1].
6. **Vol targeting** — scale combined portfolio to 10% annualized ex-ante vol using a 63-day EWMA vol estimate; cap leverage at 2×.
7. **Backtest** — apply 5 bps one-way transaction costs on turnover; compare 3 variants (static / trend / trend+voltarget).
8. **Analytics** — Sharpe, Sortino, Calmar, max drawdown, hit rate, factor attribution, rolling Sharpe.
9. **Visualization** — equity curves, factor signal heatmap, exposure chart, drawdown chart, monthly return heatmap, QuantStats tearsheet.

## 8. Data Collection Pipeline

- **FF5 factors:** `pandas_datareader.famafrench` → dataset `F-F_Research_Data_5_Factors_2x3_daily`. Returns come in percent; converted to decimals.
- **Prices:** batch download via `yfinance` with retry logic and per-ticker failure tolerance (a failed ticker is logged and dropped, never crashes the run).
- **Alignment:** inner-join on trading calendar; stocks require ≥ 90% data availability over the sample to enter the universe.

## 9. Data Cleaning & Feature Engineering

- Log-return computation with NaN-safe handling.
- Winsorization of daily returns at ±20% to guard against split/adjustment glitches.
- Excess returns: `r_excess = r − RF` (daily risk-free from the FF file).
- Rolling betas standardized cross-sectionally each month (z-scores) before ranking.
- Ex-ante volatility: 63-day exponentially weighted std, annualized.

## 10. Core Models / Algorithms

| Component | Method |
|---|---|
| Factor exposures | Rolling 252-day OLS (`numpy.linalg.lstsq`, vectorized) |
| Portfolio construction | Decile spread: long top 10 %, short bottom 10 %, equal weight |
| Factor timing | EWMA crossover trend signal `s = (EWMA₂₁ − EWMA₁₂₆)/σ`, tanh-squashed |
| Risk management | Ex-ante vol targeting, leverage cap, turnover-based cost model |
| Statistical validation | Newey-West t-stats on strategy alphas vs. FF5 |

## 11. Visualizations & Dashboard Components

1. Cumulative return comparison — static vs. trend-enhanced vs. vol-targeted (log scale)
2. Factor sleeve equity curves (5 panels)
3. Trend signal heatmap (factor × time, red-blue diverging)
4. Rolling 1-year Sharpe ratio of all variants
5. Drawdown curve comparison
6. Monthly return heatmap (year × month)
7. Factor contribution stacked area chart
8. Realized vs. target volatility tracking chart
9. Full QuantStats HTML tearsheet vs. SPY benchmark

## 12. Performance Metrics

- CAGR, annualized volatility, **Sharpe**, **Sortino**, **Calmar**
- Maximum drawdown, average drawdown duration
- Hit rate (daily & monthly), skewness, kurtosis
- Annualized alpha & beta vs. SPY and vs. FF5 (Newey-West t-stats)
- Average annual turnover and total cost drag

## 13. Final Deliverables

- ✅ One polished Colab notebook (copy-paste-per-cell, fully commented)
- ✅ This README for the GitHub repo
- ✅ QuantStats HTML tearsheet
- ✅ All charts exportable as PNG (300 dpi)

## 14. Resume Description

> **Trend-Enhanced Multi-Factor Equity Strategy (Python)** — Designed and backtested a systematic long-short equity strategy combining Fama-French 5-factor cross-sectional portfolios with CTA-style factor-momentum timing and ex-ante volatility targeting. Implemented vectorized rolling regressions across a 100-stock universe, EWMA trend signals, transaction-cost-aware backtesting, and Newey-West statistical validation; improved risk-adjusted performance (Sharpe) vs. the static factor benchmark and produced an institutional QuantStats tearsheet.

## 15. Potential Upgrades

1. **Universe expansion** — full Russell 1000 via bulk data provider.
2. **Signal blend** — add 12-1 time-series momentum and factor-valuation spreads (value-of-value) as timing inputs.
3. **Regime conditioning** — gate the trend overlay with an HMM regime detector (see Project 6).
4. **Execution realism** — Almgren-Chriss cost model instead of flat bps (see Project 19).
5. **Risk model integration** — replace naive vol scaling with a Barra-style factor covariance matrix (see Project 16).
6. **Live paper trading** — daily cron job + broker API (Alpaca) for out-of-sample tracking.

---

*Research/educational project. Not investment advice. Historical backtests do not guarantee future performance.*
