# Trend-Enhanced Fama-French Factor Portfolio — Streamlit App
### A Systematic Long-Short Equity Strategy with CTA-Style Factor Timing & Volatility Targeting

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Research-orange.svg)

An interactive research dashboard implementing a three-layer institutional-style strategy:

- **Layer 1 — Cross-Sectional Factor Engine:** Rolling 252-day FF5 betas on ~110 liquid US large caps → monthly-rebalanced decile long-short sleeves per factor (SMB, HML, RMW, CMA).
- **Layer 2 — CTA Trend Overlay (Factor Momentum):** EWMA(21)−EWMA(126) crossover on each sleeve's own return stream, vol-normalized, tanh-squashed to [−1, +1], lagged one day.
- **Layer 3 — Volatility Targeting:** Combined portfolio scaled to a 10% ex-ante annualized vol target with a 2× leverage cap, plus a turnover-based transaction-cost model (5 bps one-way).

The same architecture used in AQR's *Factor Momentum Everywhere* (Gupta & Kelly, 2019) and Man AHL's trend-overlay products.

---

## 🚀 Run the app

### Locally
```bash
git clone <your-repo-url>
cd trend-factor-app
pip install -r requirements.txt
streamlit run app.py
```
The app opens at `http://localhost:8501`. Set parameters in the sidebar and press **Run backtest**. The first run downloads ~15 years of prices + factors (1–3 minutes); everything is cached for 24 h afterwards.

### Deploy to Streamlit Community Cloud (free)
1. Push this folder to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repo, branch `main`, main file `app.py` → **Deploy**.

No API keys needed — Yahoo Finance and the Ken French Data Library are both free and keyless.

---

## 📁 Project structure

```
trend-factor-app/
├── app.py             ← Streamlit UI: sidebar controls, tabs, Plotly charts
├── strategy.py        ← Core engine: data loaders, 3 layers, analytics
├── test_pipeline.py   ← Synthetic end-to-end smoke test (no network needed)
├── requirements.txt
└── README.md
```

Run the smoke test any time you modify the engine:
```bash
python test_pipeline.py   # prints stats table + "✅ PIPELINE OK"
```

---

## 🖥 App features

| Tab | Contents |
|---|---|
| **📈 Performance** | Log-scale equity curves + drawdowns for all 3 variants vs. SPY; per-sleeve equity curves |
| **🎛 Trend signals** | Factor-momentum signal heatmap, current positioning bar, cumulative sleeve contribution |
| **🛡 Risk diagnostics** | Vol-target leverage, realized vs. target vol, rolling 1Y Sharpe, monthly return heatmap |
| **📊 Statistics** | Full stats table (CAGR, Sharpe, Sortino, Calmar, MaxDD, hit rate, skew, kurtosis) + Newey-West FF5 alpha |
| **🔬 Sensitivity** | Trend-overlay Sharpe across a (fast, slow) EWMA grid — overfitting check |
| **📥 Export** | CSV downloads (variant returns, signals) + on-demand QuantStats HTML tearsheet |

Every strategy parameter is adjustable live from the sidebar: sample period, universe size, OLS window, decile count, traded sleeves, EWMA spans, vol target, leverage cap, transaction costs.

---

## 🔧 What changed vs. the original Colab script

| Issue in Colab version | Fix in this version |
|---|---|
| Colab-only `display()` calls | Removed; Streamlit renders dataframes natively |
| Everything executed at import (top-level script) | Refactored into pure functions in `strategy.py` — cacheable & testable |
| `pandas_datareader` FF5 download breaks on some pandas versions | Added direct-ZIP fallback straight from the Ken French site |
| yfinance single-ticker / MultiIndex column edge cases | Handled both layouts explicitly |
| Possible division by zero in cross-sectional z-score | Guarded (`std == 0` → skip rebalance) |
| Betas re-estimated once *per factor* per rebalance date (4× redundant work) | Estimated once per date, shared across sleeves — ~4× faster |
| matplotlib/seaborn static charts, `plt.show()` | All charts rebuilt in Plotly (interactive, zoomable) |
| QuantStats tearsheet crashes on new pandas | Wrapped in try/except; generated on demand with a download button |
| No progress feedback on slow steps | Streamlit spinners + `@st.cache_data` (24 h TTL) |

---

## 📊 Data sources

| Source | Data | Access | Cost |
|---|---|---|---|
| Yahoo Finance (`yfinance`) | Daily adjusted prices | No key | Free |
| Ken French Data Library | FF5 daily factors + RF | No key | Free |

**Known limitation:** the fixed present-day ticker list carries **survivorship bias** — standard for free-data research and acknowledged in the app footer.

## 🔬 Methodology safeguards

- Trend signals **shifted 1 day** → no look-ahead.
- Vol-target leverage decided on **yesterday's** vol estimate.
- Daily returns winsorized at ±20% (guards against bad adjustments, keeps real crashes).
- Newey-West (HAC, 21 lags) t-stats on strategy alpha vs. FF5.
- Parameter-sensitivity grid as an overfitting check.

## 📈 Potential upgrades

1. Full Russell 1000 universe via a bulk data provider
2. Add 12-1 time-series momentum & factor-valuation spreads as timing inputs
3. HMM regime gating of the trend overlay
4. Almgren-Chriss execution cost model
5. Barra-style factor covariance for risk scaling
6. Live paper trading via Alpaca

---

*Research/educational project. Not investment advice. Historical backtests do not guarantee future performance.*
