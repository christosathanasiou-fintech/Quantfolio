# Trend-Enhanced Fama-French Factor Portfolio
### A Systematic Long-Short Equity Strategy with CTA-Style Factor Timing & Volatility Targeting

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Research-orange.svg)

---

## 1. Project Overview
This project implements an institutional-grade **multi-layer systematic equity strategy** that combines Fama-French 5-Factor cross-sectional equity portfolios with a CTA-style trend overlay (Factor Momentum).

## 2. Interactive Web Dashboard (Streamlit)
The project includes a production-ready dashboard built with **Streamlit** to visualize backtest performances, risk diagnostics, and equity curves dynamically.

### How to Run Locally:
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `streamlit run factor_trend_strategy_colab.py`

## 3. Resume Description
> **Trend-Enhanced Multi-Factor Equity Strategy (Python)** — Designed and backtested a systematic long-short equity strategy combining Fama-French 5-factor cross-sectional portfolios with CTA-style factor-momentum timing and ex-ante volatility targeting. Implemented vectorized rolling regressions across a 100-stock universe, EWMA trend signals, transaction-cost-aware backtesting, and Newey-West statistical validation.