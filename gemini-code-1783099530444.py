# ============================================================================
#  TREND-ENHANCED FAMA-FRENCH FACTOR PORTFOLIO
#  A Systematic Long-Short Equity Strategy with CTA-Style Factor Timing
#  and Ex-Ante Volatility Targeting
#
#  Author  : Chris
#  Stack   : Python 3.10+ | yfinance | pandas_datareader | statsmodels
#  Usage   : Copy each "CELL" block into a separate Google Colab cell.
#  License : MIT — research / educational use only. Not investment advice.
# ============================================================================


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 1 — ENVIRONMENT SETUP & INSTALLS                                ║
# ╚══════════════════════════════════════════════════════════════════════╝
import warnings
warnings.filterwarnings("ignore")

import os
import time
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as pdr
import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Global plotting style ───────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize":   (13, 6),
    "figure.dpi":       110,
    "axes.grid":        True,
    "grid.alpha":       0.30,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.size":        11,
})
sns.set_palette("deep")

# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("factor-trend")

log.info("Environment ready.")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 2 — CONFIGURATION                                               ║
# ╚══════════════════════════════════════════════════════════════════════╝
@dataclass
class Config:
    start_date: str = "2010-01-01"
    end_date:   str = None                       

    universe: tuple = (
        "AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ADBE","AMD","INTC","CSCO",
        "TXN","QCOM","IBM","NOW","INTU","AMAT","MU","ADI","LRCX","KLAC",
        "GOOGL","META","NFLX","DIS","CMCSA","TMUS","VZ","T",
        "AMZN","TSLA","HD","MCD","NKE","LOW","SBUX","TJX","BKNG","GM","F",
        "PG","KO","PEP","COST","WMT","MDLZ","CL","KMB","GIS","KHC",
        "UNH","JNJ","LLY","PFE","MRK","ABBV","TMO","ABT","DHR","BMY",
        "AMGN","GILD","CVS","MDT","ISRG",
        "JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP","USB",
        "PNC","TFC","CB","MMC","AIG",
        "CAT","BA","HON","UNP","UPS","RTX","LMT","GE","DE","MMM",
        "FDX","EMR","ETN","CSX","NSC",
        "XOM","CVX","COP","SLB","EOG","PSX","MPC","OXY",
        "LIN","APD","SHW","FCX","NEM","AMT","PLD","SPG","NEE","DUK","SO","D",
    )
    benchmark: str = "SPY"

    beta_window:   int = 252     
    n_deciles:     int = 10      
    min_history:   int = 300     
    factors: tuple = ("Mkt-RF", "SMB", "HML", "RMW", "CMA")
    traded_factors: tuple = ("SMB", "HML", "RMW", "CMA")

    trend_fast:   int = 21       
    trend_slow:   int = 126      
    trend_vol_win: int = 63      
    signal_cap:   float = 1.0    

    vol_target:   float = 0.10   
    vol_window:   int = 63       
    max_leverage: float = 2.0    

    tc_bps:       float = 5.0    

    trading_days: int = 252
    seed:         int = 42

CFG = Config()
ANNUALIZE = np.sqrt(CFG.trading_days)
log.info(f"Config loaded | universe={len(CFG.universe)} stocks")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 3 — DATA COLLECTION I: LOCAL FAMA-FRENCH 5 FACTORS              ║
# ╚══════════════════════════════════════════════════════════════════════╝
def load_ff5_factors(start: str, end: str = None) -> pd.DataFrame:
    """
    Loads Fama-French 5 factors from a local CSV file to prevent connection issues on Streamlit Cloud.
    If the file is missing, it falls back to downloading it and caches it locally.
    """
    csv_filename = "fama_french_5_factors.csv"
    
    if os.path.exists(csv_filename):
        log.info(f"Loading Fama-French factors from local storage: {csv_filename}")
        ff = pd.read_csv(csv_filename, index_col=0, parse_dates=True)
        ff = ff.loc[start:]
        if end:
            ff = ff.loc[:end]
        return ff
    else:
        log.warning(f"{csv_filename} not found locally! Downloading fallback data from Ken French Library...")
        dataset = "F-F_Research_Data_5_Factors_2x3_daily"
        try:
            raw = pdr.DataReader(dataset, "famafrench", start=start, end=end)
            ff = raw[0].copy()                    
            ff.index = pd.to_datetime(ff.index)
            ff = ff / 100.0                       
            # Cache locally
            ff.to_csv(csv_filename)
            log.info(f"Successfully downloaded and cached factors to {csv_filename}")
            return ff
        except Exception as exc:
            raise ConnectionError(f"Could not download or find local {csv_filename}: {exc}")

ff5 = load_ff5_factors(CFG.start_date, CFG.end_date)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 4 — DATA COLLECTION II: EQUITY PRICES (yfinance)                ║
# ╚══════════════════════════════════════════════════════════════════════╝
def load_prices(tickers, start: str, end: str = None, min_coverage: float = 0.90) -> pd.DataFrame:
    tickers = list(dict.fromkeys(tickers))        
    try:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True)["Close"]
    except Exception as exc:
        raise ConnectionError(f"yfinance batch download failed: {exc}")

    if isinstance(raw, pd.Series):                
        raw = raw.to_frame(tickers[0])

    coverage = raw.notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    prices = raw[keep].sort_index()
    log.info(f"Prices loaded: {prices.shape[1]} tickers × {prices.shape[0]} days")
    return prices

prices    = load_prices(CFG.universe, CFG.start_date, CFG.end_date)
spy_price = load_prices([CFG.benchmark], CFG.start_date, CFG.end_date)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 5 — DATA CLEANING & FEATURE ENGINEERING                         ║
# ╚══════════════════════════════════════════════════════════════════════╝
def build_returns(prices: pd.DataFrame, ff: pd.DataFrame, winsor: float = 0.20):
    rets = prices.pct_change().iloc[1:]
    rets = rets.clip(lower=-winsor, upper=winsor)

    idx = rets.index.intersection(ff.index)
    if len(idx) < CFG.beta_window * 2:
        raise ValueError("Insufficient overlapping history between prices and factor data.")
    rets, ff_a = rets.loc[idx], ff.loc[idx]

    excess = rets.sub(ff_a["RF"], axis=0)
    log.info(f"Aligned panel: {excess.shape[0]} days × {excess.shape[1]} stocks")
    return excess, ff_a

excess_ret, ff5 = build_returns(prices, ff5)
spy_ret = spy_price[CFG.benchmark].pct_change().reindex(excess_ret.index).fillna(0.0)
rebalance_dates = excess_ret.groupby(excess_ret.index.to_period("M")).tail(1).index


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 6 — LAYER 1: FACTOR ENGINE                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
def estimate_betas_at(date: pd.Timestamp, excess: pd.DataFrame, ff: pd.DataFrame, window: int) -> pd.DataFrame:
    win_idx = excess.index[excess.index <= date][-window:]
    if len(win_idx) < window:
        return None                               

    Y = excess.loc[win_idx]
    X = ff.loc[win_idx, list(CFG.factors)].values
    X = np.column_stack([np.ones(len(X)), X])     

    valid = Y.notna().mean() >= 0.95
    Y = Y.loc[:, valid].fillna(0.0)
    if Y.shape[1] < 30:
        return None                               

    coefs, *_ = np.linalg.lstsq(X, Y.values, rcond=None)
    betas = pd.DataFrame(coefs[1:].T, index=Y.columns, columns=list(CFG.factors))
    return betas

def build_factor_sleeves(excess: pd.DataFrame, ff: pd.DataFrame, rebal_dates) -> dict:
    sleeves, weights_panel = {}, {}
    daily_index = excess.index

    for factor in CFG.traded_factors:
        w = pd.DataFrame(0.0, index=daily_index, columns=excess.columns)
        for i, rd in enumerate(rebal_dates):
            betas = estimate_betas_at(rd, excess, ff, CFG.beta_window)
            if betas is None:
                continue
            z = (betas[factor] - betas[factor].mean()) / betas[factor].std()
            n_bucket = max(len(z) // CFG.n_deciles, 3)

            longs  = z.nlargest(n_bucket).index
            shorts = z.nsmallest(n_bucket).index

            current_w = pd.Series(0.0, index=excess.columns)
            current_w[longs]  =  1.0 / n_bucket   
            current_w[shorts] = -1.0 / n_bucket   

            nxt = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else daily_index[-1]
            hold = daily_index[(daily_index > rd) & (daily_index <= nxt)]
            w.loc[hold] = current_w.values

        sleeve_ret = (w * excess).sum(axis=1)
        sleeves[factor], weights_panel[factor] = sleeve_ret, w

    return sleeves, weights_panel

sleeves, sleeve_weights = build_factor_sleeves(excess_ret, ff5, rebalance_dates)
sleeve_df = pd.DataFrame(sleeves).fillna(0.0)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 7 — LAYER 2: CTA TREND OVERLAY (FACTOR MOMENTUM)                ║
# ╚══════════════════════════════════════════════════════════════════════╝
def trend_signal(sleeve_returns: pd.DataFrame) -> pd.DataFrame:
    idx = (1.0 + sleeve_returns).cumprod()
    fast = idx.ewm(span=CFG.trend_fast,  adjust=False).mean()
    slow = idx.ewm(span=CFG.trend_slow, adjust=False).mean()
    raw  = fast - slow
    norm = raw / idx.diff().rolling(CFG.trend_vol_win).std().replace(0, np.nan)
    sig  = np.tanh(norm).clip(-CFG.signal_cap, CFG.signal_cap)
    return sig.shift(1).fillna(0.0)               

signals = trend_signal(sleeve_df)
trend_sleeve_ret = (signals * sleeve_df).mean(axis=1)
static_sleeve_ret = sleeve_df.mean(axis=1)        


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 8 — LAYER 3: VOLATILITY TARGETING                               ║
# ╚══════════════════════════════════════════════════════════════════════╝
def vol_target(returns: pd.Series, target: float = CFG.vol_target, window: int = CFG.vol_window, max_lev: float = CFG.max_leverage):
    ewma_vol = returns.ewm(span=window, adjust=False).std() * ANNUALIZE
    lev = (target / ewma_vol.replace(0, np.nan)).shift(1)   
    lev = lev.clip(upper=max_lev).fillna(0.0)
    return returns * lev, lev

final_ret, leverage = vol_target(trend_sleeve_ret)

def apply_costs(sleeve_weights: dict, signals: pd.DataFrame, leverage: pd.Series, gross_ret: pd.Series, tc_bps: float = CFG.tc_bps) -> pd.Series:
    n = len(CFG.traded_factors)
    total_w = None
    for f in CFG.traded_factors:
        eff = sleeve_weights[f].mul(signals[f], axis=0).mul(leverage, axis=0) / n
        total_w = eff if total_w is None else total_w + eff

    turnover = total_w.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * tc_bps / 10_000.0
    return gross_ret - cost

net_ret = apply_costs(sleeve_weights, signals, leverage, final_ret)
variants = pd.DataFrame({
    "Static Factor (EW)":        static_sleeve_ret,
    "Trend-Enhanced":            trend_sleeve_ret,
    "Trend + VolTarget (net)":   net_ret,
    "SPY":                       spy_ret,
}).dropna()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 13 — VISUALIZATION & STREAMLIT DASHBOARD                        ║
# ╚══════════════════════════════════════════════════════════════════════╝
import sys

# Check if the script is running inside a Streamlit application instance
if "streamlit" in sys.modules or any("streamlit" in arg for arg in sys.argv):
    import streamlit as st
    st.set_page_config(page_title="Quantfolio Engine", layout="wide")
    st.title("📊 Quantfolio — Multi-Factor Portfolio Analytics")
    st.markdown("An institutional-grade pipeline combining Fama-French multi-factor decoding with systematic CTA trend overlays.")

    st.subheader("Performance Metrics Summary")
    st.dataframe(variants.describe())

    st.subheader("Cumulative Growth Strategy Comparison")
    fig_cum = go.Figure()
    for col in variants.columns:
        fig_cum.add_trace(go.Scatter(x=variants.index, y=(1 + variants[col]).cumprod(), name=col))
    fig_cum.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Growth of $1")
    st.plotly_chart(fig_cum, use_container_width=True)
else:
    # Fallback to standard matplotlib visualizations if executed as a script/notebook
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in variants.columns:
        ax.plot((1 + variants[col]).cumprod(), label=col)
    ax.set_yscale("log")
    ax.legend()
    plt.title("Cumulative Growth (Notebook Mode)")
    plt.show()