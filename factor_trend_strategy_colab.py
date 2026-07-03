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
# In Colab, run this cell first. The `-q` flag keeps output clean.

# !pip install -q yfinance pandas_datareader statsmodels quantstats plotly seaborn

import warnings
warnings.filterwarnings("ignore")

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

# ── Logging: professional runs log, they don't print randomly ──────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("factor-trend")

log.info("Environment ready.")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 2 — CONFIGURATION                                               ║
# ║  Every tunable parameter of the strategy lives here.                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

@dataclass
class Config:
    # ── Sample period ───────────────────────────────────────────────────
    start_date: str = "2010-01-01"
    end_date:   str = None                       # None → today

    # ── Universe: ~100 liquid US large caps across all GICS sectors.
    #    NOTE: a fixed present-day list carries survivorship bias — this is
    #    acknowledged in the README and is standard for free-data research.
    universe: tuple = (
        # Technology
        "AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ADBE","AMD","INTC","CSCO",
        "TXN","QCOM","IBM","NOW","INTU","AMAT","MU","ADI","LRCX","KLAC",
        # Communication / Media
        "GOOGL","META","NFLX","DIS","CMCSA","TMUS","VZ","T",
        # Consumer Discretionary
        "AMZN","TSLA","HD","MCD","NKE","LOW","SBUX","TJX","BKNG","GM","F",
        # Consumer Staples
        "PG","KO","PEP","COST","WMT","MDLZ","CL","KMB","GIS","KHC",
        # Health Care
        "UNH","JNJ","LLY","PFE","MRK","ABBV","TMO","ABT","DHR","BMY",
        "AMGN","GILD","CVS","MDT","ISRG",
        # Financials
        "JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP","USB",
        "PNC","TFC","CB","MMC","AIG",
        # Industrials
        "CAT","BA","HON","UNP","UPS","RTX","LMT","GE","DE","MMM",
        "FDX","EMR","ETN","CSX","NSC",
        # Energy
        "XOM","CVX","COP","SLB","EOG","PSX","MPC","OXY",
        # Materials / Real Estate / Utilities
        "LIN","APD","SHW","FCX","NEM","AMT","PLD","SPG","NEE","DUK","SO","D",
    )
    benchmark: str = "SPY"

    # ── Layer 1: factor engine ──────────────────────────────────────────
    beta_window:   int = 252     # rolling OLS lookback (1 trading year)
    n_deciles:     int = 10      # decile portfolios → top/bottom 10%
    min_history:   int = 300     # min obs before a stock can be ranked
    factors: tuple = ("Mkt-RF", "SMB", "HML", "RMW", "CMA")
    # Sleeves we actually trade (Mkt-RF timing is left to the overlay
    # literature; we trade the 4 style factors cross-sectionally).
    traded_factors: tuple = ("SMB", "HML", "RMW", "CMA")

    # ── Layer 2: CTA trend overlay on factor sleeves ────────────────────
    trend_fast:   int = 21       # ~1 month EWMA
    trend_slow:   int = 126      # ~6 month EWMA
    trend_vol_win: int = 63      # normalization window
    signal_cap:   float = 1.0    # |signal| ≤ 1 after tanh squash

    # ── Layer 3: volatility targeting ───────────────────────────────────
    vol_target:   float = 0.10   # 10% annualized
    vol_window:   int = 63       # ex-ante vol estimation window (EWMA)
    max_leverage: float = 2.0    # hard leverage cap

    # ── Frictions ───────────────────────────────────────────────────────
    tc_bps:       float = 5.0    # one-way transaction cost in basis points

    # ── Misc ────────────────────────────────────────────────────────────
    trading_days: int = 252
    seed:         int = 42

CFG = Config()
ANNUALIZE = np.sqrt(CFG.trading_days)
log.info(f"Config loaded | universe={len(CFG.universe)} stocks | "
         f"{CFG.start_date} → {CFG.end_date or 'today'}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 3 — DATA COLLECTION I: FAMA-FRENCH 5 FACTORS                    ║
# ║  Source: Ken French Data Library (via pandas_datareader).             ║
# ╚══════════════════════════════════════════════════════════════════════╝

def load_ff5_factors(start: str, end: str = None,
                     max_retries: int = 3) -> pd.DataFrame:
    """
    Download the daily Fama-French 5-factor dataset.

    Returns a DataFrame indexed by date with columns:
    ['Mkt-RF','SMB','HML','RMW','CMA','RF'] — in DECIMAL units.

    The Ken French library serves percent values; we convert to decimals
    immediately so every downstream calculation is unit-consistent.
    """
    dataset = "F-F_Research_Data_5_Factors_2x3_daily"
    for attempt in range(1, max_retries + 1):
        try:
            raw = pdr.DataReader(dataset, "famafrench", start=start, end=end)
            ff = raw[0].copy()                    # table 0 = daily data
            ff.index = pd.to_datetime(ff.index)
            ff = ff / 100.0                       # percent → decimal
            log.info(f"FF5 factors loaded: {ff.shape[0]} days "
                     f"({ff.index[0].date()} → {ff.index[-1].date()})")
            return ff
        except Exception as exc:
            log.warning(f"FF5 download attempt {attempt}/{max_retries} "
                        f"failed: {exc}")
            time.sleep(2 * attempt)               # linear backoff
    raise ConnectionError("Could not download FF5 factors after retries. "
                          "Check network access to the Ken French library.")

ff5 = load_ff5_factors(CFG.start_date, CFG.end_date)
display(ff5.tail(3))


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 4 — DATA COLLECTION II: EQUITY PRICES (yfinance)                ║
# ║  Batch download with per-ticker fault tolerance.                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

def load_prices(tickers, start: str, end: str = None,
                min_coverage: float = 0.90) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.

    Fault tolerance:
      * A ticker that fails to download is logged and DROPPED — one bad
        symbol never crashes the pipeline.
      * Tickers with < `min_coverage` non-NaN observations over the
        sample are removed (delisted / too young / illiquid data).
    """
    tickers = list(dict.fromkeys(tickers))        # dedupe, keep order
    try:
        raw = yf.download(
            tickers, start=start, end=end,
            auto_adjust=True, progress=False, threads=True,
        )["Close"]
    except Exception as exc:
        raise ConnectionError(f"yfinance batch download failed: {exc}")

    if isinstance(raw, pd.Series):                # single-ticker edge case
        raw = raw.to_frame(tickers[0])

    # ── Coverage filter ────────────────────────────────────────────────
    coverage = raw.notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    dropped = sorted(set(raw.columns) - set(keep))
    if dropped:
        log.warning(f"Dropped {len(dropped)} tickers with poor coverage: "
                    f"{dropped}")
    prices = raw[keep].sort_index()
    log.info(f"Prices loaded: {prices.shape[1]} tickers × "
             f"{prices.shape[0]} days")
    return prices

prices    = load_prices(CFG.universe, CFG.start_date, CFG.end_date)
spy_price = load_prices([CFG.benchmark], CFG.start_date, CFG.end_date)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 5 — DATA CLEANING & FEATURE ENGINEERING                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

def build_returns(prices: pd.DataFrame, ff: pd.DataFrame,
                  winsor: float = 0.20):
    """
    1. Simple daily returns from adjusted prices.
    2. Winsorize at ±`winsor` (20%) — protects rolling regressions from
       data glitches (bad adjustments), NOT from real crashes, which stay.
    3. Align to the FF calendar (inner join) and compute EXCESS returns.

    Returns (excess_returns, rf_aligned, market_excess).
    """
    rets = prices.pct_change().iloc[1:]
    rets = rets.clip(lower=-winsor, upper=winsor)

    idx = rets.index.intersection(ff.index)
    if len(idx) < CFG.beta_window * 2:
        raise ValueError("Insufficient overlapping history between prices "
                         "and factor data — extend the sample period.")
    rets, ff_a = rets.loc[idx], ff.loc[idx]

    excess = rets.sub(ff_a["RF"], axis=0)
    log.info(f"Aligned panel: {excess.shape[0]} days × "
             f"{excess.shape[1]} stocks")
    return excess, ff_a

excess_ret, ff5 = build_returns(prices, ff5)
spy_ret = spy_price[CFG.benchmark].pct_change().reindex(excess_ret.index).fillna(0.0)

# Month-end rebalance dates (last trading day of each month in-sample)
rebalance_dates = excess_ret.groupby(excess_ret.index.to_period("M")).tail(1).index
log.info(f"{len(rebalance_dates)} monthly rebalance dates.")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 6 — LAYER 1: FACTOR ENGINE                                      ║
# ║  Rolling FF5 betas → cross-sectional ranks → decile L/S portfolios.   ║
# ╚══════════════════════════════════════════════════════════════════════╝

def estimate_betas_at(date: pd.Timestamp,
                      excess: pd.DataFrame,
                      ff: pd.DataFrame,
                      window: int) -> pd.DataFrame:
    """
    Estimate FF5 betas for every stock using the trailing `window` days
    ending at `date`, via ONE vectorized least-squares solve:

        B = (XᵀX)⁻¹ Xᵀ Y      X: (T×6 incl. intercept)   Y: (T×N)

    numpy.linalg.lstsq solves all N stocks simultaneously — orders of
    magnitude faster than looping statsmodels per stock.
    """
    win_idx = excess.index[excess.index <= date][-window:]
    if len(win_idx) < window:
        return None                               # not enough history yet

    Y = excess.loc[win_idx]
    X = ff.loc[win_idx, list(CFG.factors)].values
    X = np.column_stack([np.ones(len(X)), X])     # add intercept

    # Require near-complete data in-window per stock
    valid = Y.notna().mean() >= 0.95
    Y = Y.loc[:, valid].fillna(0.0)
    if Y.shape[1] < 30:
        return None                               # cross-section too thin

    coefs, *_ = np.linalg.lstsq(X, Y.values, rcond=None)
    betas = pd.DataFrame(coefs[1:].T, index=Y.columns,
                         columns=list(CFG.factors))
    return betas


def build_factor_sleeves(excess: pd.DataFrame, ff: pd.DataFrame,
                         rebal_dates) -> dict:
    """
    For each traded factor, build a monthly-rebalanced decile long-short
    portfolio:  LONG the top decile of beta, SHORT the bottom decile,
    equal-weighted, held until the next rebalance.

    Returns:
      sleeves : dict  factor → daily return Series of the L/S sleeve
      weights : dict  factor → DataFrame of daily portfolio weights
                                (needed later for turnover / costs)
    """
    sleeves, weights_panel = {}, {}
    daily_index = excess.index

    for factor in CFG.traded_factors:
        w = pd.DataFrame(0.0, index=daily_index, columns=excess.columns)
        current_w = pd.Series(0.0, index=excess.columns)

        for i, rd in enumerate(rebal_dates):
            betas = estimate_betas_at(rd, excess, ff, CFG.beta_window)
            if betas is None:
                continue
            # Cross-sectional z-score → robust to beta level shifts
            z = (betas[factor] - betas[factor].mean()) / betas[factor].std()
            n_bucket = max(len(z) // CFG.n_deciles, 3)

            longs  = z.nlargest(n_bucket).index
            shorts = z.nsmallest(n_bucket).index

            current_w = pd.Series(0.0, index=excess.columns)
            current_w[longs]  =  1.0 / n_bucket   # +100% gross long
            current_w[shorts] = -1.0 / n_bucket   # −100% gross short

            # Hold weights from day AFTER rebalance to next rebalance
            nxt = rebal_dates[i + 1] if i + 1 < len(rebal_dates) \
                  else daily_index[-1]
            hold = daily_index[(daily_index > rd) & (daily_index <= nxt)]
            w.loc[hold] = current_w.values

        sleeve_ret = (w * excess).sum(axis=1)
        sleeves[factor], weights_panel[factor] = sleeve_ret, w
        ann_sr = sleeve_ret.mean() / sleeve_ret.std() * ANNUALIZE
        log.info(f"Sleeve {factor:>4}: Sharpe={ann_sr:5.2f} | "
                 f"AnnVol={sleeve_ret.std()*ANNUALIZE:6.2%}")

    return sleeves, weights_panel

sleeves, sleeve_weights = build_factor_sleeves(excess_ret, ff5, rebalance_dates)
sleeve_df = pd.DataFrame(sleeves).fillna(0.0)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 7 — LAYER 2: CTA TREND OVERLAY (FACTOR MOMENTUM)                ║
# ╚══════════════════════════════════════════════════════════════════════╝

def trend_signal(sleeve_returns: pd.DataFrame) -> pd.DataFrame:
    """
    CTA-style EWMA crossover applied to each FACTOR SLEEVE's own
    cumulative return index (this is 'factor momentum'):

        raw   = EWMA_fast(index) − EWMA_slow(index)
        norm  = raw / (rolling σ of the index level changes)
        sig   = tanh(norm)  ∈ (−1, +1)

    tanh squashing keeps the position response smooth and bounded —
    the same functional form used in Man AHL's published research.

    CRITICAL: the signal is SHIFTED BY 1 DAY before use.
    A signal computed on today's close can only be traded tomorrow.
    """
    idx = (1.0 + sleeve_returns).cumprod()

    fast = idx.ewm(span=CFG.trend_fast,  adjust=False).mean()
    slow = idx.ewm(span=CFG.trend_slow, adjust=False).mean()
    raw  = fast - slow

    norm = raw / idx.diff().rolling(CFG.trend_vol_win).std().replace(0, np.nan)
    sig  = np.tanh(norm).clip(-CFG.signal_cap, CFG.signal_cap)

    return sig.shift(1).fillna(0.0)               # ← no look-ahead

signals = trend_signal(sleeve_df)

# Trend-weighted combination of sleeves (equal risk budget per sleeve)
trend_sleeve_ret = (signals * sleeve_df).mean(axis=1)
static_sleeve_ret = sleeve_df.mean(axis=1)        # benchmark: no timing

log.info(f"Static  combo Sharpe: "
         f"{static_sleeve_ret.mean()/static_sleeve_ret.std()*ANNUALIZE:.2f}")
log.info(f"Trended combo Sharpe: "
         f"{trend_sleeve_ret.mean()/trend_sleeve_ret.std()*ANNUALIZE:.2f}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 8 — LAYER 3: VOLATILITY TARGETING                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

def vol_target(returns: pd.Series,
               target: float = CFG.vol_target,
               window: int = CFG.vol_window,
               max_lev: float = CFG.max_leverage):
    """
    Scale a return stream to a constant ex-ante annualized volatility.

        lev_t = min( target / σ̂_{t-1} , max_lev )

    σ̂ is an EWMA estimate using information up to YESTERDAY (shifted),
    so the leverage decision never peeks at today's return.
    Returns (scaled_returns, leverage_series).
    """
    ewma_vol = returns.ewm(span=window, adjust=False).std() * ANNUALIZE
    lev = (target / ewma_vol.replace(0, np.nan)).shift(1)   # ← lag 1 day
    lev = lev.clip(upper=max_lev).fillna(0.0)
    return returns * lev, lev

final_ret, leverage = vol_target(trend_sleeve_ret)


# ── Transaction costs ───────────────────────────────────────────────────
def apply_costs(sleeve_weights: dict, signals: pd.DataFrame,
                leverage: pd.Series, gross_ret: pd.Series,
                tc_bps: float = CFG.tc_bps) -> pd.Series:
    """
    Turnover-based cost model. Effective stock-level weight =
    sleeve weight × trend signal × leverage / n_sleeves.
    Cost_t = Σ|Δw| × (tc_bps / 10,000). Subtracted from gross returns.
    """
    n = len(CFG.traded_factors)
    total_w = None
    for f in CFG.traded_factors:
        eff = sleeve_weights[f].mul(signals[f], axis=0) \
                               .mul(leverage, axis=0) / n
        total_w = eff if total_w is None else total_w + eff

    turnover = total_w.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * tc_bps / 10_000.0
    log.info(f"Avg annual turnover: {turnover.mean()*CFG.trading_days:,.1f}x "
             f"| Total cost drag: {cost.sum():.2%} over sample")
    return gross_ret - cost

net_ret = apply_costs(sleeve_weights, signals, leverage, final_ret)

# Assemble the three strategy variants for comparison
variants = pd.DataFrame({
    "Static Factor (EW)":        static_sleeve_ret,
    "Trend-Enhanced":            trend_sleeve_ret,
    "Trend + VolTarget (net)":   net_ret,
    "SPY":                       spy_ret,
}).dropna()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 9 — PERFORMANCE ANALYTICS                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

def perf_stats(r: pd.Series, name: str = "") -> pd.Series:
    """Institutional summary statistics for a daily return stream."""
    r = r.dropna()
    if len(r) < 2 or r.std() == 0:
        return pd.Series(dtype=float, name=name)

    cum   = (1 + r).cumprod()
    yrs   = len(r) / CFG.trading_days
    cagr  = cum.iloc[-1] ** (1 / yrs) - 1
    vol   = r.std() * ANNUALIZE
    dd    = cum / cum.cummax() - 1
    downside = r[r < 0].std() * ANNUALIZE

    return pd.Series({
        "CAGR":        f"{cagr:8.2%}",
        "Ann.Vol":     f"{vol:8.2%}",
        "Sharpe":      f"{(r.mean()*CFG.trading_days)/vol:8.2f}",
        "Sortino":     f"{(r.mean()*CFG.trading_days)/downside:8.2f}"
                       if downside > 0 else "     n/a",
        "MaxDD":       f"{dd.min():8.2%}",
        "Calmar":      f"{cagr/abs(dd.min()):8.2f}" if dd.min() < 0 else " n/a",
        "HitRate(d)":  f"{(r > 0).mean():8.2%}",
        "Skew":        f"{r.skew():8.2f}",
        "Kurtosis":    f"{r.kurtosis():8.2f}",
    }, name=name)

stats_table = pd.concat(
    [perf_stats(variants[c], c) for c in variants.columns], axis=1
)
print("═" * 70)
print(" PERFORMANCE SUMMARY")
print("═" * 70)
display(stats_table)

# ── Newey-West alpha of the final strategy vs. FF5 ─────────────────────
def nw_alpha(strategy: pd.Series, ff: pd.DataFrame, lags: int = 21):
    """Annualized FF5 alpha with HAC (Newey-West) standard errors."""
    df = pd.concat([strategy, ff[list(CFG.factors)]], axis=1).dropna()
    X  = sm.add_constant(df.iloc[:, 1:])
    res = sm.OLS(df.iloc[:, 0], X).fit(cov_type="HAC",
                                       cov_kwds={"maxlags": lags})
    ann_alpha = res.params["const"] * CFG.trading_days
    print(f"\nFF5 Newey-West alpha:  {ann_alpha:6.2%} p.a. "
          f"(t-stat = {res.tvalues['const']:.2f})")
    return res

_ = nw_alpha(net_ret, ff5)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 10 — VISUALIZATION I: EQUITY CURVES & DRAWDOWNS                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(13, 9), sharex=True,
    gridspec_kw={"height_ratios": [2.4, 1]},
)
colors = {"Static Factor (EW)": "#8a8a8a",
          "Trend-Enhanced": "#1f77b4",
          "Trend + VolTarget (net)": "#0a3d62",
          "SPY": "#c8a24a"}

for col in variants.columns:
    curve = (1 + variants[col]).cumprod()
    lw = 2.4 if "VolTarget" in col else 1.4
    ax1.plot(curve, label=col, color=colors[col], lw=lw)
    ax2.plot(curve / curve.cummax() - 1, color=colors[col], lw=1.0)

ax1.set_yscale("log")
ax1.set_title("Trend-Enhanced Factor Portfolio — Cumulative Growth of $1 "
              "(log scale)", fontsize=13, fontweight="bold")
ax1.legend(frameon=False, loc="upper left")
ax2.set_title("Drawdowns", fontsize=11)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
plt.tight_layout()
plt.savefig("equity_curves.png", dpi=300, bbox_inches="tight")
plt.show()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 11 — VISUALIZATION II: TREND SIGNAL HEATMAP & SLEEVES           ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── Factor momentum signal heatmap ──────────────────────────────────────
monthly_sig = signals.resample("ME").last()
fig, ax = plt.subplots(figsize=(13, 3.6))
sns.heatmap(
    monthly_sig.T, cmap="RdBu", center=0, vmin=-1, vmax=1,
    cbar_kws={"label": "Trend signal"}, ax=ax,
    xticklabels=[d.strftime("%Y") if d.month == 1 else ""
                 for d in monthly_sig.index],
)
ax.set_title("CTA Trend Signal per Factor Sleeve (monthly)",
             fontweight="bold")
plt.tight_layout()
plt.savefig("signal_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# ── Individual sleeve equity curves ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5.5))
for f in CFG.traded_factors:
    ax.plot((1 + sleeve_df[f]).cumprod(), label=f"{f} L/S sleeve", lw=1.4)
ax.set_title("Factor Sleeve Equity Curves (before timing)",
             fontweight="bold")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("sleeves.png", dpi=300, bbox_inches="tight")
plt.show()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 12 — VISUALIZATION III: RISK MANAGEMENT DIAGNOSTICS             ║
# ╚══════════════════════════════════════════════════════════════════════╝

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# (a) Leverage applied by the vol-targeting layer
axes[0, 0].plot(leverage, color="#0a3d62", lw=0.9)
axes[0, 0].axhline(CFG.max_leverage, ls="--", color="crimson", lw=1,
                   label=f"Leverage cap {CFG.max_leverage}x")
axes[0, 0].set_title("Vol-Target Leverage", fontweight="bold")
axes[0, 0].legend(frameon=False)

# (b) Realized rolling vol vs. target
roll_vol = net_ret.rolling(63).std() * ANNUALIZE
axes[0, 1].plot(roll_vol, color="#1f77b4", lw=1.0, label="Realized 3M vol")
axes[0, 1].axhline(CFG.vol_target, ls="--", color="crimson", lw=1.2,
                   label=f"Target {CFG.vol_target:.0%}")
axes[0, 1].yaxis.set_major_formatter(
    plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
axes[0, 1].set_title("Realized vs. Target Volatility", fontweight="bold")
axes[0, 1].legend(frameon=False)

# (c) Rolling 1Y Sharpe: static vs. final
for col, c in [("Static Factor (EW)", "#8a8a8a"),
               ("Trend + VolTarget (net)", "#0a3d62")]:
    r = variants[col]
    rs = (r.rolling(252).mean() / r.rolling(252).std()) * ANNUALIZE
    axes[1, 0].plot(rs, label=col, color=c, lw=1.3)
axes[1, 0].axhline(0, color="black", lw=0.7)
axes[1, 0].set_title("Rolling 1-Year Sharpe Ratio", fontweight="bold")
axes[1, 0].legend(frameon=False)

# (d) Monthly return heatmap of the final strategy
m = net_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
heat = m.to_frame("ret")
heat["Year"], heat["Month"] = heat.index.year, heat.index.month
pivot = heat.pivot_table(index="Year", columns="Month", values="ret")
sns.heatmap(pivot, cmap="RdYlGn", center=0, annot=True, fmt=".1%",
            annot_kws={"size": 6.5}, cbar=False, ax=axes[1, 1])
axes[1, 1].set_title("Monthly Returns — Final Strategy", fontweight="bold")

plt.tight_layout()
plt.savefig("risk_diagnostics.png", dpi=300, bbox_inches="tight")
plt.show()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 13 — VISUALIZATION IV: INTERACTIVE PLOTLY DASHBOARD             ║
# ╚══════════════════════════════════════════════════════════════════════╝

fig = make_subplots(
    rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.08,
    subplot_titles=("Cumulative Returns", "Factor Signal (last value)",
                    "Drawdown — Final Strategy", "Sleeve Contribution"),
)

for col in variants.columns:
    fig.add_trace(go.Scatter(
        x=variants.index, y=(1 + variants[col]).cumprod(),
        name=col, mode="lines"), row=1, col=1)

fig.add_trace(go.Bar(
    x=list(CFG.traded_factors),
    y=signals.iloc[-1].values,
    marker_color=["#2e86de" if v >= 0 else "#c0392b"
                  for v in signals.iloc[-1]],
    showlegend=False), row=1, col=2)

dd = (1 + net_ret).cumprod()
fig.add_trace(go.Scatter(
    x=dd.index, y=dd / dd.cummax() - 1, fill="tozeroy",
    line=dict(color="#c0392b"), showlegend=False), row=2, col=1)

contrib = (signals * sleeve_df).cumsum()
for f in CFG.traded_factors:
    fig.add_trace(go.Scatter(
        x=contrib.index, y=contrib[f], name=f"{f} contrib",
        stackgroup="one"), row=2, col=2)

fig.update_layout(
    height=760, template="plotly_white",
    title=dict(text="<b>Trend-Enhanced Factor Portfolio — Research "
                    "Dashboard</b>", font=dict(size=18)),
    legend=dict(orientation="h", y=-0.08),
)
fig.show()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 14 — QUANTSTATS INSTITUTIONAL TEARSHEET                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

import quantstats as qs
qs.extend_pandas()

# Full HTML tearsheet vs. SPY — downloads as a standalone report file.
qs.reports.html(
    net_ret, benchmark=spy_ret,
    title="Trend-Enhanced FF5 Factor Portfolio",
    output="tearsheet.html",
)
log.info("Tearsheet saved → tearsheet.html (download from Colab sidebar)")

# Quick inline snapshot as well:
qs.reports.metrics(net_ret, benchmark=spy_ret, mode="basic")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 15 — ROBUSTNESS CHECK: PARAMETER SENSITIVITY                    ║
# ║  A strategy that only works for ONE parameter set is overfit.         ║
# ╚══════════════════════════════════════════════════════════════════════╝

def sharpe_for_params(fast: int, slow: int) -> float:
    """Recompute the trend overlay Sharpe for a (fast, slow) EWMA pair."""
    if fast >= slow:
        return np.nan
    idx  = (1 + sleeve_df).cumprod()
    raw  = idx.ewm(span=fast, adjust=False).mean() \
         - idx.ewm(span=slow, adjust=False).mean()
    nrm  = raw / idx.diff().rolling(CFG.trend_vol_win).std()
    sig  = np.tanh(nrm).shift(1).fillna(0.0)
    r    = (sig * sleeve_df).mean(axis=1)
    return (r.mean() / r.std()) * ANNUALIZE if r.std() > 0 else np.nan

fast_grid = [10, 21, 42, 63]
slow_grid = [84, 126, 189, 252]
grid = pd.DataFrame(
    [[sharpe_for_params(f, s) for s in slow_grid] for f in fast_grid],
    index=[f"fast={f}" for f in fast_grid],
    columns=[f"slow={s}" for s in slow_grid],
)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.heatmap(grid, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax,
            cbar_kws={"label": "Sharpe"})
ax.set_title("Parameter Sensitivity — Trend Overlay Sharpe "
             "(should be broadly positive)", fontweight="bold")
plt.tight_layout()
plt.savefig("sensitivity.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n" + "═" * 70)
print(" PIPELINE COMPLETE — deliverables: equity_curves.png, "
      "signal_heatmap.png,")
print(" sleeves.png, risk_diagnostics.png, sensitivity.png, tearsheet.html")
print("═" * 70)
