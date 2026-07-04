# ============================================================================
#  strategy.py — Core engine: Trend-Enhanced Fama-French Factor Portfolio
#
#  Refactored from the original Colab script into pure, importable functions
#  so the Streamlit app (app.py) can call and cache each stage independently.
#
#  Key fixes vs. the Colab version:
#    * No top-level execution / no Colab-only `display()` calls
#    * FF5 loader: pandas_datareader with a direct-ZIP fallback from the
#      Ken French site (datareader breaks on some pandas versions)
#    * yfinance loader handles both old/new column layouts & single tickers
#    * Guard against zero std in cross-sectional z-scores
#    * All matplotlib removed — charts are built in app.py with Plotly
#  License: MIT — research / educational use only. Not investment advice.
# ============================================================================

from __future__ import annotations

import io
import time
import logging
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

log = logging.getLogger("factor-trend")

TRADING_DAYS = 252
ANNUALIZE = np.sqrt(TRADING_DAYS)

FULL_UNIVERSE: tuple = (
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC", "CSCO",
    "TXN", "QCOM", "IBM", "NOW", "INTU", "AMAT", "MU", "ADI", "LRCX", "KLAC",
    # Communication / Media
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "GM", "F",
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT", "MDLZ", "CL", "KMB", "GIS", "KHC",
    # Health Care
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "CVS", "MDT", "ISRG",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
    "PNC", "TFC", "CB", "MMC", "AIG",
    # Industrials
    "CAT", "BA", "HON", "UNP", "UPS", "RTX", "LMT", "GE", "DE", "MMM",
    "FDX", "EMR", "ETN", "CSX", "NSC",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "OXY",
    # Materials / Real Estate / Utilities
    "LIN", "APD", "SHW", "FCX", "NEM", "AMT", "PLD", "SPG", "NEE", "DUK", "SO", "D",
)


@dataclass
class Config:
    start_date: str = "2010-01-01"
    end_date: str | None = None                  # None → today
    universe: tuple = FULL_UNIVERSE
    benchmark: str = "SPY"

    # Layer 1: factor engine
    beta_window: int = 252
    n_deciles: int = 10
    min_history: int = 300
    factors: tuple = ("Mkt-RF", "SMB", "HML", "RMW", "CMA")
    traded_factors: tuple = ("SMB", "HML", "RMW", "CMA")

    # Layer 2: CTA trend overlay
    trend_fast: int = 21
    trend_slow: int = 126
    trend_vol_win: int = 63
    signal_cap: float = 1.0

    # Layer 3: volatility targeting
    vol_target: float = 0.10
    vol_window: int = 63
    max_leverage: float = 2.0

    # Frictions
    tc_bps: float = 5.0

    trading_days: int = TRADING_DAYS


# ════════════════════════════════════════════════════════════════════════
#  DATA LAYER
# ════════════════════════════════════════════════════════════════════════

FF5_ZIP_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)


def _ff5_via_datareader(start: str, end: str | None) -> pd.DataFrame:
    import pandas_datareader.data as pdr
    raw = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_daily",
                         "famafrench", start=start, end=end)
    ff = raw[0].copy()
    ff.index = pd.to_datetime(ff.index)
    return ff


def _ff5_via_zip(start: str, end: str | None) -> pd.DataFrame:
    """Fallback: download the CSV zip straight from Ken French's site."""
    resp = requests.get(FF5_ZIP_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as fh:
            text = fh.read().decode("utf-8", errors="ignore")

    lines = text.splitlines()
    # Find the header row (starts with a comma: ",Mkt-RF,SMB,...")
    hdr = next(i for i, ln in enumerate(lines) if ln.strip().startswith(","))
    rows = []
    for ln in lines[hdr + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 7 or not parts[0].isdigit() or len(parts[0]) != 8:
            continue
        rows.append(parts[:7])
    ff = pd.DataFrame(rows, columns=["Date", "Mkt-RF", "SMB", "HML",
                                     "RMW", "CMA", "RF"])
    ff["Date"] = pd.to_datetime(ff["Date"], format="%Y%m%d")
    ff = ff.set_index("Date").astype(float)
    ff = ff.loc[ff.index >= pd.Timestamp(start)]
    if end:
        ff = ff.loc[ff.index <= pd.Timestamp(end)]
    return ff


def load_ff5_factors(start: str, end: str | None = None,
                     max_retries: int = 3) -> pd.DataFrame:
    """
    Daily Fama-French 5 factors + RF, in DECIMAL units.
    Tries pandas_datareader first, then the direct ZIP download.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        for loader in (_ff5_via_datareader, _ff5_via_zip):
            try:
                ff = loader(start, end)
                ff = ff / 100.0                       # percent → decimal
                log.info("FF5 loaded via %s: %d days",
                         loader.__name__, len(ff))
                return ff
            except Exception as exc:                  # noqa: BLE001
                last_exc = exc
                log.warning("FF5 loader %s failed (attempt %d): %s",
                            loader.__name__, attempt, exc)
        time.sleep(2 * attempt)
    raise ConnectionError(
        f"Could not download FF5 factors after retries: {last_exc}")


def load_prices(tickers, start: str, end: str | None = None,
                min_coverage: float = 0.90) -> pd.DataFrame:
    """
    Adjusted close prices via yfinance, with per-ticker fault tolerance.
    Tickers below `min_coverage` non-NaN availability are dropped.
    """
    import yfinance as yf

    tickers = list(dict.fromkeys(tickers))
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    if raw is None or len(raw) == 0:
        raise ConnectionError("yfinance returned no data — check tickers "
                              "and network access.")

    # Handle both MultiIndex ('Close', ticker) and flat layouts
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])

    coverage = close.notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    dropped = sorted(set(close.columns) - set(keep))
    if dropped:
        log.warning("Dropped %d low-coverage tickers: %s",
                    len(dropped), dropped)
    prices = close[keep].sort_index()
    if prices.empty:
        raise ValueError("No tickers passed the coverage filter.")
    return prices


# ════════════════════════════════════════════════════════════════════════
#  CLEANING / FEATURES
# ════════════════════════════════════════════════════════════════════════

def build_returns(prices: pd.DataFrame, ff: pd.DataFrame, cfg: Config,
                  winsor: float = 0.20):
    """Winsorized daily returns, aligned to the FF calendar, in excess of RF."""
    rets = prices.pct_change().iloc[1:]
    rets = rets.clip(lower=-winsor, upper=winsor)

    idx = rets.index.intersection(ff.index)
    if len(idx) < cfg.beta_window * 2:
        raise ValueError("Insufficient overlapping history between prices "
                         "and factor data — extend the sample period.")
    rets, ff_a = rets.loc[idx], ff.loc[idx]
    excess = rets.sub(ff_a["RF"], axis=0)
    return excess, ff_a


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index=index, data=0.0)
    return s.groupby(index.to_period("M")).tail(1).index


# ════════════════════════════════════════════════════════════════════════
#  LAYER 1 — FACTOR ENGINE
# ════════════════════════════════════════════════════════════════════════

def estimate_betas_at(date, excess: pd.DataFrame, ff: pd.DataFrame,
                      cfg: Config):
    """Vectorized rolling-window FF5 betas for all stocks at one date."""
    win_idx = excess.index[excess.index <= date][-cfg.beta_window:]
    if len(win_idx) < cfg.beta_window:
        return None

    Y = excess.loc[win_idx]
    X = ff.loc[win_idx, list(cfg.factors)].values
    X = np.column_stack([np.ones(len(X)), X])

    valid = Y.notna().mean() >= 0.95
    Y = Y.loc[:, valid].fillna(0.0)
    if Y.shape[1] < 30:
        return None

    coefs, *_ = np.linalg.lstsq(X, Y.values, rcond=None)
    return pd.DataFrame(coefs[1:].T, index=Y.columns,
                        columns=list(cfg.factors))


def build_factor_sleeves(excess: pd.DataFrame, ff: pd.DataFrame,
                         rebal_dates, cfg: Config,
                         progress_cb=None):
    """
    Monthly-rebalanced decile long-short portfolio per traded factor.
    Returns (sleeve_df, weights_panel dict).
    """
    sleeves, weights_panel = {}, {}
    daily_index = excess.index

    # Estimate betas once per rebalance date (shared across factors)
    betas_by_date = {}
    for k, rd in enumerate(rebal_dates):
        betas_by_date[rd] = estimate_betas_at(rd, excess, ff, cfg)
        if progress_cb:
            progress_cb((k + 1) / len(rebal_dates))

    for factor in cfg.traded_factors:
        w = pd.DataFrame(0.0, index=daily_index, columns=excess.columns)

        for i, rd in enumerate(rebal_dates):
            betas = betas_by_date[rd]
            if betas is None:
                continue
            std = betas[factor].std()
            if not np.isfinite(std) or std == 0:      # degenerate x-section
                continue
            z = (betas[factor] - betas[factor].mean()) / std
            n_bucket = max(len(z) // cfg.n_deciles, 3)

            longs = z.nlargest(n_bucket).index
            shorts = z.nsmallest(n_bucket).index

            current_w = pd.Series(0.0, index=excess.columns)
            current_w[longs] = 1.0 / n_bucket
            current_w[shorts] = -1.0 / n_bucket

            nxt = rebal_dates[i + 1] if i + 1 < len(rebal_dates) \
                else daily_index[-1]
            hold = daily_index[(daily_index > rd) & (daily_index <= nxt)]
            w.loc[hold] = current_w.values

        sleeve_ret = (w * excess).sum(axis=1)
        sleeves[factor], weights_panel[factor] = sleeve_ret, w

    sleeve_df = pd.DataFrame(sleeves).fillna(0.0)
    return sleeve_df, weights_panel


# ════════════════════════════════════════════════════════════════════════
#  LAYER 2 — CTA TREND OVERLAY (FACTOR MOMENTUM)
# ════════════════════════════════════════════════════════════════════════

def trend_signal(sleeve_returns: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    EWMA crossover on each sleeve's cumulative index, vol-normalized and
    tanh-squashed to (−1, +1). Shifted by 1 day → no look-ahead.
    """
    idx = (1.0 + sleeve_returns).cumprod()
    fast = idx.ewm(span=cfg.trend_fast, adjust=False).mean()
    slow = idx.ewm(span=cfg.trend_slow, adjust=False).mean()
    raw = fast - slow
    norm = raw / idx.diff().rolling(cfg.trend_vol_win).std().replace(0, np.nan)
    sig = np.tanh(norm).clip(-cfg.signal_cap, cfg.signal_cap)
    return sig.shift(1).fillna(0.0)


# ════════════════════════════════════════════════════════════════════════
#  LAYER 3 — VOLATILITY TARGETING + COSTS
# ════════════════════════════════════════════════════════════════════════

def vol_target(returns: pd.Series, cfg: Config):
    """Ex-ante vol scaling with a 1-day lag on the leverage decision."""
    ewma_vol = returns.ewm(span=cfg.vol_window, adjust=False).std() * ANNUALIZE
    lev = (cfg.vol_target / ewma_vol.replace(0, np.nan)).shift(1)
    lev = lev.clip(upper=cfg.max_leverage).fillna(0.0)
    return returns * lev, lev


def apply_costs(sleeve_weights: dict, signals: pd.DataFrame,
                leverage: pd.Series, gross_ret: pd.Series,
                cfg: Config):
    """
    Turnover-based cost model on effective stock-level weights.
    Returns (net_returns, turnover_series).
    """
    n = len(cfg.traded_factors)
    total_w = None
    for f in cfg.traded_factors:
        eff = sleeve_weights[f].mul(signals[f], axis=0) \
                               .mul(leverage, axis=0) / n
        total_w = eff if total_w is None else total_w + eff

    turnover = total_w.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * cfg.tc_bps / 10_000.0
    return gross_ret - cost, turnover


# ════════════════════════════════════════════════════════════════════════
#  ANALYTICS
# ════════════════════════════════════════════════════════════════════════

def perf_stats(r: pd.Series, name: str = "") -> pd.Series:
    """Institutional summary statistics for a daily return stream."""
    r = r.dropna()
    if len(r) < 2 or r.std() == 0:
        return pd.Series(dtype=float, name=name)

    cum = (1 + r).cumprod()
    yrs = len(r) / TRADING_DAYS
    cagr = cum.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * ANNUALIZE
    dd = cum / cum.cummax() - 1
    downside = r[r < 0].std() * ANNUALIZE

    return pd.Series({
        "CAGR": f"{cagr:.2%}",
        "Ann. Vol": f"{vol:.2%}",
        "Sharpe": f"{(r.mean() * TRADING_DAYS) / vol:.2f}",
        "Sortino": (f"{(r.mean() * TRADING_DAYS) / downside:.2f}"
                    if downside and downside > 0 else "n/a"),
        "Max DD": f"{dd.min():.2%}",
        "Calmar": (f"{cagr / abs(dd.min()):.2f}" if dd.min() < 0 else "n/a"),
        "Hit rate (d)": f"{(r > 0).mean():.2%}",
        "Skew": f"{r.skew():.2f}",
        "Kurtosis": f"{r.kurtosis():.2f}",
    }, name=name)


def nw_alpha(strategy: pd.Series, ff: pd.DataFrame, cfg: Config,
             lags: int = 21):
    """Annualized FF5 alpha with Newey-West (HAC) standard errors."""
    df = pd.concat([strategy, ff[list(cfg.factors)]], axis=1).dropna()
    X = sm.add_constant(df.iloc[:, 1:])
    res = sm.OLS(df.iloc[:, 0], X).fit(cov_type="HAC",
                                       cov_kwds={"maxlags": lags})
    return (res.params["const"] * TRADING_DAYS,
            res.tvalues["const"], res)


def sensitivity_grid(sleeve_df: pd.DataFrame, cfg: Config,
                     fast_grid=(10, 21, 42, 63),
                     slow_grid=(84, 126, 189, 252)) -> pd.DataFrame:
    """Trend-overlay Sharpe across (fast, slow) EWMA pairs."""
    idx = (1 + sleeve_df).cumprod()
    dvol = idx.diff().rolling(cfg.trend_vol_win).std()

    def sharpe(fast, slow):
        if fast >= slow:
            return np.nan
        raw = idx.ewm(span=fast, adjust=False).mean() \
            - idx.ewm(span=slow, adjust=False).mean()
        sig = np.tanh(raw / dvol).shift(1).fillna(0.0)
        r = (sig * sleeve_df).mean(axis=1)
        return (r.mean() / r.std()) * ANNUALIZE if r.std() > 0 else np.nan

    return pd.DataFrame(
        [[sharpe(f, s) for s in slow_grid] for f in fast_grid],
        index=[f"fast={f}" for f in fast_grid],
        columns=[f"slow={s}" for s in slow_grid],
    )


def current_holdings(sleeve_weights: dict, cfg: Config) -> dict:
    """
    For each traded factor, return the stocks currently held LONG and
    SHORT (from the last date with non-zero weights), plus the as-of date.
    """
    out = {}
    for f in cfg.traded_factors:
        w = sleeve_weights[f]
        nonzero = w.index[(w != 0).any(axis=1)]
        if len(nonzero) == 0:
            out[f] = {"longs": [], "shorts": [], "as_of": None}
            continue
        last = w.loc[nonzero[-1]]
        out[f] = {
            "longs": sorted(last[last > 0].index.tolist()),
            "shorts": sorted(last[last < 0].index.tolist()),
            "as_of": nonzero[-1],
        }
    return out


# ════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ════════════════════════════════════════════════════════════════════════

def run_backtest(cfg: Config, ff5: pd.DataFrame, prices: pd.DataFrame,
                 spy_price: pd.DataFrame, progress_cb=None) -> dict:
    """
    Runs Layers 1→3 end-to-end on pre-loaded data.
    Returns a dict with every intermediate the app needs to plot.
    """
    excess_ret, ff5a = build_returns(prices, ff5, cfg)
    bench_col = spy_price.columns[0]
    spy_ret = (spy_price[bench_col].pct_change()
               .reindex(excess_ret.index).fillna(0.0))

    rebal_dates = month_end_dates(excess_ret.index)

    sleeve_df, sleeve_weights = build_factor_sleeves(
        excess_ret, ff5a, rebal_dates, cfg, progress_cb=progress_cb)

    signals = trend_signal(sleeve_df, cfg)
    trend_ret = (signals * sleeve_df).mean(axis=1)
    static_ret = sleeve_df.mean(axis=1)

    scaled_ret, leverage = vol_target(trend_ret, cfg)
    net_ret, turnover = apply_costs(sleeve_weights, signals,
                                    leverage, scaled_ret, cfg)

    variants = pd.DataFrame({
        "Static Factor (EW)": static_ret,
        "Trend-Enhanced": trend_ret,
        "Trend + VolTarget (net)": net_ret,
        "SPY": spy_ret,
    }).dropna()

    alpha_ann, alpha_t, _ = nw_alpha(net_ret, ff5a, cfg)

    return dict(
        cfg=cfg, ff5=ff5a, excess_ret=excess_ret, spy_ret=spy_ret,
        holdings=current_holdings(sleeve_weights, cfg),
        sleeve_df=sleeve_df, signals=signals, leverage=leverage,
        turnover=turnover, net_ret=net_ret, variants=variants,
        alpha_ann=alpha_ann, alpha_t=alpha_t,
        stats_table=pd.concat(
            [perf_stats(variants[c], c) for c in variants.columns], axis=1),
    )
