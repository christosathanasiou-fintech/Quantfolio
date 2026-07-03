# ============================================================================
#  app.py — Streamlit front-end for the Trend-Enhanced FF5 Factor Portfolio
#
#  Run locally:      streamlit run app.py
#  Deploy:           push repo → share.streamlit.io (Streamlit Community Cloud)
#
#  Research / educational use only. Not investment advice.
# ============================================================================

import io
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import strategy as S

# ────────────────────────────────────────────────────────────────────────
#  Page setup
# ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trend-Enhanced Factor Portfolio",
    page_icon="📈",
    layout="wide",
)

ACCENT = "#0a3d62"      # deep desk-blue — the "final strategy" color everywhere
PALETTE = {
    "Static Factor (EW)": "#9aa0a6",
    "Trend-Enhanced": "#2e86de",
    "Trend + VolTarget (net)": ACCENT,
    "SPY": "#c8a24a",
}

st.title("Trend-Enhanced Fama-French Factor Portfolio")
st.caption(
    "Systematic long-short equity: FF5 cross-sectional factor sleeves → "
    "CTA-style factor-momentum overlay → ex-ante volatility targeting. "
    "Research/educational tool — **not investment advice**."
)

# ────────────────────────────────────────────────────────────────────────
#  Sidebar — every tunable parameter of the strategy
# ────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Strategy parameters")

    st.subheader("Sample")
    start_date = st.date_input("Start date", dt.date(2010, 1, 1),
                               min_value=dt.date(1990, 1, 1))
    end_date = st.date_input("End date", dt.date.today())
    n_universe = st.slider(
        "Universe size (tickers)", 40, len(S.FULL_UNIVERSE),
        len(S.FULL_UNIVERSE), step=10,
        help="Fewer tickers → faster run. Full list ≈ 110 large caps.")

    st.subheader("Layer 1 · Factor engine")
    beta_window = st.select_slider("Rolling OLS window (days)",
                                   [126, 189, 252, 378, 504], value=252)
    n_deciles = st.slider("Number of ranking buckets", 5, 10, 10)
    traded_factors = st.multiselect(
        "Traded factor sleeves",
        ["SMB", "HML", "RMW", "CMA"],
        default=["SMB", "HML", "RMW", "CMA"])

    st.subheader("Layer 2 · Trend overlay")
    trend_fast = st.slider("Fast EWMA span", 5, 63, 21)
    trend_slow = st.slider("Slow EWMA span", 84, 252, 126)

    st.subheader("Layer 3 · Vol targeting")
    vol_target_pct = st.slider("Volatility target (%)", 5, 20, 10)
    max_leverage = st.slider("Leverage cap (×)", 1.0, 4.0, 2.0, 0.5)

    st.subheader("Frictions")
    tc_bps = st.slider("One-way transaction cost (bps)", 0.0, 20.0, 5.0, 0.5)

    run_btn = st.button("🚀 Run backtest", type="primary",
                        width="stretch")

if not traded_factors:
    st.warning("Select at least one factor sleeve in the sidebar.")
    st.stop()
if trend_fast >= trend_slow:
    st.warning("Fast EWMA span must be smaller than the slow span.")
    st.stop()

cfg = S.Config(
    start_date=str(start_date),
    end_date=str(end_date),
    universe=S.FULL_UNIVERSE[:n_universe],
    beta_window=beta_window,
    n_deciles=n_deciles,
    traded_factors=tuple(traded_factors),
    trend_fast=trend_fast,
    trend_slow=trend_slow,
    vol_target=vol_target_pct / 100.0,
    max_leverage=max_leverage,
    tc_bps=tc_bps,
)

# ────────────────────────────────────────────────────────────────────────
#  Cached data loaders (data survives across parameter tweaks)
# ────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Downloading Fama-French 5 factors…", ttl=24 * 3600)
def cached_ff5(start: str, end: str) -> pd.DataFrame:
    return S.load_ff5_factors(start, end)


@st.cache_data(show_spinner="Downloading prices (yfinance)…", ttl=24 * 3600)
def cached_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    return S.load_prices(list(tickers), start, end)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def cached_backtest(cfg_key: str, _cfg: S.Config, _ff5, _prices, _spy):
    return S.run_backtest(_cfg, _ff5, _prices, _spy)


# ────────────────────────────────────────────────────────────────────────
#  Run
# ────────────────────────────────────────────────────────────────────────
if run_btn:
    st.session_state["run_requested"] = True

if not st.session_state.get("run_requested"):
    st.info("Set your parameters in the sidebar and press **Run backtest**. "
            "First run downloads ~15 years of data and takes 1–3 minutes; "
            "afterwards everything is cached.")
    st.stop()

try:
    ff5 = cached_ff5(cfg.start_date, cfg.end_date)
    prices = cached_prices(cfg.universe, cfg.start_date, cfg.end_date)
    spy = cached_prices((cfg.benchmark,), cfg.start_date, cfg.end_date)
except Exception as exc:
    st.error(f"Data download failed: {exc}\n\n"
             "Yahoo Finance / Ken French can rate-limit — wait a minute "
             "and press Run again.")
    st.stop()

cfg_key = str(sorted(cfg.__dict__.items()))
with st.spinner("Running rolling regressions & backtest…"):
    try:
        R = cached_backtest(cfg_key, cfg, ff5, prices, spy)
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        st.stop()

variants, net_ret = R["variants"], R["net_ret"]
signals, sleeve_df = R["signals"], R["sleeve_df"]
leverage = R["leverage"]

# ────────────────────────────────────────────────────────────────────────
#  Headline metrics
# ────────────────────────────────────────────────────────────────────────
final = variants["Trend + VolTarget (net)"]
cum = (1 + final).cumprod()
sharpe = final.mean() / final.std() * S.ANNUALIZE
maxdd = (cum / cum.cummax() - 1).min()
cagr = cum.iloc[-1] ** (S.TRADING_DAYS / len(final)) - 1

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("CAGR", f"{cagr:.2%}")
c2.metric("Sharpe", f"{sharpe:.2f}")
c3.metric("Max drawdown", f"{maxdd:.2%}")
c4.metric("Ann. vol (realized)", f"{final.std() * S.ANNUALIZE:.2%}")
c5.metric("FF5 alpha (NW)", f"{R['alpha_ann']:.2%}",
          delta=f"t = {R['alpha_t']:.2f}", delta_color="off")

st.divider()

# ────────────────────────────────────────────────────────────────────────
#  Tabs
# ────────────────────────────────────────────────────────────────────────
tab_perf, tab_signals, tab_risk, tab_stats, tab_sens, tab_export = st.tabs(
    ["📈 Performance", "🎛 Trend signals", "🛡 Risk diagnostics",
     "📊 Statistics", "🔬 Sensitivity", "📥 Export"]
)

# ── TAB 1: Performance ─────────────────────────────────────────────────
with tab_perf:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04,
                        subplot_titles=("Cumulative growth of $1 (log scale)",
                                        "Drawdowns"))
    for col in variants.columns:
        curve = (1 + variants[col]).cumprod()
        lw = 2.6 if "VolTarget" in col else 1.4
        fig.add_trace(go.Scatter(x=curve.index, y=curve, name=col,
                                 line=dict(color=PALETTE[col], width=lw)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=curve.index, y=curve / curve.cummax() - 1,
                                 name=col, showlegend=False,
                                 line=dict(color=PALETTE[col], width=1)),
                      row=2, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_layout(height=640, template="plotly_white",
                      legend=dict(orientation="h", y=1.06),
                      margin=dict(t=60, b=10))
    st.plotly_chart(fig, width="stretch")

    # Sleeve equity curves
    fig2 = go.Figure()
    for f in cfg.traded_factors:
        fig2.add_trace(go.Scatter(
            x=sleeve_df.index, y=(1 + sleeve_df[f]).cumprod(),
            name=f"{f} L/S sleeve"))
    fig2.update_layout(title="Factor sleeve equity curves (before timing)",
                       height=380, template="plotly_white",
                       legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig2, width="stretch")

# ── TAB 2: Trend signals ───────────────────────────────────────────────
with tab_signals:
    monthly_sig = signals.resample("ME").last()
    fig = go.Figure(go.Heatmap(
        z=monthly_sig.T.values,
        x=monthly_sig.index, y=list(monthly_sig.columns),
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="Signal")))
    fig.update_layout(title="CTA trend signal per factor sleeve (monthly)",
                      height=320, template="plotly_white")
    st.plotly_chart(fig, width="stretch")

    last = signals.iloc[-1]
    fig = go.Figure(go.Bar(
        x=list(last.index), y=last.values,
        marker_color=["#2e86de" if v >= 0 else "#c0392b" for v in last]))
    fig.update_layout(
        title=f"Current positioning (as of {signals.index[-1].date()})",
        yaxis=dict(range=[-1.05, 1.05]), height=340,
        template="plotly_white")
    st.plotly_chart(fig, width="stretch")

    contrib = (signals * sleeve_df).cumsum()
    fig = go.Figure()
    for f in cfg.traded_factors:
        fig.add_trace(go.Scatter(x=contrib.index, y=contrib[f],
                                 name=f"{f} contribution",
                                 stackgroup="one"))
    fig.update_layout(title="Cumulative sleeve contribution (timed)",
                      height=380, template="plotly_white",
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")

# ── TAB 3: Risk diagnostics ────────────────────────────────────────────
with tab_risk:
    colA, colB = st.columns(2)

    with colA:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=leverage.index, y=leverage,
                                 line=dict(color=ACCENT, width=1),
                                 name="Leverage"))
        fig.add_hline(y=cfg.max_leverage, line_dash="dash",
                      line_color="crimson",
                      annotation_text=f"cap {cfg.max_leverage}×")
        fig.update_layout(title="Vol-target leverage", height=340,
                          template="plotly_white", showlegend=False)
        st.plotly_chart(fig, width="stretch")

        r = variants["Trend + VolTarget (net)"]
        rs_final = (r.rolling(252).mean() / r.rolling(252).std()) * S.ANNUALIZE
        r0 = variants["Static Factor (EW)"]
        rs_static = (r0.rolling(252).mean() / r0.rolling(252).std()) * S.ANNUALIZE
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rs_static.index, y=rs_static,
                                 name="Static factor",
                                 line=dict(color="#9aa0a6")))
        fig.add_trace(go.Scatter(x=rs_final.index, y=rs_final,
                                 name="Final strategy",
                                 line=dict(color=ACCENT)))
        fig.add_hline(y=0, line_color="black", line_width=0.7)
        fig.update_layout(title="Rolling 1-year Sharpe", height=340,
                          template="plotly_white",
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch")

    with colB:
        roll_vol = net_ret.rolling(63).std() * S.ANNUALIZE
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol,
                                 name="Realized 3M vol",
                                 line=dict(color="#2e86de", width=1)))
        fig.add_hline(y=cfg.vol_target, line_dash="dash",
                      line_color="crimson",
                      annotation_text=f"target {cfg.vol_target:.0%}")
        fig.update_layout(title="Realized vs. target volatility",
                          yaxis_tickformat=".0%", height=340,
                          template="plotly_white", showlegend=False)
        st.plotly_chart(fig, width="stretch")

        m = net_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        heat = m.to_frame("ret")
        heat["Year"], heat["Month"] = heat.index.year, heat.index.month
        pivot = heat.pivot_table(index="Year", columns="Month", values="ret")
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            colorscale="RdYlGn", zmid=0,
            text=np.round(pivot.values * 100, 1),
            texttemplate="%{text}", textfont=dict(size=9),
            colorbar=dict(title="%", tickformat=".1%"), showscale=False))
        fig.update_layout(title="Monthly returns — final strategy (%)",
                          height=340, template="plotly_white",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")

    avg_turnover = R["turnover"].mean() * S.TRADING_DAYS
    st.caption(f"Average annual turnover ≈ {avg_turnover:,.1f}× · "
               f"cost model: {cfg.tc_bps} bps one-way on turnover")

# ── TAB 4: Statistics ──────────────────────────────────────────────────
with tab_stats:
    st.subheader("Performance summary")
    st.dataframe(R["stats_table"], width="stretch")
    st.markdown(
        f"**FF5 Newey-West alpha:** {R['alpha_ann']:.2%} p.a. "
        f"(t-stat = {R['alpha_t']:.2f}, HAC lags = 21). "
        "A |t| above ~2 suggests the alpha is statistically distinguishable "
        "from zero after accounting for autocorrelation."
    )
    st.subheader("Universe actually used")
    st.caption(f"{R['excess_ret'].shape[1]} tickers passed the 90% "
               "coverage filter:")
    st.code(", ".join(R["excess_ret"].columns), language=None)

# ── TAB 5: Sensitivity ─────────────────────────────────────────────────
with tab_sens:
    st.caption("A strategy that only works for one parameter set is "
               "overfit. The trend-overlay Sharpe should be broadly "
               "positive across the (fast, slow) EWMA grid.")
    grid = S.sensitivity_grid(sleeve_df, cfg)
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=list(grid.columns), y=list(grid.index),
        colorscale="YlGnBu",
        text=np.round(grid.values, 2), texttemplate="%{text}",
        colorbar=dict(title="Sharpe")))
    fig.update_layout(title="Trend-overlay Sharpe across EWMA pairs",
                      height=380, template="plotly_white")
    st.plotly_chart(fig, width="stretch")

# ── TAB 6: Export ──────────────────────────────────────────────────────
with tab_export:
    st.subheader("Download results")

    csv_variants = variants.to_csv().encode()
    st.download_button("⬇️ Daily returns of all variants (CSV)",
                       csv_variants, "strategy_variants.csv", "text/csv")

    csv_signals = signals.to_csv().encode()
    st.download_button("⬇️ Trend signals (CSV)",
                       csv_signals, "trend_signals.csv", "text/csv")

    st.divider()
    st.subheader("QuantStats tearsheet (HTML)")
    st.caption("Full institutional tearsheet vs. SPY. Generated on demand — "
               "takes ~20 seconds.")
    if st.button("Generate tearsheet"):
        try:
            import tempfile, os
            import quantstats as qs
            qs.extend_pandas()
            with st.spinner("Building tearsheet…"):
                with tempfile.NamedTemporaryFile(
                        suffix=".html", delete=False) as tmp:
                    qs.reports.html(net_ret, benchmark=R["spy_ret"],
                                    title="Trend-Enhanced FF5 Factor Portfolio",
                                    output=tmp.name)
                    tmp_path = tmp.name
            with open(tmp_path, "rb") as fh:
                html_bytes = fh.read()
            os.unlink(tmp_path)
            st.download_button("⬇️ tearsheet.html", html_bytes,
                               "tearsheet.html", "text/html")
        except Exception as exc:
            st.warning(
                f"QuantStats tearsheet failed ({exc}). QuantStats sometimes "
                "lags behind new pandas releases — the CSV exports above "
                "contain everything needed to rebuild it offline.")

st.divider()
st.caption("Data: Yahoo Finance (prices) · Ken French Data Library (FF5). "
           "Fixed present-day universe carries survivorship bias — "
           "acknowledged limitation of free-data research. "
           "Historical backtests do not guarantee future performance.")
