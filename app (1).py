# ============================================================================
#  app.py — Streamlit front-end for the Trend-Enhanced FF5 Factor Portfolio
#
#  Run locally:      streamlit run app.py
#  Deploy:           push repo → share.streamlit.io (Streamlit Community Cloud)
#
#  Research / educational use only. Not investment advice.
# ============================================================================

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

ACCENT = "#0a3d62"
PALETTE = {
    "Static Factor (EW)": "#9aa0a6",
    "Trend-Enhanced": "#2e86de",
    "Trend + VolTarget (net)": ACCENT,
    "SPY": "#c8a24a",
}
FACTOR_NAMES = {
    "SMB": "SMB — Μικρές vs Μεγάλες εταιρείες",
    "HML": "HML — Φθηνές (value) vs Ακριβές (growth)",
    "RMW": "RMW — Κερδοφόρες vs Μη κερδοφόρες",
    "CMA": "CMA — Συντηρητικές vs Επιθετικές επενδύσεις",
}

st.title("Trend-Enhanced Fama-French Factor Portfolio")
st.caption(
    "Systematic long-short equity: FF5 factor sleeves → CTA trend overlay → "
    "volatility targeting. Ερευνητικό/εκπαιδευτικό εργαλείο — "
    "**δεν αποτελεί επενδυτική συμβουλή**."
)


def explain(title: str, body: str):
    """Plain-language explanation box, shown only in beginner mode."""
    if st.session_state.get("beginner_mode", True):
        with st.expander(f"💡 {title}"):
            st.markdown(body)


# ────────────────────────────────────────────────────────────────────────
#  Sidebar — every tunable parameter, with plain-language tooltips
#  (hover το ❔ δίπλα σε κάθε κουμπί για να δεις τι κάνει)
# ────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Ρυθμίσεις στρατηγικής")

    st.toggle("📖 Λειτουργία αρχαρίου (εξηγήσεις)", value=True,
              key="beginner_mode",
              help="Δείχνει κουτάκια «💡» με απλές εξηγήσεις κάτω από κάθε "
                   "γράφημα και νούμερο. Κλείσ' το όταν τα μάθεις!")

    st.subheader("🗓 Περίοδος & μετοχές")
    start_date = st.date_input(
        "Ημερομηνία έναρξης", dt.date(2010, 1, 1),
        min_value=dt.date(1990, 1, 1),
        help="Από πότε ξεκινάει το ιστορικό τεστ. Όσο πιο παλιά, τόσο πιο "
             "αξιόπιστα τα συμπεράσματα — αλλά πιο αργό το τρέξιμο.")
    end_date = st.date_input(
        "Ημερομηνία λήξης", dt.date.today(),
        help="Μέχρι πότε τρέχει το τεστ. Συνήθως το αφήνεις στο σήμερα.")
    n_universe = st.slider(
        "Πόσες μετοχές να χρησιμοποιηθούν", 40, len(S.FULL_UNIVERSE),
        len(S.FULL_UNIVERSE), step=10,
        help="Η λίστα είναι ~110 μεγάλες αμερικανικές εταιρείες (Apple, "
             "Microsoft, JPMorgan, Exxon...) απ' όλους τους κλάδους — ΔΕΝ "
             "είναι τυχαίες, είναι γραμμένες στον κώδικα. Λιγότερες = πιο "
             "γρήγορο τρέξιμο για πειραματισμό.")

    st.subheader("🏗 Στρώση 1 · Καλάθια μετοχών")
    beta_window = st.select_slider(
        "Παράθυρο μέτρησης (μέρες)", [126, 189, 252, 378, 504], value=252,
        help="Πόσες μέρες ιστορικού κοιτάει για να μετρήσει το «προφίλ» κάθε "
             "μετοχής. 252 μέρες = 1 χρόνος συναλλαγών (η κλασική επιλογή). "
             "Μικρότερο = πιο «νευρικές» μετρήσεις.")
    n_deciles = st.slider(
        "Σε πόσα κομμάτια κόβεται η κατάταξη", 5, 10, 10,
        help="10 = αγοράζει το καλύτερο 10% των μετοχών και σορτάρει το "
             "χειρότερο 10%. 5 = παίζει το top/bottom 20% (περισσότερες "
             "μετοχές, πιο «ήπιο» στοίχημα).")
    traded_factors = st.multiselect(
        "Ποια κριτήρια (παράγοντες) να παίξει",
        ["SMB", "HML", "RMW", "CMA"],
        default=["SMB", "HML", "RMW", "CMA"],
        help="SMB: μικρές vs μεγάλες εταιρείες · HML: φθηνές vs ακριβές · "
             "RMW: κερδοφόρες vs μη · CMA: συντηρητικές vs επιθετικές. "
             "Βγάλε ένα για να δεις πόσο συνεισφέρει στο σύνολο.")

    st.subheader("📡 Στρώση 2 · Ραντάρ φόρμας")
    trend_fast = st.slider(
        "Γρήγορος μέσος όρος (μέρες)", 5, 63, 21,
        help="Πιάνει την ΠΡΟΣΦΑΤΗ φόρμα κάθε κριτηρίου (21 ≈ 1 μήνας). "
             "Μικρότερος = αντιδράει πιο γρήγορα, αλλά με περισσότερους "
             "ψεύτικους συναγερμούς.")
    trend_slow = st.slider(
        "Αργός μέσος όρος (μέρες)", 84, 252, 126,
        help="Η ΜΕΓΑΛΗ εικόνα (126 ≈ 6 μήνες). Το σήμα = γρήγορος − αργός: "
             "αν ο γρήγορος είναι από πάνω, το κριτήριο «ανεβαίνει» και "
             "παίρνει μεγαλύτερη θέση.")

    st.subheader("🎚 Στρώση 3 · Έλεγχος ρίσκου")
    vol_target_pct = st.slider(
        "Στόχος ρίσκου (% ετησίως)", 5, 20, 10,
        help="Το «cruise control»: πόσο έντονες διακυμάνσεις θες. 10% = "
             "ήπιες (για σύγκριση, ο S&P 500 έχει ~15-20%). Ανέβασέ το → "
             "μεγαλύτερα κέρδη ΚΑΙ μεγαλύτερες βουτιές.")
    max_leverage = st.slider(
        "Μέγιστο «γκάζι» (μόχλευση ×)", 1.0, 4.0, 2.0, 0.5,
        help="Δίχτυ ασφαλείας: σε πολύ ήσυχες αγορές το σύστημα θα ήθελε "
             "τεράστια μόχλευση για να πιάσει τον στόχο ρίσκου — αυτό το "
             "φρενάρει. 2× = ποτέ πάνω από διπλάσια έκθεση.")

    st.subheader("💸 Κόστη συναλλαγών")
    tc_bps = st.slider(
        "Κόστος ανά συναλλαγή (bps)", 0.0, 20.0, 5.0, 0.5,
        help="1 bp = 0,01%. Δηλαδή 5 bps = πληρώνεις 0,05% κάθε φορά που "
             "αγοράζεις ή πουλάς. Βάλ' το 0 για να δεις τον «παράδεισο "
             "χωρίς προμήθειες» και σύγκρινε.")

    run_btn = st.button("🚀 Τρέξε το backtest", type="primary",
                        width="stretch")

if not traded_factors:
    st.warning("Διάλεξε τουλάχιστον ένα κριτήριο (παράγοντα) στο sidebar.")
    st.stop()
if trend_fast >= trend_slow:
    st.warning("Ο γρήγορος μέσος όρος πρέπει να είναι ΜΙΚΡΟΤΕΡΟΣ από τον "
               "αργό (αλλιώς το «ραντάρ» δεν βγάζει νόημα).")
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
#  Cached data loaders
# ────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Κατεβάζω τους παράγοντες Fama-French…",
               ttl=24 * 3600)
def cached_ff5(start: str, end: str) -> pd.DataFrame:
    return S.load_ff5_factors(start, end)


@st.cache_data(show_spinner="Κατεβάζω τιμές μετοχών (Yahoo Finance)…",
               ttl=24 * 3600)
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
    st.info("👈 Ρύθμισε τις παραμέτρους στο sidebar (ή άφησέ τες ως έχουν) "
            "και πάτα **🚀 Τρέξε το backtest**. Το πρώτο τρέξιμο κατεβάζει "
            "~15 χρόνια δεδομένων και θέλει 1-3 λεπτά· μετά όλα είναι "
            "αποθηκευμένα και τα επόμενα τρεξίματα είναι σχεδόν ακαριαία.")
    explain(
        "Τι κάνει αυτή η εφαρμογή, σε 4 γραμμές",
        """
1. **Στρώση 1:** Χωρίζει ~110 μεγάλες αμερικανικές μετοχές σε «καλάθια» με
   βάση 4 κλασικά κριτήρια (μέγεθος, φθήνια, κερδοφορία, συντηρητικότητα)
   και **αγοράζει τις καλύτερες ενώ ταυτόχρονα σορτάρει τις χειρότερες** —
   έτσι ποντάρει στη ΔΙΑΦΟΡΑ τους, όχι στο αν θα ανέβει η αγορά.
2. **Στρώση 2:** Παρακολουθεί ποιο κριτήριο «έχει φόρμα» τελευταία και του
   δίνει μεγαλύτερη θέση· όποιο πάει άσχημα, το μειώνει ή το αντιστρέφει.
3. **Στρώση 3:** Ρυθμίζει αυτόματα το «γκάζι» ώστε το συνολικό ρίσκο να
   μένει σταθερό — σαν cruise control.
4. Σου δείχνει πώς θα είχε πάει όλο αυτό ιστορικά, με ρεαλιστικά κόστη.
        """)
    st.stop()

try:
    ff5 = cached_ff5(cfg.start_date, cfg.end_date)
    prices = cached_prices(cfg.universe, cfg.start_date, cfg.end_date)
    spy = cached_prices((cfg.benchmark,), cfg.start_date, cfg.end_date)
except Exception as exc:
    st.error(f"Η λήψη δεδομένων απέτυχε: {exc}\n\n"
             "Το Yahoo Finance / Ken French βάζει καμιά φορά προσωρινά "
             "όρια — περίμενε 1 λεπτό και ξαναπάτα «Τρέξε».")
    st.stop()

cfg_key = str(sorted(cfg.__dict__.items()))
with st.spinner("Τρέχω τις παλινδρομήσεις και το backtest…"):
    try:
        R = cached_backtest(cfg_key, cfg, ff5, prices, spy)
    except Exception as exc:
        st.error(f"Το backtest απέτυχε: {exc}")
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
c1.metric("CAGR", f"{cagr:.2%}",
          help="Μέση ετήσια απόδοση: πόσο μεγάλωνε το κεφάλαιο κάθε χρόνο "
               "κατά μέσο όρο.")
c2.metric("Sharpe", f"{sharpe:.2f}",
          help="Απόδοση ανά μονάδα ρίσκου — ΤΟ πιο σημαντικό νούμερο. "
               "Χοντρικά: <0,5 μέτριο · 0,5-1 καλό · >1 πολύ καλό για "
               "long-short στρατηγική.")
c3.metric("Χειρότερη βουτιά", f"{maxdd:.2%}",
          help="Max drawdown: η μεγαλύτερη πτώση από κορυφή σε πάτο. "
               "−20% σημαίνει ότι κάποια στιγμή θα έβλεπες το 1/5 του "
               "κεφαλαίου «χαμένο». Θα το άντεχες;")
c4.metric("Πραγματικό ρίσκο", f"{final.std() * S.ANNUALIZE:.2%}",
          help="Το ρίσκο (μεταβλητότητα) που τελικά βγήκε. Θες να είναι "
               "κοντά στον στόχο που έβαλες — σημάδι ότι το «cruise "
               "control» δουλεύει.")
c5.metric("Alpha (καθαρή αξία)", f"{R['alpha_ann']:.2%}",
          delta=f"t = {R['alpha_t']:.2f}", delta_color="off",
          help="Η απόδοση που ΔΕΝ εξηγείται από τους ίδιους τους "
               "παράγοντες — η «προστιθέμενη αξία» του timing. Το t από "
               "κάτω είναι τεστ αξιοπιστίας: αν |t| > 2, το alpha μάλλον "
               "δεν είναι τυχαίο.")

st.divider()

# ────────────────────────────────────────────────────────────────────────
#  Tabs
# ────────────────────────────────────────────────────────────────────────
(tab_perf, tab_hold, tab_signals, tab_risk,
 tab_stats, tab_sens, tab_export) = st.tabs(
    ["📈 Απόδοση", "🔍 Χαρτοφυλάκιο", "📡 Σήματα φόρμας",
     "🛡 Διαχείριση ρίσκου", "📊 Στατιστικά", "🔬 Έλεγχος αξιοπιστίας",
     "📥 Λήψεις"]
)

# ── TAB 1: Performance ─────────────────────────────────────────────────
with tab_perf:
    explain(
        "Πώς διαβάζεται αυτό το γράφημα",
        """
Δείχνει πώς θα μεγάλωνε **1 δολάριο** από την αρχή μέχρι σήμερα, για 4
εκδοχές:
- **Γκρι** = μόνο τα καλάθια, χωρίς timing (το σημείο αναφοράς).
- **Γαλάζιο** = καλάθια + ραντάρ φόρμας.
- **Σκούρο μπλε (χοντρή γραμμή)** = **η τελική στρατηγική** με όλα τα
  στάδια ΚΑΙ τα κόστη — αυτή μας ενδιαφέρει.
- **Χρυσό** = «απλά αγόρασε τον S&P 500» (SPY), για σύγκριση.

Η κλίμακα είναι λογαριθμική: **ίδια κλίση = ίδιος ρυθμός ανάπτυξης**,
ώστε να συγκρίνεις δίκαια παλιές και πρόσφατες χρονιές. Το κάτω γράφημα
(«βουτιές») δείχνει πόσο ΚΑΤΩ από το ιστορικό της υψηλό ήταν κάθε γραμμή
ανά πάσα στιγμή — ο «χάρτης του πόνου». Σημείωση: τα πρώτα ~1,5 χρόνια
είναι «νεκρά» επειδή το σύστημα χρειάζεται 1 χρόνο ιστορικού για τις
πρώτες μετρήσεις — είναι φυσιολογικό.
        """)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04,
                        subplot_titles=("Πόσο έγινε $1 (λογαριθμική κλίμακα)",
                                        "Βουτιές (drawdowns)"))
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

    fig2 = go.Figure()
    for f in cfg.traded_factors:
        fig2.add_trace(go.Scatter(
            x=sleeve_df.index, y=(1 + sleeve_df[f]).cumprod(),
            name=FACTOR_NAMES.get(f, f)))
    fig2.update_layout(
        title="Κάθε κριτήριο ξεχωριστά (πριν το timing)",
        height=400, template="plotly_white",
        legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig2, width="stretch")
    explain(
        "Τι δείχνει το δεύτερο γράφημα",
        "Η πορεία του κάθε κριτηρίου **μόνου του**, χωρίς το ραντάρ φόρμας. "
        "Έτσι βλέπεις ποιο κριτήριο «δούλεψε» ιστορικά και ποιο όχι — και "
        "καταλαβαίνεις γιατί χρειάζεται το timing: κανένα δεν πάει καλά "
        "συνέχεια.")

# ── TAB 2: Holdings (peek inside) ──────────────────────────────────────
with tab_hold:
    holdings = R["holdings"]
    any_date = next((h["as_of"] for h in holdings.values()
                     if h["as_of"] is not None), None)
    st.subheader("Τι κρατάει η στρατηγική αυτή τη στιγμή")
    if any_date is not None:
        st.caption(f"Θέσεις από την τελευταία μηνιαία αναδιάρθρωση "
                   f"({any_date.date()}). Το σύστημα τις ανανεώνει στο "
                   f"τέλος κάθε μήνα.")
    explain(
        "Πώς διαβάζεται αυτή η σελίδα",
        """
Για κάθε κριτήριο, το σύστημα κατέταξε όλες τις μετοχές και:
- 🟢 **ΑΓΟΡΑΣΕ (long)** τις κορυφαίες → κερδίζει αν ανέβουν
- 🔴 **ΣΟΡΤΑΡΕ (short)** τις τελευταίες → κερδίζει αν πέσουν

Το κέρδος βγαίνει από τη **διαφορά** τους: αν οι πράσινες πάνε καλύτερα
από τις κόκκινες, κερδίζεις — **ακόμα κι αν πέσουν όλες μαζί**. Έτσι το
στοίχημα δεν εξαρτάται από το αν θα ανέβει η αγορά συνολικά.

Προσοχή: το ίδιο όνομα μπορεί να είναι πράσινο σε ένα κριτήριο και
κόκκινο σε άλλο — είναι φυσιολογικό. Π.χ. μια εταιρεία μπορεί να είναι
πολύ κερδοφόρα (long στο RMW) αλλά «ακριβή» (short στο HML).
        """)

    for f in cfg.traded_factors:
        h = holdings[f]
        st.markdown(f"#### {FACTOR_NAMES.get(f, f)}")
        if not h["longs"] and not h["shorts"]:
            st.caption("Δεν υπάρχουν ενεργές θέσεις (ανεπαρκές ιστορικό).")
            continue
        sig_now = signals[f].iloc[-1]
        direction = ("🟢 Το ραντάρ φόρμας είναι ΥΠΕΡ αυτού του κριτηρίου"
                     if sig_now > 0.1 else
                     "🔴 Το ραντάρ φόρμας είναι ΚΑΤΑ (οι θέσεις "
                     "αντιστρέφονται στην πράξη)"
                     if sig_now < -0.1 else
                     "⚪ Το ραντάρ φόρμας είναι σχεδόν ουδέτερο "
                     "(μικρή θέση)")
        st.caption(f"{direction} — τρέχον σήμα: {sig_now:+.2f}")
        colL, colR = st.columns(2)
        with colL:
            st.markdown("🟢 **Long (αγορασμένες)**")
            st.dataframe(pd.DataFrame({"Μετοχή": h["longs"]}),
                         hide_index=True, width="stretch")
        with colR:
            st.markdown("🔴 **Short (σορταρισμένες)**")
            st.dataframe(pd.DataFrame({"Μετοχή": h["shorts"]}),
                         hide_index=True, width="stretch")

# ── TAB 3: Trend signals ───────────────────────────────────────────────
with tab_signals:
    explain(
        "Τι είναι το «σήμα φόρμας»",
        """
Για κάθε κριτήριο, το σύστημα συγκρίνει τον μέσο όρο του τελευταίου μήνα
με τον μέσο όρο του τελευταίου 6μήνου. Το αποτέλεσμα είναι ένας αριθμός
από **−1 έως +1**:
- **+1** = το κριτήριο «τρέχει» δυνατά → πλήρης θέση υπέρ του
- **0** = ουδέτερο → σχεδόν καμία θέση
- **−1** = πάει άσχημα → η θέση αντιστρέφεται

Σημαντικό: το σήμα υπολογίζεται με τα ΧΘΕΣΙΝΑ δεδομένα και εφαρμόζεται
σήμερα — δεν «κλέβει» κοιτώντας το μέλλον.
        """)

    monthly_sig = signals.resample("ME").last()
    fig = go.Figure(go.Heatmap(
        z=monthly_sig.T.values,
        x=monthly_sig.index, y=list(monthly_sig.columns),
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="Σήμα")))
    fig.update_layout(
        title="Το σήμα φόρμας ανά κριτήριο, μήνα-μήνα "
              "(μπλε = υπέρ · κόκκινο = κατά)",
        height=320, template="plotly_white")
    st.plotly_chart(fig, width="stretch")

    last = signals.iloc[-1]
    fig = go.Figure(go.Bar(
        x=list(last.index), y=last.values,
        marker_color=["#2e86de" if v >= 0 else "#c0392b" for v in last]))
    fig.update_layout(
        title=f"Τρέχουσα τοποθέτηση ({signals.index[-1].date()}) — "
              "μπάρα πάνω = υπέρ, κάτω = κατά",
        yaxis=dict(range=[-1.05, 1.05]), height=340,
        template="plotly_white")
    st.plotly_chart(fig, width="stretch")

    contrib = (signals * sleeve_df).cumsum()
    fig = go.Figure()
    for f in cfg.traded_factors:
        fig.add_trace(go.Scatter(x=contrib.index, y=contrib[f],
                                 name=FACTOR_NAMES.get(f, f),
                                 stackgroup="one"))
    fig.update_layout(
        title="Πόσο κέρδος/ζημιά έφερε κάθε κριτήριο συνολικά",
        height=400, template="plotly_white",
        legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width="stretch")

# ── TAB 4: Risk diagnostics ────────────────────────────────────────────
with tab_risk:
    explain(
        "Τι ελέγχουμε σε αυτή τη σελίδα",
        """
Εδώ βλέπεις αν το «cruise control» του ρίσκου δουλεύει:
- **Πάνω αριστερά:** το «γκάζι» (μόχλευση) μέρα-μέρα. Ανεβαίνει όταν η
  αγορά είναι ήσυχη, πέφτει όταν έχει αναταράξεις. Η κόκκινη
  διακεκομμένη = το όριο που έβαλες.
- **Πάνω δεξιά:** το πραγματικό ρίσκο (μπλε) πρέπει να «χορεύει» γύρω
  από τον στόχο (κόκκινη γραμμή). Αν ξεφεύγει μόνιμα, κάτι δεν πάει καλά.
- **Κάτω αριστερά:** η «φόρμα» της στρατηγικής σε κυλιόμενα 12μηνα.
  Καμία στρατηγική δεν είναι πάντα πάνω από το μηδέν — εδώ βλέπεις
  τίμια τις κακές περιόδους.
- **Κάτω δεξιά:** κάθε κελί = ένας μήνας. Πράσινο = κερδοφόρος,
  κόκκινο = ζημιογόνος.
        """)

    colA, colB = st.columns(2)

    with colA:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=leverage.index, y=leverage,
                                 line=dict(color=ACCENT, width=1),
                                 name="Μόχλευση"))
        fig.add_hline(y=cfg.max_leverage, line_dash="dash",
                      line_color="crimson",
                      annotation_text=f"όριο {cfg.max_leverage}×")
        fig.update_layout(title="Το «γκάζι» (μόχλευση) στον χρόνο",
                          height=340, template="plotly_white",
                          showlegend=False)
        st.plotly_chart(fig, width="stretch")

        r = variants["Trend + VolTarget (net)"]
        rs_final = (r.rolling(252).mean() / r.rolling(252).std()) * S.ANNUALIZE
        r0 = variants["Static Factor (EW)"]
        rs_static = (r0.rolling(252).mean() / r0.rolling(252).std()) * S.ANNUALIZE
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rs_static.index, y=rs_static,
                                 name="Χωρίς timing",
                                 line=dict(color="#9aa0a6")))
        fig.add_trace(go.Scatter(x=rs_final.index, y=rs_final,
                                 name="Τελική στρατηγική",
                                 line=dict(color=ACCENT)))
        fig.add_hline(y=0, line_color="black", line_width=0.7)
        fig.update_layout(title="Sharpe κυλιόμενου 12μήνου (η «φόρμα»)",
                          height=340, template="plotly_white",
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch")

    with colB:
        roll_vol = net_ret.rolling(63).std() * S.ANNUALIZE
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol,
                                 name="Πραγματικό ρίσκο 3μήνου",
                                 line=dict(color="#2e86de", width=1)))
        fig.add_hline(y=cfg.vol_target, line_dash="dash",
                      line_color="crimson",
                      annotation_text=f"στόχος {cfg.vol_target:.0%}")
        fig.update_layout(title="Πραγματικό ρίσκο vs στόχος",
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
            showscale=False))
        fig.update_layout(title="Μηνιαίες αποδόσεις (%) — τελική στρατηγική",
                          height=340, template="plotly_white",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")

    avg_turnover = R["turnover"].mean() * S.TRADING_DAYS
    st.caption(f"Μέση ετήσια «κίνηση» χαρτοφυλακίου ≈ {avg_turnover:,.1f}× "
               f"(πόσες φορές τον χρόνο ανακυκλώνεται το κεφάλαιο) · "
               f"κόστος: {cfg.tc_bps} bps ανά συναλλαγή")

# ── TAB 5: Statistics ──────────────────────────────────────────────────
with tab_stats:
    st.subheader("Πλήρης πίνακας στατιστικών")
    explain(
        "Τι σημαίνει κάθε γραμμή του πίνακα",
        """
- **CAGR** — μέση ετήσια απόδοση.
- **Ann. Vol** — ετήσιο ρίσκο (πόσο «κουνιέται» η στρατηγική).
- **Sharpe** — απόδοση ÷ ρίσκο. Το βασικό μέτρο σύγκρισης.
- **Sortino** — σαν το Sharpe, αλλά «τιμωρεί» μόνο τις κακές μέρες
  (θεωρείται πιο δίκαιο).
- **Max DD** — η χειρότερη βουτιά από κορυφή σε πάτο.
- **Calmar** — απόδοση ÷ χειρότερη βουτιά: «πόσο πληρώνομαι για τον
  χειρότερο πόνο».
- **Hit rate** — ποσοστό κερδοφόρων ημερών. Μην τρομάξεις αν είναι κάτω
  από 50%: οι trend στρατηγικές κερδίζουν λίγες φορές ΠΟΛΛΑ, όχι πολλές
  φορές λίγα.
- **Skew** — θετικό = σπάνιες μεγάλες ευχάριστες εκπλήξεις (καλό)·
  αρνητικό = σπάνιες μεγάλες δυσάρεστες.
- **Kurtosis** — πόσο συχνά συμβαίνουν «ακραία» γεγονότα. Ψηλό νούμερο =
  πιο συχνές ακρότητες απ' ό,τι στη «φυσιολογική» τύχη.
        """)
    st.dataframe(R["stats_table"], width="stretch")
    st.markdown(
        f"**Alpha έναντι των παραγόντων (Newey-West):** {R['alpha_ann']:.2%} "
        f"ετησίως, t = {R['alpha_t']:.2f}."
    )
    explain(
        "Το alpha και το «t» με απλά λόγια",
        "Το alpha είναι η απόδοση που **δεν** εξηγείται από τα ίδια τα "
        "κριτήρια — δηλαδή η προστιθέμενη αξία του timing και της "
        "διαχείρισης ρίσκου. Το t είναι τεστ αξιοπιστίας: μετράει πόσο "
        "απίθανο είναι το alpha να προέκυψε από καθαρή τύχη. **Χοντρικός "
        "κανόνας: |t| > 2 = μάλλον πραγματικό, |t| < 1 = μάλλον θόρυβος.**")
    st.subheader("Ποιες μετοχές μπήκαν τελικά στο τεστ")
    st.caption(f"{R['excess_ret'].shape[1]} μετοχές πέρασαν το φίλτρο "
               "(απαιτείται 90%+ διαθεσιμότητα δεδομένων στην περίοδο):")
    st.code(", ".join(R["excess_ret"].columns), language=None)

# ── TAB 6: Sensitivity ─────────────────────────────────────────────────
with tab_sens:
    explain(
        "Το τεστ «μήπως κοροϊδεύω τον εαυτό μου;»",
        """
Μεγάλη παγίδα στα backtests: δοκιμάζεις 100 συνδυασμούς ρυθμίσεων,
κρατάς αυτόν που έτυχε να δουλέψει στο παρελθόν, και νομίζεις ότι βρήκες
χρυσωρυχείο. Λέγεται **overfitting**.

Το τεστ εδώ: ξανατρέχουμε το ραντάρ φόρμας με **16 διαφορετικούς
συνδυασμούς** (γρήγορου, αργού) μέσου όρου και δείχνουμε το Sharpe του
καθενός.
- Αν **σχεδόν όλο το πλέγμα είναι θετικό** → το φαινόμενο είναι
  πραγματικό, δεν κρέμεται από μία τυχερή ρύθμιση. ✅
- Αν δουλεύει **μόνο ένα-δύο κελιά** → μάλλον τύχη, μην το
  εμπιστεύεσαι. ⚠️
        """)
    grid = S.sensitivity_grid(sleeve_df, cfg)
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=list(grid.columns), y=list(grid.index),
        colorscale="YlGnBu",
        text=np.round(grid.values, 2), texttemplate="%{text}",
        colorbar=dict(title="Sharpe")))
    fig.update_layout(
        title="Sharpe για κάθε συνδυασμό ρυθμίσεων του ραντάρ",
        height=380, template="plotly_white")
    st.plotly_chart(fig, width="stretch")

# ── TAB 7: Export ──────────────────────────────────────────────────────
with tab_export:
    st.subheader("Κατέβασε τα αποτελέσματα")
    explain(
        "Τι περιέχει το κάθε αρχείο",
        "Τα **CSV** ανοίγουν σε Excel: το πρώτο έχει τις ημερήσιες "
        "αποδόσεις όλων των εκδοχών (για δικές σου αναλύσεις), το δεύτερο "
        "τα σήματα φόρμας μέρα-μέρα. Το **tearsheet** είναι μια πλήρης "
        "«έκθεση θεσμικού τύπου» σε μία HTML σελίδα — δεκάδες γραφήματα "
        "και στατιστικά σε σύγκριση με τον S&P 500, ιδανικό για "
        "portfolio/βιογραφικό.")

    st.download_button("⬇️ Ημερήσιες αποδόσεις όλων των εκδοχών (CSV)",
                       variants.to_csv().encode(),
                       "strategy_variants.csv", "text/csv")
    st.download_button("⬇️ Σήματα φόρμας (CSV)",
                       signals.to_csv().encode(),
                       "trend_signals.csv", "text/csv")

    st.divider()
    st.subheader("Πλήρης αναφορά QuantStats (HTML)")
    st.caption("Δημιουργείται όταν τη ζητήσεις — θέλει ~20 δευτερόλεπτα.")
    if st.button("Δημιούργησε την αναφορά"):
        try:
            import tempfile, os
            import quantstats as qs
            qs.extend_pandas()
            with st.spinner("Φτιάχνω την αναφορά…"):
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
                f"Η αναφορά QuantStats απέτυχε ({exc}). Το QuantStats "
                "καμιά φορά δεν προλαβαίνει τις νέες εκδόσεις pandas — "
                "τα CSV παραπάνω περιέχουν ό,τι χρειάζεσαι.")

st.divider()
st.caption("Δεδομένα: Yahoo Finance (τιμές) · Ken French Data Library "
           "(παράγοντες FF5). Η σταθερή σημερινή λίστα μετοχών έχει "
           "survivorship bias (γνωστός περιορισμός της δωρεάν έρευνας). "
           "Τα ιστορικά backtests δεν εγγυώνται μελλοντικές αποδόσεις. "
           "Δεν αποτελεί επενδυτική συμβουλή.")
