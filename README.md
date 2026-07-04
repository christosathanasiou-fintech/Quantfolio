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

# 📖 Οδηγός Χρήσης — Trend-Enhanced Factor Portfolio App
### Τι κάνει το κάθε κουμπί, με απλά λόγια

---

## Τι κάνει η εφαρμογή σε μία πρόταση

Παίρνει ~110 μεγάλες αμερικανικές μετοχές, τις χωρίζει σε «καλάθια» με βάση 4 κλασικά ακαδημαϊκά κριτήρια (τους «παράγοντες» Fama-French), ποντάρει υπέρ των καλών καλαθιών και κατά των κακών, ενισχύει όποιο κριτήριο «τρέχει καλά» τελευταία και φρενάρει όποιο πάει άσχημα, και στο τέλος ρυθμίζει το συνολικό ρίσκο ώστε να μένει σταθερό. Μετά σου δείχνει πώς θα είχε πάει αυτή η στρατηγική ιστορικά (backtest).

**Οι 3 «στρώσεις» της στρατηγικής, με αναλογία:**

| Στρώση | Τι κάνει | Αναλογία |
|---|---|---|
| **Layer 1 — Factor Engine** | Χτίζει 4 μίνι-χαρτοφυλάκια (ένα ανά παράγοντα): αγοράζει τις μετοχές που «φοράνε» πιο πολύ τον παράγοντα, σορτάρει αυτές που τον φοράνε λιγότερο | Έχεις 4 παίκτες σε μια ομάδα, καθένας με το δικό του στυλ παιχνιδιού |
| **Layer 2 — Trend Overlay** | Κοιτάει ποιος «παίκτης» έχει φόρμα τελευταία και του δίνει περισσότερο χρόνο· όποιος είναι εκτός φόρμας παίζει λιγότερο (ή αντίστροφα) | Ο προπονητής που αλλάζει τη σύνθεση ανάλογα με τη φόρμα |
| **Layer 3 — Vol Targeting** | Ανεβοκατεβάζει το «γκάζι» (μόχλευση) ώστε το συνολικό ρίσκο να μένει σταθερό στο 10% ετησίως | Το cruise control του αυτοκινήτου: σταθερή ταχύτητα σε ανηφόρες και κατηφόρες |

**Οι 4 παράγοντες (τι σημαίνουν τα SMB / HML / RMW / CMA):**

- **SMB** (Small Minus Big) — μικρές εταιρείες vs μεγάλες. Ιστορικά οι μικρότερες αποδίδουν λίγο παραπάνω.
- **HML** (High Minus Low) — «φθηνές» μετοχές (value) vs «ακριβές» (growth).
- **RMW** (Robust Minus Weak) — εταιρείες με γερή κερδοφορία vs αδύναμη.
- **CMA** (Conservative Minus Aggressive) — εταιρείες που επενδύουν συντηρητικά vs επιθετικά.

---

## Βήμα 1 — Πώς ξεκινάς (30 δευτερόλεπτα)

1. Άνοιξε την εφαρμογή. Αριστερά βλέπεις τη **μπάρα ρυθμίσεων (sidebar)**.
2. Την πρώτη φορά **μην αλλάξεις τίποτα** — οι προεπιλογές είναι οι «σωστές» ακαδημαϊκές τιμές.
3. Πάτα το μπλε κουμπί **🚀 Run backtest**.
4. Περίμενε 1–3 λεπτά (κατεβάζει 15+ χρόνια δεδομένων — μόνο την πρώτη φορά, μετά τα θυμάται για 24 ώρες).
5. Δες τα αποτελέσματα στα tabs. Μετά πειραματίσου με τις ρυθμίσεις και ξαναπάτα Run.

---

## Βήμα 2 — Τι σημαίνει το κάθε κουμπί στο sidebar

### 🗓 Sample (η περίοδος του τεστ)

| Κουμπί | Τι κάνει | Πρακτικά |
|---|---|---|
| **Start / End date** | Από πότε μέχρι πότε τρέχει το ιστορικό τεστ | Μεγαλύτερη περίοδος = πιο αξιόπιστο συμπέρασμα, πιο αργό τρέξιμο |
| **Universe size** | Πόσες μετοχές μπαίνουν στο παιχνίδι (40–110) | Λιγότερες = πιο γρήγορο τρέξιμο για πειραματισμό. Για «σοβαρά» αποτελέσματα άσε το στο μέγιστο |

### ⚙️ Layer 1 · Factor engine (πώς χτίζονται τα καλάθια)

| Κουμπί | Τι κάνει | Πρακτικά |
|---|---|---|
| **Rolling OLS window** | Πόσες μέρες ιστορικού κοιτάει για να μετρήσει πόσο «φοράει» κάθε μετοχή τον κάθε παράγοντα (252 = 1 χρόνος συναλλαγών) | Μικρό παράθυρο = πιο «νευρικές» μετρήσεις, μεγάλο = πιο αργές να προσαρμοστούν |
| **Number of ranking buckets** | Σε πόσα κομμάτια κόβεται η κατάταξη. 10 = αγοράζεις το top 10% και σορτάρεις το bottom 10% | 10 = πιο «καθαρό» στοίχημα με λιγότερες μετοχές· 5 = πιο διαφοροποιημένο |
| **Traded factor sleeves** | Ποιους από τους 4 παράγοντες θα παίξεις | Βγάλε έναν για να δεις πόσο συνεισφέρει στο σύνολο |

### 📈 Layer 2 · Trend overlay (το «ραντάρ φόρμας»)

| Κουμπί | Τι κάνει | Πρακτικά |
|---|---|---|
| **Fast EWMA span** | Ο «γρήγορος» μέσος όρος (21 ≈ 1 μήνας) — πιάνει την πρόσφατη φόρμα | Μικρότερος = αντιδράει πιο γρήγορα αλλά με περισσότερους «ψεύτικους συναγερμούς» |
| **Slow EWMA span** | Ο «αργός» μέσος όρος (126 ≈ 6 μήνες) — η μεγάλη εικόνα | Το σήμα είναι η διαφορά γρήγορου − αργού: αν ο γρήγορος είναι πάνω από τον αργό, ο παράγοντας «ανεβαίνει» → μεγαλύτερη θέση |

### 🛡 Layer 3 · Vol targeting (το cruise control)

| Κουμπί | Τι κάνει | Πρακτικά |
|---|---|---|
| **Volatility target (%)** | Το επίπεδο ρίσκου-στόχος. 10% = ήπιες διακυμάνσεις (για σύγκριση: ο S&P 500 έχει ~15–20%) | Ανέβασέ το και θα δεις μεγαλύτερα κέρδη ΚΑΙ μεγαλύτερες βουτιές — το ρίσκο δεν χαρίζεται |
| **Leverage cap (×)** | Το μέγιστο «γκάζι». 2× = ποτέ πάνω από διπλάσια έκθεση | Δίχτυ ασφαλείας: σε πολύ ήσυχες αγορές το σύστημα θα ήθελε τεράστια μόχλευση — αυτό το φρενάρει |

### 💸 Frictions (το ρεαλιστικό κόστος)

| Κουμπί | Τι κάνει | Πρακτικά |
|---|---|---|
| **Transaction cost (bps)** | Πόσο κοστίζει κάθε αγοραπωλησία. 5 bps = 0,05% ανά συναλλαγή | Βάλ' το 0 για να δεις τον «παράδεισο χωρίς κόστη» και σύγκρινε — έτσι βλέπεις πόσο «τρώνε» οι προμήθειες |

---

## Βήμα 3 — Τα 5 νούμερα στην κορυφή (headline metrics)

Αφού τρέξει, βλέπεις 5 κουτάκια. Αυτά είναι η «ταυτότητα» της στρατηγικής:

- **CAGR** — η μέση ετήσια απόδοση. «Πόσο μεγάλωνε το κεφάλαιο κάθε χρόνο κατά μέσο όρο».
- **Sharpe** — απόδοση ανά μονάδα ρίσκου. **Το πιο σημαντικό νούμερο.** Κάτω από 0,5 = μέτριο, 0,5–1 = καλό, πάνω από 1 = πολύ καλό για long-short στρατηγική.
- **Max drawdown** — η χειρότερη «βουτιά» από κορυφή σε πάτο. −20% σημαίνει ότι κάποια στιγμή θα έβλεπες το 1/5 του κεφαλαίου να έχει εξαφανιστεί. Ρώτα τον εαυτό σου: θα το άντεχα;
- **Ann. vol (realized)** — το ρίσκο που τελικά βγήκε. Αν έβαλες στόχο 10%, θες να το βλέπεις κοντά στο 10% — σημάδι ότι το Layer 3 δουλεύει.
- **FF5 alpha (NW)** — η «καθαρή αξία» της στρατηγικής: η απόδοση που ΔΕΝ εξηγείται από τους ίδιους τους παράγοντες. Το `t` από κάτω είναι το τεστ αξιοπιστίας: **αν |t| > 2, το alpha μάλλον δεν είναι τυχαίο**. Αν t = 0,8, μπορεί να είναι απλώς θόρυβος.

---

## Βήμα 4 — Τα 6 tabs, ένα-ένα

### 📈 Performance
Το βασικό γράφημα: πώς μεγάλωσε $1 από την αρχή μέχρι σήμερα, για 4 γραμμές:
- **Static Factor (EW)** (γκρι) — μόνο το Layer 1, χωρίς timing. Το «σημείο αναφοράς».
- **Trend-Enhanced** (γαλάζιο) — Layer 1 + 2.
- **Trend + VolTarget (net)** (σκούρο μπλε, χοντρή γραμμή) — **η τελική στρατηγική**, με όλα τα layers ΚΑΙ τα κόστη.
- **SPY** (χρυσό) — απλά «αγόρασε τον S&P 500», για σύγκριση.

Η κλίμακα είναι λογαριθμική: ίδια κλίση = ίδιος ρυθμός ανάπτυξης, οπότε συγκρίνεις δίκαια παλιές και νέες χρονιές. Από κάτω, το γράφημα **Drawdowns** δείχνει πόσο κάτω από το ιστορικό υψηλό βρισκόταν η κάθε γραμμή ανά πάσα στιγμή — «ο χάρτης του πόνου». Πιο κάτω, οι καμπύλες των 4 sleeves ξεχωριστά, για να δεις ποιος παράγοντας δούλεψε και ποιος όχι.

### 🎛 Trend signals
- **Heatmap**: μπλε = το σύστημα ποντάρει ΥΠΕΡ του παράγοντα εκείνο τον μήνα, κόκκινο = ΚΑΤΑ, άσπρο = ουδέτερο. Βλέπεις με μια ματιά πότε π.χ. το HML ήταν «καυτό».
- **Current positioning**: τι θέσεις θα είχε το σύστημα ΣΗΜΕΡΑ. Μπάρα προς τα πάνω = long στον παράγοντα, προς τα κάτω = short.
- **Cumulative contribution**: πόσο κέρδος/ζημιά έφερε ο κάθε παράγοντας συνολικά στο timed χαρτοφυλάκιο.

### 🛡 Risk diagnostics
- **Vol-target leverage**: το «γκάζι» μέρα-μέρα. Όταν η αγορά είναι ήσυχη ανεβαίνει, όταν έχει πανικό πέφτει. Η κόκκινη γραμμή = το cap σου.
- **Realized vs target volatility**: η μπλε γραμμή (πραγματικό ρίσκο) πρέπει να χορεύει γύρω από την κόκκινη (στόχος). Αν ξεφεύγει συνέχεια, το cruise control δεν προλαβαίνει.
- **Rolling 1-year Sharpe**: η «φόρμα» της στρατηγικής σε κυλιόμενα 12μηνα. Καμία στρατηγική δεν είναι πάντα πάνω από το μηδέν — εδώ βλέπεις τις κακές περιόδους.
- **Monthly returns heatmap**: κάθε κελί = ένας μήνας. Πράσινο = κερδοφόρος, κόκκινο = ζημιογόνος. Ψάχνεις να δεις αν τα κόκκινα μαζεύονται σε συγκεκριμένες εποχές.

### 📊 Statistics
Ο πλήρης πίνακας με όλα τα στατιστικά για τις 4 παραλλαγές, δίπλα-δίπλα. Extra μεγέθη:
- **Sortino** — σαν το Sharpe, αλλά «τιμωρεί» μόνο τις κακές μέρες (πιο δίκαιο μέτρο).
- **Calmar** — απόδοση ÷ χειρότερη βουτιά. «Πόσο πληρώνομαι για τον χειρότερο πόνο».
- **Hit rate** — ποσοστό κερδοφόρων ημερών. Μην τρομάξεις αν είναι κάτω από 50% — οι trend στρατηγικές κερδίζουν λίγες φορές πολλά, όχι πολλές φορές λίγα.
- **Skew / Kurtosis** — το «σχήμα» των αποδόσεων: θετικό skew = σπάνιες μεγάλες θετικές εκπλήξεις (καλό), υψηλή kurtosis = συχνά ακραία γεγονότα προς οποιαδήποτε κατεύθυνση.

Κάτω-κάτω βλέπεις και ποιες μετοχές τελικά μπήκαν στο τεστ (όσες πέρασαν το φίλτρο 90% διαθεσιμότητας δεδομένων).

### 🔬 Sensitivity — το τεστ «μήπως κοροϊδεύω τον εαυτό μου;»
Το πιο έξυπνο tab. Ξανατρέχει το trend overlay με 16 διαφορετικούς συνδυασμούς (fast, slow) και δείχνει το Sharpe του καθενός.

**Γιατί έχει σημασία:** αν η στρατηγική δουλεύει ΜΟΝΟ με fast=21/slow=126 και πουθενά αλλού, τότε απλώς «διάλεξες» τυχαία τον έναν συνδυασμό που έτυχε να δουλέψει στο παρελθόν (overfitting) — και στο μέλλον μάλλον θα αποτύχει. Αν όμως **σχεδόν όλο το πλέγμα είναι θετικό**, το φαινόμενο είναι πραγματικό και όχι τυχαίο.

### 📥 Export
- **CSV κουμπιά**: κατεβάζεις τις ημερήσιες αποδόσεις και τα σήματα για να τα δουλέψεις σε Excel/Python.
- **Generate tearsheet**: φτιάχνει μια πλήρη «έκθεση θεσμικού τύπου» (QuantStats) σε ένα HTML αρχείο — δεκάδες γραφήματα και στατιστικά vs SPY, έτοιμο να το στείλεις ή να το βάλεις σε portfolio/βιογραφικό.

---

## Βήμα 5 — 4 πειράματα για να «καταλάβεις» τη στρατηγική

1. **Πόσο αξίζει το timing;** Κοίτα στο Performance τη γκρι γραμμή (static) vs τη γαλάζια (trend). Η διαφορά τους = η αξία του Layer 2.
2. **Πόσο τρώνε τα κόστη;** Τρέξε με 0 bps, σημείωσε το CAGR, ξανατρέξε με 10 bps. Η διαφορά είναι το «φιλοδώρημα» στους brokers.
3. **Ποιος παράγοντας κουβαλάει την ομάδα;** Βγάλε ένα-ένα τα sleeves και δες πώς αλλάζει το Sharpe.
4. **Είναι σταθερό ή τυχερό;** Άλλαξε την περίοδο (π.χ. 2010–2018 vs 2018–σήμερα). Αν τα συμπεράσματα αλλάζουν δραματικά, προσοχή.

---

## Συχνές απορίες

**«Long-short» τι σημαίνει;** Long = αγοράζεις (κερδίζεις αν ανέβει). Short = πουλάς δανεικά (κερδίζεις αν πέσει). Παίζοντας και τα δύο ταυτόχρονα, ποντάρεις στη ΔΙΑΦΟΡΑ καλών-κακών μετοχών, όχι στο αν θα ανέβει η αγορά συνολικά.

**Γιατί άργησε το πρώτο τρέξιμο;** Κατεβάζει 15+ χρόνια τιμών για ~110 μετοχές + τους παράγοντες. Αποθηκεύονται για 24 ώρες, οπότε κάθε επόμενο Run με άλλες παραμέτρους είναι σχεδόν ακαριαίο.

**Βγήκε error «Data download failed»;** Το Yahoo Finance βάζει προσωρινά όρια. Περίμενε 1 λεπτό και ξαναπάτα Run.

**Τα πρώτα ~1,5 χρόνια το γράφημα είναι «νεκρό» — γιατί;** Το σύστημα χρειάζεται 252 μέρες ιστορικού για να μετρήσει τα πρώτα betas, συν χρόνο για να «ζεσταθούν» τα trend σήματα. Είναι φυσιολογικό, όχι bug.

**Μπορώ να το εμπιστευτώ με λεφτά;** ❌ Όχι. Είναι ερευνητικό/εκπαιδευτικό εργαλείο με γνωστές αδυναμίες (π.χ. survivorship bias: το universe περιέχει εταιρείες που ξέρουμε ΣΗΜΕΡΑ ότι επιβίωσαν — το 2010 δεν το ήξερες). Τα ιστορικά backtests δεν εγγυώνται τίποτα για το μέλλον.
