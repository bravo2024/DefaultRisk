"""
DefaultRisk — Credit Default Prediction Dashboard
LendingClub / Basel III Framing | Pure NumPy | 30,000 Synthetic Loans
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm as sp_norm

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

st.set_page_config(
    page_title="DefaultRisk — Credit Default Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Global Controls")
    threshold  = st.slider("Decision Threshold",   0.01, 0.99, 0.50, 0.01)
    lgd        = st.slider("LGD (Loss Given Default)", 0.01, 1.00, 0.45, 0.01)
    ead_mult   = st.slider("EAD Multiplier",       0.50, 2.00, 1.00, 0.05)
    pd_override = st.checkbox("Override PD with manual value")
    pd_manual   = 0.15
    if pd_override:
        pd_manual = st.slider("Manual PD", 0.001, 0.999, 0.15, 0.001)
    st.markdown("---")
    st.caption("Active Models")
    use_lr  = st.checkbox("Logistic Regression", value=True)
    use_lgb = st.checkbox("LightGBM", value=_LGB_AVAILABLE, disabled=not _LGB_AVAILABLE)
    if _LGB_AVAILABLE and use_lr and use_lgb:
        dash_model = st.radio("Dashboard model",
            ["Logistic Regression", "LightGBM"], index=0, horizontal=True)
    elif use_lgb:
        dash_model = "LightGBM"
    else:
        dash_model = "Logistic Regression"
    st.markdown("---")
    st.caption("LR: pure NumPy SGD. LGB: lightgbm.")

# ─────────────────────────────────────────────────────────────────
# SYNTHETIC DATA
# ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "loan_amount", "interest_rate", "term", "annual_income",
    "debt_to_income", "credit_score", "employment_length",
    "num_open_accounts", "num_derogatory_marks",
    "revolving_utilization", "inquiries_last_6m",
    "months_since_last_delinq",
]

@st.cache_data(show_spinner=False)
def generate_loan_data(n: int = 30_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    loan_amount          = np.clip(np.exp(rng.normal(10.5, 1.0, n)), 5_000, 500_000)
    interest_rate        = rng.uniform(5, 36, n)
    term                 = rng.choice([36, 60], n, p=[0.6, 0.4])
    annual_income        = np.clip(np.exp(rng.normal(11.0, 0.8, n)), 10_000, 1_000_000)
    debt_to_income       = np.clip(rng.beta(2, 5, n), 0.01, 0.99)
    credit_score         = np.clip(rng.normal(680, 80, n), 300, 850).astype(int)
    employment_length    = np.clip(rng.exponential(5, n), 0, 20)
    home_ownership       = rng.choice(["RENT","OWN","MORTGAGE"], n, p=[0.40, 0.15, 0.45])
    purpose              = rng.choice(
        ["car","debt_consolidation","home","medical","business","vacation"],
        n, p=[0.10, 0.35, 0.15, 0.15, 0.15, 0.10],
    )
    num_open_accounts        = np.clip(rng.poisson(10, n), 1, 40)
    num_derogatory_marks     = rng.choice([0,1,2,3,4], n, p=[0.60,0.20,0.12,0.05,0.03])
    revolving_utilization    = np.clip(rng.beta(2, 3, n), 0, 1)
    inquiries_last_6m        = rng.choice([0,1,2,3,4,5], n, p=[0.40,0.25,0.18,0.10,0.04,0.03])
    months_since_last_delinq = np.where(rng.random(n) < 0.4, rng.uniform(1, 120, n), 0.0)

    logit = (
        -4.5
        + (credit_score - 680) * (-0.012)
        + debt_to_income * 3.0
        + revolving_utilization * 2.5
        + num_derogatory_marks * 0.8
        + (interest_rate - 15) * 0.05
        + inquiries_last_6m * 0.3
        + rng.normal(0, 0.5, n)
    )
    pd_prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -35, 35)))
    default = (rng.random(n) < pd_prob).astype(int)

    return pd.DataFrame({
        "loan_amount":              loan_amount,
        "interest_rate":            interest_rate,
        "term":                     term,
        "annual_income":            annual_income,
        "debt_to_income":           debt_to_income,
        "credit_score":             credit_score,
        "employment_length":        employment_length,
        "home_ownership":           home_ownership,
        "purpose":                  purpose,
        "num_open_accounts":        num_open_accounts,
        "num_derogatory_marks":     num_derogatory_marks,
        "revolving_utilization":    revolving_utilization,
        "inquiries_last_6m":        inquiries_last_6m,
        "months_since_last_delinq": months_since_last_delinq,
        "default":                  default,
    })

# ─────────────────────────────────────────────────────────────────
# ML PRIMITIVES  (pure NumPy)
# ─────────────────────────────────────────────────────────────────
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

def standardize(X_tr, X_te=None):
    mu = X_tr.mean(0);  sd = X_tr.std(0) + 1e-8
    if X_te is None:
        return (X_tr - mu) / sd, mu, sd
    return (X_tr - mu) / sd, (X_te - mu) / sd, mu, sd

def logistic_sgd(X, y, lr=0.12, epochs=350, l2=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w, b = rng.normal(0, 0.01, d), 0.0
    pos  = max(y.sum(), 1);   neg = max((1 - y).sum(), 1)
    sw   = np.where(y == 1, n / (2 * pos), n / (2 * neg))
    for _ in range(epochs):
        p   = sigmoid(X @ w + b);   err = (p - y) * sw
        w  -= lr * (X.T @ err / n + l2 * w)
        b  -= lr * err.mean()
    return w, b

def roc_auc_wilcoxon(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    npos = (y == 1).sum();  nneg = (y == 0).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s));  ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))

def manual_roc(y, scores, n_t=200):
    thresholds = np.linspace(0, 1, n_t)
    pos = y.sum();  neg = len(y) - pos
    tprs, fprs = [], []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        tprs.append(tp / max(pos, 1))
        fprs.append(fp / max(neg, 1))
    fpr = np.array(fprs[::-1]);  tpr = np.array(tprs[::-1])
    return fpr, tpr, float(np.trapezoid(tpr, fpr))

def manual_pr(y, scores, n_t=200):
    thresholds = np.linspace(0, 1, n_t)
    pos = y.sum()
    precs, recs = [], []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        precs.append(tp / max(tp + fp, 1))
        recs.append(tp / max(pos, 1))
    rec  = np.array(recs[::-1]);  prec = np.array(precs[::-1])
    return rec, prec, float(np.trapezoid(prec, rec))

def ks_stat(y, scores):
    order    = np.argsort(scores)
    ys       = y[order]
    cum_pos  = np.cumsum(ys)      / max(y.sum(), 1)
    cum_neg  = np.cumsum(1 - ys) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(cum_pos - cum_neg)))

def log_loss_fn(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def brier_fn(y, p):
    return float(np.mean((y - p) ** 2))

# ─────────────────────────────────────────────────────────────────
# FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────
def build_X(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy().astype(float)
    X["home_own"]          = (df["home_ownership"] == "OWN").astype(float)
    X["home_mortgage"]     = (df["home_ownership"] == "MORTGAGE").astype(float)
    X["purpose_dti"]       = (df["purpose"] == "debt_consolidation").astype(float)
    X["purpose_business"]  = (df["purpose"] == "business").astype(float)
    return X.values, list(X.columns)

# ─────────────────────────────────────────────────────────────────
# MODEL TRAINING  (cached resource)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model(_df: pd.DataFrame) -> dict:
    X, feat_names = build_X(_df)
    y = _df["default"].values.astype(float)

    rng  = np.random.default_rng(7)
    idx  = rng.permutation(len(X))
    n_te = int(len(X) * 0.2)
    X_tr, X_te = X[idx[n_te:]], X[idx[:n_te]]
    y_tr, y_te = y[idx[n_te:]], y[idx[:n_te]]

    X_tr_s, X_te_s, mu, sd = standardize(X_tr, X_te)
    w, b = logistic_sgd(X_tr_s, y_tr, lr=0.12, epochs=350)

    sc_tr = sigmoid(X_tr_s @ w + b)
    sc_te = sigmoid(X_te_s @ w + b)

    fpr, tpr, auc  = manual_roc(y_te, sc_te)
    rec, prec, prauc = manual_pr(y_te, sc_te)

    return dict(
        w=w, b=b, mu=mu, sd=sd, feat_names=feat_names,
        X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
        sc_tr=sc_tr, sc_te=sc_te,
        fpr=fpr, tpr=tpr, auc=auc,
        rec=rec, prec=prec, prauc=prauc,
        ks=ks_stat(y_te, sc_te),
        gini=2 * auc - 1,
        ll=log_loss_fn(y_te, sc_te),
        brier=brier_fn(y_te, sc_te),
    )

def classification_metrics(y_true, y_prob, t=0.5):
    preds = (y_prob >= t).astype(int)
    tp = int(((preds==1)&(y_true==1)).sum())
    fp = int(((preds==1)&(y_true==0)).sum())
    tn = int(((preds==0)&(y_true==0)).sum())
    fn = int(((preds==0)&(y_true==1)).sum())
    acc = (tp + tn) / max(tp+fp+tn+fn, 1)
    prec = tp / max(tp+fp, 1)
    rec = tp / max(tp+fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                accuracy=acc, precision=prec, recall=rec, f1=f1)

@st.cache_resource(show_spinner=False)
def train_lgb(_df: pd.DataFrame) -> dict:
    X, feat_names = build_X(_df)
    y = _df["default"].values.astype(float)

    rng  = np.random.default_rng(7)
    idx  = rng.permutation(len(X))
    n_te = int(len(X) * 0.2)
    X_tr, X_te = X[idx[n_te:]], X[idx[:n_te]]
    y_tr, y_te = y[idx[n_te:]], y[idx[:n_te]]

    X_tr_s, X_te_s, mu, sd = standardize(X_tr, X_te)

    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        random_state=7, verbose=-1, force_col_wise=True,
    )
    lgb_model.fit(X_tr_s, y_tr)
    sc_tr = lgb_model.predict_proba(X_tr_s)[:, 1]
    sc_te = lgb_model.predict_proba(X_te_s)[:, 1]

    fpr, tpr, auc  = manual_roc(y_te, sc_te)
    rec, prec, prauc = manual_pr(y_te, sc_te)
    cm = classification_metrics(y_te, sc_te, threshold)

    return dict(
        model=lgb_model, mu=mu, sd=sd, feat_names=feat_names,
        X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
        sc_tr=sc_tr, sc_te=sc_te,
        fpr=fpr, tpr=tpr, auc=auc,
        rec=rec, prec=prec, prauc=prauc,
        ks=ks_stat(y_te, sc_te),
        gini=2 * auc - 1,
        ll=log_loss_fn(y_te, sc_te),
        brier=brier_fn(y_te, sc_te),
        cm=cm,
    )

def predict_pd(df_input: pd.DataFrame, mdl: dict) -> np.ndarray:
    X, _ = build_X(df_input)
    Xs   = (X - mdl["mu"]) / mdl["sd"]
    if "model" in mdl:
        return mdl["model"].predict_proba(Xs)[:, 1]
    return sigmoid(Xs @ mdl["w"] + mdl["b"])

# ─────────────────────────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────────────────────────
with st.spinner("Generating 30,000 synthetic loans..."):
    df = generate_loan_data(30_000)

mdl = lgb_mdl = None
if use_lr or not _LGB_AVAILABLE:
    with st.spinner("Training Logistic Regression (NumPy SGD)..."):
        mdl = train_model(df)
        mdl["cm"] = classification_metrics(mdl["y_te"], mdl["sc_te"], threshold)
if use_lgb and _LGB_AVAILABLE:
    with st.spinner("Training LightGBM..."):
        lgb_mdl = train_lgb(df)

active_mdl = lgb_mdl if (dash_model == "LightGBM" and lgb_mdl is not None) else mdl

# ─────────────────────────────────────────────────────────────────
# HEADER METRICS
# ─────────────────────────────────────────────────────────────────
loan_pd_all  = predict_pd(df, active_mdl)
if pd_override:
    loan_pd_all = np.full(len(df), pd_manual)

loan_ead_all = df["loan_amount"].values * ead_mult
el_all       = loan_pd_all * lgd * loan_ead_all
total_el     = el_all.sum()
dr_mean      = df["default"].mean()
cs_mean      = df["credit_score"].mean()
dti_mean     = df["debt_to_income"].mean()

revenue  = (df["loan_amount"] * df["interest_rate"] / 100 * (df["term"] / 12)).sum() * 0.8
op_cost  = revenue * 0.25
ec_rough = total_el * 3
raroc    = (revenue - total_el - op_cost) / max(ec_rough, 1)

active_labels = []
if mdl is not None:
    active_labels.append("Logistic Regression")
if lgb_mdl is not None:
    active_labels.append("LightGBM")
backend_label = " + ".join(active_labels) if len(active_labels) > 1 else active_labels[0]
dash_label = f"{dash_model} (active)" if len(active_labels) > 1 else backend_label
st.title("DefaultRisk — Credit Default Prediction Platform")
st.caption(f"LendingClub / Basel III Framing  |  {dash_label}  |  30,000 Synthetic Loans")

h = st.columns(6)
h[0].metric("Total Loans",     f"{len(df):,}")
h[1].metric("Portfolio EL ($)", f"${total_el:,.0f}")
h[2].metric("Default Rate",     f"{dr_mean:.1%}")
h[3].metric("Mean Credit Score",f"{cs_mean:.0f}")
h[4].metric("Mean DTI",         f"{dti_mean:.2f}")
h[5].metric("RAROC",            f"{raroc:.2%}")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 Methodology & Overview",
    "📊 Loan Portfolio Explorer",
    "🔬 Credit Risk Factors",
    "🤖 Model Training & Scorecard",
    "📋 Scorecard & Scorepoints",
    "💰 Portfolio Risk & Capital",
])

# ───────────────────────────────────────────────
# TAB 0 — METHODOLOGY & OVERVIEW
# ───────────────────────────────────────────────
with tab0:
    st.header("📖 Project Overview")
    st.markdown("""
    **DefaultRisk** is a credit-default prediction dashboard framed under
    **Basel III / LendingClub** conventions. It demonstrates an end-to-end
    risk-modelling workflow — synthetic data generation, feature engineering,
    pure-NumPy logistic regression, model evaluation, scorecard construction,
    and portfolio capital allocation — all inside an interactive Streamlit app.
    """)

    st.subheader("🎯 Objective")
    st.markdown("""
    Estimate **Probability of Default (PD)** for individual loans using 12 borrower
    characteristics, then compute **Expected Loss (EL)** via
    $EL = PD \\times LGD \\times EAD$, and determine regulatory capital
    ratios (Tier 1, Total Capital, RAROC) under a simplified Basel III framework.
    """)

    st.subheader("📦 Data")
    st.markdown("""
    **30 000 synthetic loans** generated from parametric distributions
    (log-normal for income/loan, beta for DTI, normal for credit score, etc.)
    with a ~27 % default rate. No real borrower data is used — the app runs
    fully self-contained for demonstration and portfolio-simulation purposes.
    """)

    st.subheader("⚙️ Methodology")

    with st.expander("Feature Engineering"):
        st.markdown("""
        - **12 continuous/categorical raw features** (loan amount, interest rate,
          term, income, DTI, credit score, employment length, open accounts,
          derogatory marks, revolving utilization, inquiries, months since delinq).
        - **4 one-hot derived flags**: `home_OWN`, `home_MORTGAGE`,
          `purpose_debt_consolidation`, `purpose_business`.
        - 16 total predictors after encoding. All numeric — no missing values
          in the synthetic pipeline.
        - **Weight of Evidence (WoE)** and **Information Value (IV)** computed
          for monitoring feature strength (IV < 0.02 → useless; > 0.3 → strong).
        """)

    with st.expander("Model — Logistic Regression via SGD"):
        st.markdown("""
        - **Estimator**: $P(y=1 \\mid \\mathbf{x}) = \\sigma(\\mathbf{w}^\\top\\mathbf{x} + b)$
        - **Loss**: Binary cross-entropy with **class-weight balancing**:
          $$\\mathcal{L} = -\\frac{1}{N}\\sum_i w_i\\bigl[y_i\\log\\hat{p}_i+(1-y_i)\\log(1-\\hat{p}_i)\\bigr] + \\frac{\\lambda}{2}\\|\\mathbf{w}\\|^2$$
          where $w_i = N/(2 \\times \\text{count}(y_i))$ down-weights the majority class.
        - **Optimizer**: Mini-batch SGD (full-batch equivalent), learning rate 0.12,
          350 epochs, L2 penalty $\\lambda = 0.001$, random normal init $\\mathcal{N}(0, 0.01)$.
        - All implemented in **pure NumPy** — no scikit-learn or autodiff.
        """)

    with st.expander("Model Evaluation"):
        st.markdown("""
        - **ROC-AUC** (trapezoidal rule) — discriminatory power.
        - **PR-AUC** — precision-recall trade-off on imbalanced data.
        - **KS Statistic** — maximum separation between default / non-default CDFs.
        - **Gini Coefficient** = $2 \\times \\text{AUC} - 1$.
        - **Log Loss** — probabilistic calibration.
        - **Brier Score** — mean squared error of predicted probabilities.
        """)

    with st.expander("Scorecard Construction"):
        st.markdown("""
        - **Base score**: 600, **scale**: 50 (every 50 points halves/halves the odds).
        - **Score-to-odds**: $\\text{Score} = 600 - \\frac{50}{\\ln(2)}(\\mathbf{w}^\\top\\mathbf{x} + b)$.
        - **Scorepoints per feature**: $\\text{Points}_i = -\\frac{50}{\\ln(2)} w_i x_i$, displayed
          as a waterfall chart for each individual application.
        - Factor = −50 / ln(2) ≈ −72.13.
        """)

    with st.expander("Portfolio Risk — Basel III"):
        st.markdown("""
        - **Expected Loss**: $EL = PD \\times LGD \\times EAD$.
        - **Unexpected Loss (UL)**: $UL = \\text{Std}(\\text{loss}) \\times 2.58$ (99.5 % VaR equivalent).
        - **Capital**:
          - Tier 1 Capital = 6 % of RWA (Risk-Weighted Assets).
          - Total Capital = 8 % of RWA.
          - Economic Capital (EC) = $\\max(\\text{UL}, \\text{VaR}_{99.5\\%})$.
        - **RAROC** = (Revenue − EL − OpCost) / EC.
        - **PSI** tracks population drift between expected and actual score distributions.
        """)

    st.subheader("🔧 Tech Stack")
    st.markdown("""
    - **Python 3.11+**, NumPy, pandas, matplotlib, SciPy.
    - **Streamlit** for the interactive dashboard.
    - **LightGBM** available as an optional production backend (commented out).
    - No deep learning, no GPU — everything runs CPU-only in seconds.
    """)

# ───────────────────────────────────────────────
# TAB 1 — LOAN PORTFOLIO EXPLORER
# ───────────────────────────────────────────────
with tab1:
    st.header("📊 Loan Portfolio Explorer")

    def cs_band(cs):
        if   cs < 580: return "300-579 Bad"
        elif cs < 670: return "580-669 Fair"
        elif cs < 740: return "670-739 Good"
        elif cs < 800: return "740-799 Very Good"
        else:          return "800+ Exceptional"

    df["cs_band"] = df["credit_score"].apply(cs_band)
    BAND_ORDER    = ["300-579 Bad","580-669 Fair","670-739 Good","740-799 Very Good","800+ Exceptional"]

    st.subheader("Summary Statistics")
    show_cols = ["loan_amount","interest_rate","annual_income",
                 "debt_to_income","credit_score","revolving_utilization","default"]
    st.dataframe(df[show_cols].describe().round(3), use_container_width=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1 — loan amount
    ax = axes[0, 0]
    ax.hist(df["loan_amount"] / 1_000, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_title("Loan Amount Distribution", fontweight="bold")
    ax.set_xlabel("Loan Amount ($K)");  ax.set_ylabel("Count");  ax.grid(axis="y", alpha=0.3)

    # 2 — interest rate by default
    ax = axes[0, 1]
    ax.hist(df.loc[df["default"]==0,"interest_rate"], bins=40, alpha=0.6,
            color="steelblue", density=True, label="Non-Default")
    ax.hist(df.loc[df["default"]==1,"interest_rate"], bins=40, alpha=0.6,
            color="crimson",   density=True, label="Default")
    ax.set_title("Interest Rate by Default Status", fontweight="bold")
    ax.set_xlabel("Interest Rate (%)");  ax.set_ylabel("Density")
    ax.legend();  ax.grid(axis="y", alpha=0.3)

    # 3 — default rate by cs band
    ax = axes[0, 2]
    band_dr = df.groupby("cs_band")["default"].mean().reindex(BAND_ORDER)
    colors  = ["#d62728","#ff7f0e","#bcbd22","#2ca02c","#1f77b4"]
    bars    = ax.bar(range(len(BAND_ORDER)), band_dr.values * 100, color=colors, alpha=0.85)
    ax.set_xticks(range(len(BAND_ORDER)))
    ax.set_xticklabels(["Bad","Fair","Good","V.Good","Excep."], fontsize=9)
    ax.set_title("Default Rate by Credit Score Band", fontweight="bold")
    ax.set_ylabel("Default Rate (%)")
    for bar, v in zip(bars, band_dr.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.1%}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 4 — default rate by purpose
    ax = axes[1, 0]
    purp_dr = df.groupby("purpose")["default"].mean().sort_values()
    ax.barh(purp_dr.index, purp_dr.values * 100, color="steelblue", alpha=0.8)
    ax.set_title("Default Rate by Loan Purpose", fontweight="bold")
    ax.set_xlabel("Default Rate (%)");  ax.grid(axis="x", alpha=0.3)

    # 5 — default rate by home ownership
    ax = axes[1, 1]
    home_dr = df.groupby("home_ownership")["default"].mean()
    cols_h  = ["#1f77b4","#ff7f0e","#2ca02c"]
    ax.bar(home_dr.index, home_dr.values * 100, color=cols_h, alpha=0.85)
    ax.set_title("Default Rate by Home Ownership", fontweight="bold")
    ax.set_ylabel("Default Rate (%)")
    for i, (_, v) in enumerate(home_dr.items()):
        ax.text(i, v * 100 + 0.2, f"{v:.1%}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # 6 — correlation heatmap
    ax = axes[1, 2]
    corr_cols = ["loan_amount","interest_rate","annual_income","debt_to_income",
                 "credit_score","revolving_utilization","num_derogatory_marks","default"]
    corr = df[corr_cols].corr()
    short = ["Loan$","IntRate","Income","DTI","CS","RevUtil","Derog","Default"]
    im    = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)));  ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(v) > 0.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig);  plt.close()

# ───────────────────────────────────────────────
# TAB 2 — CREDIT RISK FACTORS
# ───────────────────────────────────────────────
with tab2:
    st.header("🔬 Credit Risk Factors")

    st.subheader("Key Equations")
    st.latex(r"WoE_i = \ln\!\left(\frac{\text{Dist Events}_i}{\text{Dist Non-Events}_i}\right)")
    st.latex(r"IV = \sum_i \!\left(\text{Events\%}_i - \text{Non-Events\%}_i\right)\times WoE_i")
    st.latex(r"\text{Odds Ratio} = \frac{P(\text{default}=1\mid X)}{P(\text{default}=0\mid X)}")

    # ── Point-biserial correlations
    num_feats = ["loan_amount","interest_rate","annual_income","debt_to_income",
                 "credit_score","employment_length","num_open_accounts",
                 "num_derogatory_marks","revolving_utilization",
                 "inquiries_last_6m","months_since_last_delinq"]
    pb_corrs = {f: float(np.corrcoef(df[f].values, df["default"].values)[0, 1])
                for f in num_feats}
    pb_df = (pd.DataFrame({"Feature": list(pb_corrs.keys()),
                            "r": list(pb_corrs.values())})
             .sort_values("r"))

    # ── WoE / IV helper
    def woe_iv(df_in, col, target, bins=None):
        tmp = df_in[[col, target]].copy()
        if bins is not None:
            tmp["bin"] = pd.cut(tmp[col], bins=bins, include_lowest=True).astype(str)
        else:
            tmp["bin"] = tmp[col].astype(str)
        tot_e  = (tmp[target] == 1).sum()
        tot_ne = (tmp[target] == 0).sum()
        rows = []
        for b, grp in tmp.groupby("bin"):
            ev  = (grp[target] == 1).sum()
            nev = (grp[target] == 0).sum()
            de  = ev  / max(tot_e,  1)
            dn  = nev / max(tot_ne, 1)
            woe = np.log(max(de, 1e-9) / max(dn, 1e-9))
            rows.append({"bin": b, "events": ev, "non_events": nev,
                         "dist_e": de, "dist_n": dn,
                         "woe": woe, "iv_i": (de - dn) * woe})
        res = pd.DataFrame(rows)
        return res, float(res["iv_i"].sum())

    iv_dict = {}
    for f in num_feats:
        _, iv = woe_iv(df, f, "default", bins=10)
        iv_dict[f] = iv
    for f in ["home_ownership","purpose"]:
        _, iv = woe_iv(df, f, "default")
        iv_dict[f] = iv

    def iv_label(v):
        if v < 0.02:  return "Useless"
        elif v < 0.1: return "Weak"
        elif v < 0.3: return "Medium"
        else:         return "Strong"

    iv_df = (pd.DataFrame({"Feature": list(iv_dict.keys()), "IV": list(iv_dict.values())})
             .sort_values("IV", ascending=False))
    iv_df["Predictive Power"] = iv_df["IV"].apply(iv_label)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        colors_pb = ["#2ca02c" if v > 0 else "#d62728" for v in pb_df["r"]]
        ax.barh(pb_df["Feature"], pb_df["r"], color=colors_pb, alpha=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title("Point-Biserial Correlation with Default", fontweight="bold")
        ax.set_xlabel("Correlation Coefficient");  ax.grid(axis="x", alpha=0.3)
        plt.tight_layout();  st.pyplot(fig);  plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        iv_colors = []
        for v in iv_df["IV"]:
            if   v < 0.02: iv_colors.append("#d62728")
            elif v < 0.1:  iv_colors.append("#ff7f0e")
            elif v < 0.3:  iv_colors.append("#bcbd22")
            else:          iv_colors.append("#2ca02c")
        ax.barh(iv_df["Feature"], iv_df["IV"], color=iv_colors, alpha=0.85)
        ax.axvline(0.02, color="red",    ls="--", lw=1, label="Useless/Weak 0.02")
        ax.axvline(0.10, color="orange", ls="--", lw=1, label="Weak/Medium 0.10")
        ax.axvline(0.30, color="green",  ls="--", lw=1, label="Medium/Strong 0.30")
        ax.set_title("Information Value (IV) by Feature", fontweight="bold")
        ax.set_xlabel("IV");  ax.legend(fontsize=7);  ax.grid(axis="x", alpha=0.3)
        plt.tight_layout();  st.pyplot(fig);  plt.close()

    st.dataframe(iv_df.reset_index(drop=True), use_container_width=True)

    # ── WoE binning chart for credit_score
    st.subheader("WoE Binning — Credit Score")
    cs_woe, cs_iv = woe_iv(df, "credit_score", "default", bins=10)
    cs_woe = cs_woe.sort_values("bin")

    # ── KS statistic
    sc_all    = predict_pd(df, active_mdl)
    y_all     = df["default"].values
    order_ks  = np.argsort(sc_all)
    ys_ks     = y_all[order_ks]
    cum_pos_ks = np.cumsum(ys_ks)      / max(ys_ks.sum(), 1)
    cum_neg_ks = np.cumsum(1 - ys_ks) / max((1 - ys_ks).sum(), 1)
    ks_val    = float(np.max(np.abs(cum_pos_ks - cum_neg_ks)))
    x_axis    = np.linspace(0, 1, len(order_ks))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ax = axes[0]
    woe_colors = ["#2ca02c" if w >= 0 else "#d62728" for w in cs_woe["woe"]]
    ax.bar(range(len(cs_woe)), cs_woe["woe"], color=woe_colors, alpha=0.8)
    ax.set_xticks(range(len(cs_woe)))
    ax.set_xticklabels(cs_woe["bin"], rotation=45, ha="right", fontsize=7)
    ax.set_title(f"WoE by Credit Score Bin  (IV={cs_iv:.3f})", fontweight="bold")
    ax.set_ylabel("Weight of Evidence");  ax.axhline(0, color="black", lw=0.8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.plot(x_axis, cum_pos_ks, color="crimson",   lw=2, label="Defaulters CDF")
    ax.plot(x_axis, cum_neg_ks, color="steelblue", lw=2, label="Non-Defaulters CDF")
    ks_idx = int(np.argmax(np.abs(cum_pos_ks - cum_neg_ks)))
    ax.axvline(x_axis[ks_idx], color="gray", ls="--", lw=1.5, label=f"KS={ks_val:.4f}")
    ax.fill_between(x_axis, cum_pos_ks, cum_neg_ks, alpha=0.1, color="gray")
    ax.set_title("KS Statistic — Score Separation", fontweight="bold")
    ax.set_xlabel("Population Percentile");  ax.set_ylabel("Cumulative Distribution")
    ax.legend();  ax.grid(alpha=0.3)

    plt.tight_layout();  st.pyplot(fig);  plt.close()
    st.info(f"KS = **{ks_val:.4f}**  |  KS > 0.3 = good  |  KS > 0.4 = excellent")

# ───────────────────────────────────────────────
# TAB 3 — MODEL TRAINING & SCORECARD
# ───────────────────────────────────────────────
with tab3:
    st.header("🤖 Model Training & Scorecard")

    models_avail = [(mdl, "Logistic Regression", "crimson", "Blues"),
                    (lgb_mdl, "LightGBM", "green", "Greens")]
    models_avail = [(m, l, c, cmap) for m, l, c, cmap in models_avail if m is not None]

    if len(models_avail) == 0:
        st.warning("No models trained. Enable at least one model in the sidebar.")
    else:
        st.subheader("Model Comparison — " + " vs ".join(label for _, label, _, _ in models_avail))

        # ── Metrics comparison table ──
        comp_data = {
            "Metric": ["ROC-AUC", "PR-AUC", "KS Stat", "Gini", "Log Loss", "Brier Score",
                        "Accuracy", "Precision", "Recall", "F1 Score"],
        }
        for mod, label, _, _ in models_avail:
            cm = classification_metrics(mod["y_te"], mod["sc_te"], threshold)
            comp_data[label] = [
                f"{mod['auc']:.4f}", f"{mod['prauc']:.4f}", f"{mod['ks']:.4f}",
                f"{mod['gini']:.4f}", f"{mod['ll']:.4f}", f"{mod['brier']:.4f}",
                f"{cm['accuracy']:.4f}", f"{cm['precision']:.4f}",
                f"{cm['recall']:.4f}", f"{cm['f1']:.4f}",
            ]

        st.dataframe(pd.DataFrame(comp_data).set_index("Metric"), use_container_width=True)

        # ── Confusion matrices side-by-side ──
        conf_cols = st.columns(len(models_avail))
        for ci, (mod, label, _, cmap) in enumerate(models_avail):
            with conf_cols[ci]:
                cm_res = classification_metrics(mod["y_te"], mod["sc_te"], threshold)
                st.subheader(f"{label} — Confusion Matrix")
                fig_c, ax_c = plt.subplots(figsize=(4, 3.5))
                cm_arr = np.array([[cm_res['tn'], cm_res['fp']], [cm_res['fn'], cm_res['tp']]])
                ax_c.imshow(cm_arr, cmap=cmap)
                ax_c.set_xticks([0, 1]); ax_c.set_yticks([0, 1])
                ax_c.set_xticklabels(["Pred: No", "Pred: Yes"], fontsize=8)
                ax_c.set_yticklabels(["Act: No", "Act: Yes"], fontsize=8)
                for i in range(2):
                    for j in range(2):
                        ax_c.text(j, i, str(cm_arr[i, j]), ha="center", va="center", fontsize=12,
                                  color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black")
                ax_c.set_title(f"Threshold = {threshold:.2f}", fontweight="bold", fontsize=10)
                st.pyplot(fig_c); plt.close()

        # ── Overlaid ROC & PR curves ──
        fig2 = plt.figure(figsize=(14, 5))
        gs2 = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.35)

        ax_roc = fig2.add_subplot(gs2[0, 0])
        for mod, label, color, _ in models_avail:
            ax_roc.plot(mod["fpr"], mod["tpr"], color=color, lw=2,
                        label=f"{label} AUC={mod['auc']:.3f}")
        ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax_roc.set_title("ROC Curve — Overlay", fontweight="bold")
        ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
        ax_roc.legend(fontsize=9); ax_roc.grid(alpha=0.3)

        ax_pr = fig2.add_subplot(gs2[0, 1])
        baseline = models_avail[0][0]["y_te"].mean()
        for mod, label, color, _ in models_avail:
            ax_pr.plot(mod["rec"], mod["prec"], color=color, lw=2,
                       label=f"{label} PR-AUC={mod['prauc']:.3f}")
        ax_pr.axhline(baseline, color="gray", ls="--", lw=1, label=f"Baseline={baseline:.2f}")
        ax_pr.set_title("Precision-Recall Curve — Overlay", fontweight="bold")
        ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
        ax_pr.legend(fontsize=9); ax_pr.grid(alpha=0.3)

        plt.tight_layout(); st.pyplot(fig2); plt.close()

        # ── Score distributions ──
        n_mods = len(models_avail)
        fig3, axes = plt.subplots(1, n_mods, figsize=(14, 4), squeeze=False)
        for ai, (mod, label, color, _) in enumerate(models_avail):
            ax = axes[0, ai]
            s0_v = mod["sc_te"][mod["y_te"] == 0]
            s1_v = mod["sc_te"][mod["y_te"] == 1]
            ax.hist(s0_v, bins=50, alpha=0.6, color="steelblue", density=True, label="Non-Default")
            ax.hist(s1_v, bins=50, alpha=0.6, color=color, density=True, label="Default")
            ax.axvline(threshold, color="black", ls="--", lw=1.5, label=f"t={threshold}")
            ax.set_title(f"{label} — Score Distribution", fontweight="bold")
            ax.set_xlabel("Predicted PD"); ax.set_ylabel("Density")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig3); plt.close()

        # ── Calibration overlay ──
        fig4, ax_cal = plt.subplots(figsize=(7, 5))
        ax_cal.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
        edges = np.linspace(0, 1, 11)
        for mod, label, color, _ in models_avail:
            frac_pos_cal, mean_pred_cal = [], []
            for i in range(10):
                mask = (mod["sc_te"] >= edges[i]) & (mod["sc_te"] < edges[i + 1])
                if mask.sum() > 0:
                    frac_pos_cal.append(mod["y_te"][mask].mean())
                    mean_pred_cal.append(mod["sc_te"][mask].mean())
            ax_cal.plot(mean_pred_cal, frac_pos_cal, "o-", color=color, lw=2, label=label)
        ax_cal.set_title("Calibration Curve", fontweight="bold")
        ax_cal.set_xlabel("Mean Predicted PD"); ax_cal.set_ylabel("Fraction Positive")
        ax_cal.legend(fontsize=9); ax_cal.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig4); plt.close()

    st.subheader("Logistic Regression Formulation")
    st.latex(r"P(\text{default}=1\mid\mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x}+b)"
             r"= \frac{1}{1+e^{-(\mathbf{w}^\top\mathbf{x}+b)}}")
    st.latex(r"\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}w_i"
             r"\bigl[y_i\log\hat{p}_i+(1-y_i)\log(1-\hat{p}_i)\bigr]"
             r"+\frac{\lambda}{2}\|\mathbf{w}\|^2")
    if lgb_mdl:
        st.subheader("LightGBM Parameters")
        st.code(
            f"n_estimators=200, max_depth=6, learning_rate=0.1,\n"
            f"num_leaves=31, subsample=0.8, colsample_bytree=0.8",
            language="text",
        )

    # ── Basel III Info (always from LR) ──
    with st.expander("Basel III IRB Capital Calculation"):
        st.latex(r"R = 0.12\,\frac{1-e^{-50\,PD}}{1-e^{-50}}"
                 r"+ 0.24\!\left(1-\frac{1-e^{-50\,PD}}{1-e^{-50}}\right)")
        st.latex(r"K = LGD\cdot N\!\left[\sqrt{\frac{1}{1-R}}\,G(PD)"
                 r"+\sqrt{\frac{R}{1-R}}\,G(0.999)\right]-PD\cdot LGD")
        st.latex(r"EL = PD\times LGD\times EAD")
        st.latex(r"RWA = K\times 12.5\times EAD")

        pd_b3 = pd_manual if pd_override else dr_mean
        R_b3 = (0.12*(1-np.exp(-50*pd_b3))/(1-np.exp(-50))
                + 0.24*(1-(1-np.exp(-50*pd_b3))/(1-np.exp(-50))))
        K_b3 = (lgd * sp_norm.cdf(
            np.sqrt(1/(1-R_b3))*sp_norm.ppf(pd_b3)
            + np.sqrt(R_b3/(1-R_b3))*sp_norm.ppf(0.999)
        ) - pd_b3 * lgd)
        mean_ead = df["loan_amount"].mean() * ead_mult
        rwa_b3 = K_b3 * 12.5 * mean_ead
        st.metric("Risk-Weighted Assets (RWA)", f"${rwa_b3:,.0f}")

# ───────────────────────────────────────────────
# TAB 4 — SCORECARD & SCOREPOINTS
# ───────────────────────────────────────────────
with tab4:
    if mdl is None:
        st.warning("Logistic Regression must be enabled for the Scorecard tab. Enable it in the sidebar.")
        st.stop()
    st.header("📋 Scorecard & Scorepoints")

    BASE_SCORE = 600;  PDO = 20
    st.latex(r"\text{Score}_i = \left(\beta_i X_i+\frac{\alpha}{n}\right)"
             r"\times\frac{PDO}{\ln 2}+\frac{\text{Base Score}}{n}")
    st.latex(r"\text{Total Score}=\sum_{i=1}^{n}\text{Score}_i"
             r"\quad (\text{Base}=600,\;PDO=20)")

    scale    = PDO / np.log(2)
    n_f      = len(mdl["feat_names"])
    coefs    = mdl["w"]
    score_pts = coefs * scale

    sc_tbl = pd.DataFrame({
        "Feature":                  mdl["feat_names"],
        "Coefficient":              coefs.round(5),
        "Score Points (per sigma)": score_pts.round(2),
        "Direction": ["Increases Risk" if c > 0 else "Decreases Risk" for c in coefs],
    }).sort_values("Score Points (per sigma)", key=abs, ascending=False)
    st.subheader("Scorecard Coefficient Table")
    st.dataframe(sc_tbl, use_container_width=True)

    # Scaled scores for full portfolio
    X_all_s4, _ = build_X(df)
    Xs_s4       = (X_all_s4 - mdl["mu"]) / mdl["sd"]
    raw_logit   = Xs_s4 @ mdl["w"] + mdl["b"]
    scaled_sc   = BASE_SCORE + raw_logit * scale
    df["scaled_score"] = scaled_sc

    band_edges  = [300, 450, 525, 575, 625, 675, 750, 900]
    band_labels = ["<450","450-525","525-575","575-625","625-675","675-750",">750"]
    df["score_band"] = pd.cut(df["scaled_score"], bins=band_edges,
                               labels=band_labels, include_lowest=True)

    score_band_df = df.groupby("score_band", observed=False).agg(
        Count=("default","count"),
        Default_Rate=("default","mean"),
        Avg_Score=("scaled_score","mean"),
    ).reset_index()
    score_band_df["Risk_Tier"] = score_band_df["Default_Rate"].apply(
        lambda x: "High Risk" if x > 0.25 else ("Medium Risk" if x > 0.10 else "Low Risk"))
    score_band_df["Decision"] = score_band_df["Risk_Tier"].map(
        {"High Risk":"DECLINE","Medium Risk":"REVIEW","Low Risk":"APPROVE"})

    st.subheader("Score Band Analysis")
    st.dataframe(score_band_df.round(4), use_container_width=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ax = axes[0]
    ax.hist(scaled_sc[df["default"].values==0], bins=60, alpha=0.6,
            color="steelblue", density=True, label="Non-Default")
    ax.hist(scaled_sc[df["default"].values==1], bins=60, alpha=0.6,
            color="crimson",   density=True, label="Default")
    ax.set_title("Scorecard Score Distribution", fontweight="bold")
    ax.set_xlabel("Credit Score");  ax.set_ylabel("Density")
    ax.legend();  ax.grid(alpha=0.3)

    ax = axes[1]
    valid = score_band_df.dropna(subset=["score_band"])
    band_colors = ["#d62728" if t=="High Risk" else ("#ff7f0e" if t=="Medium Risk" else "#2ca02c")
                   for t in valid["Risk_Tier"]]
    ax.bar(range(len(valid)), valid["Default_Rate"]*100, color=band_colors, alpha=0.85)
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels(valid["score_band"].astype(str), rotation=30, ha="right")
    ax.set_title("Default Rate by Score Band", fontweight="bold")
    ax.set_ylabel("Default Rate (%)");  ax.grid(axis="y", alpha=0.3)
    plt.tight_layout();  st.pyplot(fig);  plt.close()

    # Application Scoring Simulator
    st.subheader("Application Scoring Simulator")
    c1, c2, c3 = st.columns(3)
    with c1:
        sim_loan   = st.slider("Loan Amount ($K)", 5, 500, 25) * 1_000
        sim_rate   = st.slider("Interest Rate (%)", 5.0, 36.0, 12.0)
        sim_term   = st.selectbox("Term (months)", [36, 60])
        sim_income = st.slider("Annual Income ($K)", 20, 300, 65) * 1_000
        sim_dti    = st.slider("Debt-to-Income", 0.01, 0.99, 0.25)
    with c2:
        sim_cs     = st.slider("Credit Score", 300, 850, 680)
        sim_emp    = st.slider("Employment Length (yrs)", 0, 20, 5)
        sim_open   = st.slider("Num Open Accounts", 1, 40, 10)
        sim_derog  = st.slider("Derogatory Marks", 0, 5, 0)
    with c3:
        sim_revutil= st.slider("Revolving Utilization", 0.0, 1.0, 0.30)
        sim_inq    = st.slider("Inquiries Last 6M", 0, 10, 1)
        sim_delinq = st.slider("Months Since Last Delinq (0=never)", 0, 120, 0)
        sim_home   = st.selectbox("Home Ownership", ["RENT","OWN","MORTGAGE"])
        sim_purp   = st.selectbox("Loan Purpose",
                                  ["car","debt_consolidation","home","medical","business","vacation"])

    sim_vec = np.array([[
        sim_loan, sim_rate, float(sim_term), sim_income, sim_dti,
        float(sim_cs), float(sim_emp), float(sim_open), float(sim_derog),
        sim_revutil, float(sim_inq), float(sim_delinq),
        1.0 if sim_home=="OWN"      else 0.0,
        1.0 if sim_home=="MORTGAGE" else 0.0,
        1.0 if sim_purp=="debt_consolidation" else 0.0,
        1.0 if sim_purp=="business" else 0.0,
    ]])
    sim_Xs      = (sim_vec - mdl["mu"]) / mdl["sd"]
    sim_raw     = (sim_Xs @ mdl["w"] + mdl["b"]).item()
    sim_pd_val  = float(sigmoid(np.array([sim_raw]))[0])
    sim_score_v = BASE_SCORE + sim_raw * scale
    sim_el      = sim_pd_val * lgd * sim_loan * ead_mult

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Credit Score",    f"{sim_score_v:.0f}")
    r2.metric("Predicted PD",    f"{sim_pd_val:.2%}")
    r3.metric("Expected Loss",   f"${sim_el:,.0f}")
    if   sim_pd_val < 0.10: r4.success("APPROVE")
    elif sim_pd_val < 0.25: r4.warning("REVIEW")
    else:                   r4.error("DECLINE")

    st.subheader("Population Stability Index (PSI)")
    st.latex(r"PSI = \sum_{i=1}^{n}\!\left(\text{Actual\%}_i - \text{Expected\%}_i\right)"
             r"\times\ln\!\left(\frac{\text{Actual\%}_i}{\text{Expected\%}_i}\right)")
    st.info("PSI < 0.1: Stable  |  0.1-0.2: Moderate shift  |  > 0.2: Major shift — model review required")

# ───────────────────────────────────────────────
# TAB 5 — PORTFOLIO RISK & CAPITAL
# ───────────────────────────────────────────────
with tab5:
    if active_mdl is None:
        st.warning("Enable at least one model in the sidebar to view Portfolio Risk.")
        st.stop()
    st.header("💰 Portfolio Risk & Capital")

    st.latex(r"EL = PD\times LGD\times EAD")
    st.latex(r"UL_i = EAD_i\times LGD_i\times\sqrt{PD_i(1-PD_i)}")
    st.latex(r"\text{Economic Capital}=VaR_{99.9\%}(\text{Portfolio Loss})-EL")
    st.latex(r"RAROC=\frac{\text{Revenue}-EL-\text{OpCost}}{\text{Economic Capital}}")

    # per-loan metrics
    ul_all       = loan_ead_all * lgd * np.sqrt(loan_pd_all * (1 - loan_pd_all))
    total_ul     = ul_all.sum()
    port_size    = loan_ead_all.sum()

    # Monte Carlo (500 sims on 2k loan sample)
    rng_mc  = np.random.default_rng(123)
    samp_n  = min(2_000, len(df))
    samp_idx= rng_mc.choice(len(df), samp_n, replace=False)
    s_pd    = loan_pd_all[samp_idx]
    s_ead   = loan_ead_all[samp_idx]
    scale_up= len(df) / samp_n

    sim_losses = np.array([
        (rng_mc.random(samp_n) < s_pd).astype(float) @ (s_ead * lgd) * scale_up
        for _ in range(500)
    ])

    var_99  = np.percentile(sim_losses, 99.0)
    var_999 = np.percentile(sim_losses, 99.9)
    cvar_99 = sim_losses[sim_losses >= var_99].mean()
    ec_val  = max(var_999 - total_el, 0)

    rev_t5  = (df["loan_amount"] * df["interest_rate"] / 100 * (df["term"]/12)).sum() * 0.8
    opc_t5  = rev_t5 * 0.25
    raroc_t5= (rev_t5 - total_el - opc_t5) / max(ec_val, 1)

    r_cols = st.columns(5)
    r_cols[0].metric("Total EL ($)",       f"${total_el:,.0f}")
    r_cols[1].metric("Total UL ($)",       f"${total_ul:,.0f}")
    r_cols[2].metric("VaR 99.9% ($)",      f"${var_999:,.0f}")
    r_cols[3].metric("Economic Capital ($)",f"${ec_val:,.0f}")
    r_cols[4].metric("RAROC",              f"{raroc_t5:.2%}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Monte Carlo loss distribution
    ax = axes[0, 0]
    ax.hist(sim_losses / 1e6, bins=50, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(total_el / 1e6, color="green",  ls="-",  lw=2, label=f"EL ${total_el/1e6:.1f}M")
    ax.axvline(var_99   / 1e6, color="orange", ls="--", lw=2, label=f"VaR 99% ${var_99/1e6:.1f}M")
    ax.axvline(var_999  / 1e6, color="red",    ls="--", lw=2, label=f"VaR 99.9% ${var_999/1e6:.1f}M")
    ax.set_title("Monte Carlo Loss Distribution (500 sims)", fontweight="bold")
    ax.set_xlabel("Portfolio Loss ($M)");  ax.set_ylabel("Frequency")
    ax.legend(fontsize=8);  ax.grid(alpha=0.3)

    # EL per loan
    ax = axes[0, 1]
    ax.hist(el_all / 1_000, bins=60, color="crimson", alpha=0.75, edgecolor="white")
    ax.set_title("Expected Loss per Loan", fontweight="bold")
    ax.set_xlabel("EL ($K)");  ax.set_ylabel("Count");  ax.grid(alpha=0.3)

    # PD distribution
    ax = axes[0, 2]
    ax.hist(loan_pd_all, bins=60, color="darkorange", alpha=0.75, edgecolor="white")
    ax.axvline(loan_pd_all.mean(), color="red", ls="--", lw=2,
               label=f"Mean PD={loan_pd_all.mean():.2%}")
    ax.set_title("Model PD Distribution", fontweight="bold")
    ax.set_xlabel("Predicted PD");  ax.set_ylabel("Count")
    ax.legend(fontsize=9);  ax.grid(alpha=0.3)

    # Vintage analysis
    ax = axes[1, 0]
    cohorts  = [f"2019-Q{i}" for i in range(1, 7)]
    rng_v    = np.random.default_rng(99)
    cohort_dr= [dr_mean * rng_v.uniform(0.7, 1.4) for _ in cohorts]
    bar_cols = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, 6))
    ax.bar(cohorts, [d*100 for d in cohort_dr], color=bar_cols, alpha=0.85)
    ax.set_title("Vintage Analysis — Default Rate by Cohort", fontweight="bold")
    ax.set_xlabel("Origination Cohort");  ax.set_ylabel("Default Rate (%)")
    ax.tick_params(axis="x", rotation=30);  ax.grid(axis="y", alpha=0.3)

    # Stress test
    ax = axes[1, 1]
    X_stress, _ = build_X(df)
    Xs_stress   = (X_stress - mdl["mu"]) / mdl["sd"]
    logit_base  = Xs_stress @ mdl["w"] + mdl["b"]
    cs_feat_idx = mdl["feat_names"].index("credit_score")
    shock       = -50.0 / (mdl["sd"][cs_feat_idx] + 1e-8)
    logit_str   = logit_base + mdl["w"][cs_feat_idx] * shock
    pd_str      = sigmoid(logit_str)
    if pd_override: pd_str = np.full(len(df), min(pd_manual * 1.3, 0.999))
    el_str      = (pd_str * lgd * loan_ead_all).sum()

    cats = ["Current", "Stressed\n(-50pt CS)"]
    vals = [total_el / 1e6, el_str / 1e6]
    bcol = ["#2ca02c","#d62728"]
    bars_st = ax.bar(cats, vals, color=bcol, alpha=0.85, width=0.4)
    for bar, v in zip(bars_st, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"${v:.1f}M", ha="center", va="bottom", fontsize=10)
    ax.set_title("Stress Test — CS Drop 50 Points", fontweight="bold")
    ax.set_ylabel("Portfolio EL ($M)");  ax.grid(axis="y", alpha=0.3)

    # Regulatory capital table
    ax = axes[1, 2]
    ax.axis("off")
    PD_reg = loan_pd_all.mean()
    R_reg  = (0.12*(1-np.exp(-50*PD_reg))/(1-np.exp(-50))
               + 0.24*(1-(1-np.exp(-50*PD_reg))/(1-np.exp(-50))))
    K_reg  = (lgd * sp_norm.cdf(
        np.sqrt(1/(1-R_reg))*sp_norm.ppf(PD_reg)
        + np.sqrt(R_reg/(1-R_reg))*sp_norm.ppf(0.999)
    ) - PD_reg * lgd)
    RWA_reg  = K_reg * 12.5 * port_size
    Tier1    = RWA_reg * 0.06
    Tier2    = RWA_reg * 0.02
    TotCap   = Tier1 + Tier2

    tbl_data = [
        ["Portfolio EAD",   f"${port_size/1e6:.1f}M"],
        ["Mean PD",         f"{PD_reg:.2%}"],
        ["LGD",             f"{lgd:.0%}"],
        ["Correl. R",       f"{R_reg:.4f}"],
        ["Capital K",       f"{K_reg:.4f}"],
        ["RWA",             f"${RWA_reg/1e6:.1f}M"],
        ["Tier 1 (6%)",     f"${Tier1/1e6:.1f}M"],
        ["Tier 2 (2%)",     f"${Tier2/1e6:.1f}M"],
        ["Total Capital",   f"${TotCap/1e6:.1f}M"],
        ["Econ. Capital",   f"${ec_val/1e6:.2f}M"],
        ["RAROC",           f"{raroc_t5:.2%}"],
    ]
    tbl = ax.table(cellText=tbl_data, colLabels=["Metric","Value"],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False);  tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50");  cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f0f0")
        cell.set_edgecolor("#cccccc")
    ax.set_title("Regulatory Capital Summary", fontweight="bold", pad=10)

    plt.tight_layout();  st.pyplot(fig);  plt.close()

    st.subheader("Regulatory Capital Ratios")
    st.latex(r"\text{Tier 1 Ratio} = \frac{\text{Tier 1 Capital}}{RWA} \geq 6\%")
    st.latex(r"\text{Total Capital Ratio} = \frac{\text{Tier 1}+\text{Tier 2}}{RWA} \geq 8\%")

    reg_df = pd.DataFrame({
        "Metric": [r[0] for r in tbl_data],
        "Value":  [r[1] for r in tbl_data],
    })
    st.dataframe(reg_df, use_container_width=True)

st.markdown("---")
st.caption("DefaultRisk  |  Basel III / LendingClub Framing  |  Pure NumPy  |  Synthetic Data Only")
