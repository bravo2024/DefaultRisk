"""DefaultRisk — Credit Card Default Prediction (Indian Context).
   Self-contained Streamlit app for Streamlit Cloud deployment."""

import sys, os, json, warnings, io, base64
from pathlib import Path
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    precision_recall_curve, roc_curve, brier_score_loss
)
from scipy.stats import chi2

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(
    page_title="DefaultRisk — Credit Card Default Prediction (India)",
    page_icon="\U0001f3e6",
    layout="wide",
    initial_sidebar_state="expanded",
)

defaults = {
    "data_loaded": False,
    "dataset_name": "UCI Credit Card (Taiwan)",
    "dataset_error": None,
    "df": None,
    "results": None,
    "y_test": None,
    "X_test": None,
    "models": None,
    "cv_results": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown("""
<style>
    .main-header { font-size: 2.4rem; font-weight: 700; color: #1e293b; margin-bottom: 4px; }
    .sub-header { font-size: 1.1rem; color: #64748b; margin-bottom: 16px; }
    .section-title { font-size: 1.5rem; font-weight: 600; color: #0f172a;
                     border-bottom: 3px solid #1a56db; padding-bottom: 6px; margin: 24px 0 16px 0; }
    .metric-card { background: #f1f5f9; border-radius: 12px; padding: 20px;
                   border-left: 5px solid #1a56db; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
    .highlight { background: #dbeafe; padding: 2px 6px; border-radius: 4px; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { height: auto; padding: 8px 18px; font-weight: 500; }
    hr { margin: 20px 0; }
    .footer { text-align: center; color: #94a3b8; font-size: .85rem; padding-top: 30px; }
</style>
""", unsafe_allow_html=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore")

UCI_FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
DEMOGRAPHIC_FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
PAYMENT_HISTORY_FEATURES = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_FEATURES = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
PAYMENT_AMT_FEATURES = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

SEX_MAP = {1: "Male", 2: "Female"}
EDU_MAP = {1: "Graduate", 2: "University", 3: "High School", 4: "Others", 5: "Unknown", 6: "Unknown"}
MARRIAGE_MAP = {1: "Married", 2: "Single", 3: "Others"}
PAY_STATUS_MAP = {-2: "No consumption", -1: "Paid in full", 0: "Revolving", 1: "1 month delay",
                  2: "2 month delay", 3: "3 month delay", 4: "4 month delay", 5: "5+ month delay",
                  6: "6+ month delay", 7: "7+ month delay", 8: "8+ month delay", 9: "9+ month delay"}

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    "UCI Credit Card (Taiwan)": {
        "loader": "load_uci_default", "source": "UCI ML Repository (ID 350)",
        "country": "Taiwan", "year": 2005, "samples": 30000, "features": 23,
        "target_col": "Class", "default_rate": 0.221,
        "type": "Credit card default",
        "fetch_code": "fetch_ucirepo(id=350)",
        "url": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
        "desc": (
            "Default of Credit Card Clients dataset. Predicts whether a credit card holder "
            "will default on their payment next month. Features include demographic info "
            "(credit limit, gender, education, marital status, age) and 6-month payment history "
            "(repayment status, bill amounts, payment amounts)."
        ),
    },
    "German Credit (Germany)": {
        "loader": "load_german_credit",
        "source": "OpenML (ID 31) / UCI (ID 144)",
        "country": "Germany", "year": 1994, "samples": 1000, "features": 20,
        "target_col": "Class", "default_rate": 0.30,
        "type": "Credit risk scoring (general, not card-specific)",
        "fetch_code": "fetch_openml(data_id=31)",
        "url": "https://www.openml.org/d/31",
        "desc": (
            "Statlog German Credit dataset. Classifies borrowers as good (low risk) or bad "
            "(high risk) credit risks based on attributes like account status, credit history, "
            "purpose, credit amount, employment status, and personal information."
        ),
    },
    "Give Me Some Credit (USA)": {
        "loader": "load_give_me_some_credit_",
        "source": "empulse package (Kaggle Credit Fusion)",
        "country": "USA", "year": 2011, "samples": 112915, "features": 10,
        "target_col": "Class", "default_rate": 0.067,
        "type": "Credit default (loan, not card-specific)",
        "fetch_code": "load_give_me_some_credit()",
        "url": "https://www.kaggle.com/c/GiveMeSomeCredit",
        "desc": (
            "Kaggle Give Me Some Credit dataset. Predicts whether a borrower will experience "
            "serious delinquency within the next 2 years. Features include revolving utilization, "
            "age, late payments (30-59, 60-89, 90+ days), debt ratio, monthly income, "
            "open credit lines, real estate loans, and dependents."
        ),
    },
    "PAKDD Credit Scoring (Brazil)": {
        "loader": "load_pakdd_credit",
        "source": "empulse package (PAKDD 2009)",
        "country": "Brazil", "year": 2009, "samples": 38938, "features": 25,
        "target_col": "Class", "default_rate": 0.20,
        "type": "Credit card default (private label card)",
        "fetch_code": "load_credit_scoring_pakdd()",
        "url": "https://pakdd.org/archive/pakdd2009/",
        "desc": (
            "PAKDD 2009 Credit Scoring dataset. Predicts default on a private-label credit card "
            "of a major Brazilian retailer. Features include demographic, financial, and "
            "behavioral variables from 2003-2008."
        ),
    },
    "Credit Approval (Confidential)": {
        "loader": "load_credit_approval",
        "source": "UCI ML Repository (ID 27)",
        "country": "Confidential", "year": 1992, "samples": 690, "features": 15,
        "target_col": "Class", "default_rate": 0.44,
        "type": "Credit card application approval (not default)",
        "fetch_code": "fetch_ucirepo(id=27)",
        "url": "https://archive.ics.uci.edu/dataset/27/credit+approval",
        "desc": (
            "Credit Approval dataset. Classifies credit card applications as approved or rejected. "
            "All attribute names anonymized (A1-A15). Mix of continuous, categorical, and "
            "missing values. Note: this is about application approval, not default prediction."
        ),
    },
    "Australian Credit (Australia)": {
        "loader": "load_australian_credit",
        "source": "UCI ML Repository (ID 143)",
        "country": "Australia", "year": 1990, "samples": 690, "features": 14,
        "target_col": "Class", "default_rate": 0.44,
        "type": "Credit card application approval (not default)",
        "fetch_code": "fetch_ucirepo(id=143)",
        "url": "https://archive.ics.uci.edu/dataset/143/statlog+australian+credit+approval",
        "desc": (
            "Statlog Australian Credit Approval dataset. 6 numerical + 8 categorical attributes, "
            "all anonymized. Part of the European StatLog project. "
            "Note: this is about application approval, not default prediction."
        ),
    },
}

# ---------------------------------------------------------------------------
# Dataset loaders  (all imports done at call time to avoid stale flags)
# ---------------------------------------------------------------------------
def load_uci_default():
    from sklearn.datasets import fetch_openml
    try:
        data = fetch_openml(data_id=42477, as_frame=True, parser="auto")
        df = data.frame
        df.columns = [str(c).strip().upper() for c in df.columns]
        tc = [c for c in df.columns if "DEFAULT" in c.upper()]
        if tc:
            df = df.rename(columns={tc[0]: "Class"})
        elif "CLASS" in df.columns:
            df = df.rename(columns={"CLASS": "Class"})
        elif "Y" in df.columns:
            df = df.rename(columns={"Y": "Class"})
        df["Class"] = df["Class"].astype(int)
        return df
    except Exception:
        pass
    try:
        from ucimlrepo import fetch_ucirepo
        data = fetch_ucirepo(id=350)
        df = data.data.features.copy()
        target = data.data.targets.copy()
        df["Class"] = target.values.astype(int).ravel()
        df.columns = [str(c).strip().upper() for c in df.columns]
        if "CLASS" in df.columns:
            df = df.rename(columns={"CLASS": "Class"})
        return df
    except Exception:
        pass
    csv_paths = ["default_of_credit_card_clients.csv", "UCI_Credit_Card.csv",
                 "data/default_of_credit_card_clients.csv"]
    for p in csv_paths:
        if Path(p).exists():
            df = pd.read_csv(p)
            df.columns = [str(c).strip().upper() for c in df.columns]
            tc = [c for c in df.columns if "DEFAULT" in c]
            if tc:
                df = df.rename(columns={tc[0]: "Class"})
            elif "Y" in df.columns:
                df = df.rename(columns={"Y": "Class"})
            df["Class"] = df["Class"].astype(int)
            return df
    return None

def load_german_credit():
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml(data_id=31, return_X_y=True, as_frame=True, parser="auto")
    df = X.copy()
    y_str = y.astype(str).str.strip()
    mapping = {"1": 0, "2": 1, "good": 0, "bad": 1}
    df["Class"] = y_str.map(mapping).astype(int)
    return df

def load_give_me_some_credit_():
    from empulse.datasets import load_give_me_some_credit as _gmsc
    dataset = _gmsc()
    df = dataset.data.copy()
    df["Class"] = dataset.target.astype(int)
    return df

def load_pakdd_credit():
    from empulse.datasets import load_credit_scoring_pakdd as _pakdd
    dataset = _pakdd()
    df = dataset.data.copy()
    df["Class"] = dataset.target.astype(int)
    return df

def load_credit_approval():
    from ucimlrepo import fetch_ucirepo
    data = fetch_ucirepo(id=27)
    df = data.data.features.copy()
    target = data.data.targets.copy()
    tc = target.columns[0]
    df["Class"] = target[tc].map({"+": 0, "-": 1}).astype(int)
    return df

def load_australian_credit():
    from ucimlrepo import fetch_ucirepo
    data = fetch_ucirepo(id=143)
    df = data.data.features.copy()
    target = data.data.targets.copy()
    tc = target.columns[0]
    df["Class"] = target[tc].map({1: 0, 2: 1}).astype(int)
    return df

LOADER_MAP = {
    "load_uci_default": load_uci_default,
    "load_german_credit": load_german_credit,
    "load_give_me_some_credit_": load_give_me_some_credit_,
    "load_pakdd_credit": load_pakdd_credit,
    "load_credit_approval": load_credit_approval,
    "load_australian_credit": load_australian_credit,
}


def load_data(dataset_name):
    """Load dataset by name from the registry."""
    if dataset_name not in DATASET_REGISTRY:
        return None
    loader_name = DATASET_REGISTRY[dataset_name]["loader"]
    fn = LOADER_MAP.get(loader_name)
    if fn is None:
        return None
    try:
        df = fn()
        if df is None:
            return None
        for c in df.columns:
            if c != "Class":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["Class"] = df["Class"].astype(int)
        return df
    except Exception as e:
        st.warning(f"Failed to load {dataset_name}: {e}")
        return None


def make_synthetic_data():
    """Generate synthetic credit data as final fallback."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = 5000
    df = pd.DataFrame()
    df["LIMIT_BAL"] = rng.integers(10000, 1000000, n)
    df["SEX"] = rng.integers(1, 3, n)
    df["EDUCATION"] = rng.integers(1, 4, n)
    df["MARRIAGE"] = rng.integers(1, 3, n)
    df["AGE"] = rng.integers(21, 65, n)
    for p in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        df[p] = rng.integers(-2, 8, n)
    for b in ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]:
        df[b] = rng.integers(0, 100000, n)
    for p in ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]:
        df[p] = rng.integers(0, 50000, n)
    score = (
        0.3 * (df["LIMIT_BAL"] / 100000)
        - 0.5 * (df["PAY_0"] > 2).astype(int)
        - 0.4 * (df["PAY_2"] > 2).astype(int)
        - 0.2 * (df["PAY_AMT1"] / 10000)
        + 0.1 * (df["AGE"] / 10)
        + rng.normal(0, 1, n)
    )
    prob = 1 / (1 + np.exp(-score))
    df["Class"] = (rng.random(n) < prob).astype(int)
    return df


class CreditFeatures(BaseEstimator, TransformerMixin):
    """Encode categorical features and scale numeric ones. Fit on training data only."""

    def __init__(self):
        self.encoders_ = {}
        self.scaler_ = StandardScaler()

    def fit(self, X, y=None):
        self.cat_cols = [c for c in X.columns if X[c].nunique() <= 10 and c != "Class"] or []
        if not self.cat_cols:
            self.cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
            self.cat_cols = [c for c in self.cat_cols if c in X.columns]
        self.num_cols = [c for c in X.columns if c not in self.cat_cols and c != "Class"]
        for c in self.cat_cols:
            if c in X.columns:
                le = LabelEncoder()
                le.fit(X[c].astype(str))
                self.encoders_[c] = le
        num_data = X[[c for c in self.num_cols if c in X.columns]].values.astype(float)
        if num_data.shape[1] > 0:
            self.scaler_.fit(num_data)
        return self

    def transform(self, X):
        parts = []
        num_cols = [c for c in self.num_cols if c in X.columns]
        if num_cols:
            num_scaled = self.scaler_.transform(X[num_cols].values.astype(float))
            parts.append(pd.DataFrame(num_scaled, columns=num_cols, index=X.index))
        for c in self.cat_cols:
            if c in X.columns and c in self.encoders_:
                encoded = self.encoders_[c].transform(X[c].astype(str))
                parts.append(pd.DataFrame(encoded, columns=[c], index=X.index))
        if not parts:
            return X
        return pd.concat(parts, axis=1)


def train_and_evaluate(_df):
    """Run full benchmark: LR, RF, LGBM. Return results dict."""
    X_all = _df.drop(columns=["Class"])
    y_all = _df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.25, stratify=y_all, random_state=RANDOM_STATE
    )

    results = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
    models = {}

    lr_pipe = Pipeline([
        ("features", CreditFeatures()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
    lr_pred = (lr_proba >= 0.5).astype(int)
    models["Logistic Regression"] = {
        "pipeline": lr_pipe, "proba": lr_proba, "pred": lr_pred,
        "pr_auc": average_precision_score(y_test, lr_proba),
        "roc_auc": roc_auc_score(y_test, lr_proba),
    }

    rf_pipe = Pipeline([
        ("features", CreditFeatures()),
        ("clf", RandomForestClassifier(
            class_weight="balanced", n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
        )),
    ])
    rf_pipe.fit(X_train, y_train)
    rf_proba = rf_pipe.predict_proba(X_test)[:, 1]
    rf_pred = (rf_proba >= 0.5).astype(int)
    models["Random Forest"] = {
        "pipeline": rf_pipe, "proba": rf_proba, "pred": rf_pred,
        "pr_auc": average_precision_score(y_test, rf_proba),
        "roc_auc": roc_auc_score(y_test, rf_proba),
    }

    if HAS_LGB:
        lgb_base = Pipeline([
            ("features", CreditFeatures()),
            ("clf", lgb.LGBMClassifier(
                class_weight="balanced", n_estimators=300, learning_rate=0.05,
                num_leaves=31, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
            )),
        ])
        lgb_calib = CalibratedClassifierCV(estimator=lgb_base, method="isotonic", cv=3)
        lgb_calib.fit(X_train, y_train)
        lgb_proba = lgb_calib.predict_proba(X_test)[:, 1]
        lgb_pred = (lgb_proba >= 0.5).astype(int)
        models["LightGBM (Calibrated)"] = {
            "pipeline": lgb_calib, "proba": lgb_proba, "pred": lgb_pred,
            "pr_auc": average_precision_score(y_test, lgb_proba),
            "roc_auc": roc_auc_score(y_test, lgb_proba),
        }
    else:
        models["LightGBM (Calibrated)"] = {
            "pipeline": None, "proba": None, "pred": None, "pr_auc": 0, "roc_auc": 0,
        }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    for name, pipe_builder, model_cls in [
        ("Logistic Regression",
         lambda: LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
         None),
        ("Random Forest",
         lambda: RandomForestClassifier(class_weight="balanced", n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
         None),
    ]:
        fold_scores = []
        for tr, va in skf.split(X_train, y_train):
            p = Pipeline([
                ("features", CreditFeatures()),
                ("clf", model_cls() if model_cls else pipe_builder()),
            ])
            p.fit(X_train.iloc[tr], y_train.iloc[tr])
            proba = p.predict_proba(X_train.iloc[va])[:, 1]
            fold_scores.append(average_precision_score(y_train.iloc[va], proba))
        cv_results[name] = {"mean": float(np.mean(fold_scores)), "std": float(np.std(fold_scores)),
                           "folds": [float(s) for s in fold_scores]}

    if HAS_LGB:
        lgb_folds = []
        for tr, va in skf.split(X_train, y_train):
            p = Pipeline([
                ("features", CreditFeatures()),
                ("clf", lgb.LGBMClassifier(
                    class_weight="balanced", n_estimators=300, learning_rate=0.05,
                    num_leaves=31, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
                )),
            ])
            p.fit(X_train.iloc[tr], y_train.iloc[tr])
            proba = p.predict_proba(X_train.iloc[va])[:, 1]
            lgb_folds.append(average_precision_score(y_train.iloc[va], proba))
        cv_results["LightGBM"] = {"mean": float(np.mean(lgb_folds)), "std": float(np.std(lgb_folds)),
                                  "folds": [float(s) for s in lgb_folds]}

    results["models"] = models
    results["cv_results"] = cv_results
    fe_demo = CreditFeatures().fit(X_train)
    demo_transformed = fe_demo.transform(X_train.head())
    results["feature_names"] = list(demo_transformed.columns)
    return results


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


_tab_descriptions = {
    "Overview": "Business context for credit card default prediction in the Indian banking ecosystem. Covers CIBIL/credit bureau data, RBI regulatory framework, and NPA classification.",
    "Data Explorer": "Analysis of the UCI Default of Credit Card Clients dataset adapted for Indian context — demographics, payment history, bill amounts, and default distributions.",
    "Feature Engineering": "Encoding pipeline for mixed data types fitted on training data only to prevent leakage. Label encoding for categoricals, standard scaling for numerics.",
    "Model Benchmarks": "Three classifiers compared using 5-fold cross-validation, hold-out test metrics, confusion matrices, and McNemar's statistical significance test.",
    "Scorecard": "Credit score distribution, approval rates by threshold, gain/lift charts for operational queue management, and cost-sensitive threshold optimization.",
    "Explainability": "SHAP-based feature attribution showing which borrower characteristics drive default probability.",
    "Model Card & About": "Model documentation covering intended use, limitations, monitoring plan, technology stack, and project context.",
}


# ── Sidebar ──────────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    "<h1 style='font-size:1.6rem; margin-bottom:0;'>\U0001f3e6 DefaultRisk</h1>",
    unsafe_allow_html=True,
)
st.sidebar.caption("Credit default PD model  \u00b7  Indian Banking Context")
st.sidebar.markdown("---")

ds_options = list(DATASET_REGISTRY.keys())
default_idx = ds_options.index(st.session_state.dataset_name) if st.session_state.dataset_name in ds_options else 0
selected_ds = st.sidebar.selectbox("Dataset", ds_options, index=default_idx, key="ds_selector")

if selected_ds != st.session_state.dataset_name:
    for k in ["df", "results", "y_test", "X_test", "models", "cv_results"]:
        st.session_state[k] = None
    st.session_state.data_loaded = False
    st.session_state.dataset_error = None
    st.session_state.dataset_name = selected_ds
    st.session_state._load_attempted = False
    st.rerun()

meta = DATASET_REGISTRY.get(st.session_state.dataset_name)
if meta:
    ds_url = meta.get("url", "")
    url_md = f"[Source]({ds_url})" if ds_url else ""
    st.sidebar.info(
        f"**{st.session_state.dataset_name}**  \n"
        f"{meta['source']}  \n"
        f"{meta['country']} \u00b7 {meta['samples']:,} rows \u00b7 "
        f"{meta['features']} features  \n"
        f"Default rate: {meta['default_rate']:.0%}  \n"
        f"Type: {meta['type']}  \n"
        f"{url_md}"
    )

with st.sidebar.expander("Available Datasets"):
    for name, m in DATASET_REGISTRY.items():
        tick = "\u2705" if name == st.session_state.dataset_name else "\u2022"
        u = m.get("url", "")
        url_line = f"\n[Source link]({u})" if u else ""
        st.markdown(f"**{tick} {name}**  ")
        st.caption(
            f"{m['country']} \u00b7 {m['samples']:,} rows \u00b7 {m['features']} features  \n"
            f"{m['type']}  \n"
            f"Fetch: `{m['fetch_code']}`{url_line}"
        )
        st.markdown("---")

with st.sidebar.expander("Upload Custom Data"):
    uploaded = st.file_uploader("CSV file", type=["csv"], key="custom_csv")
    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
            if "Class" not in df_upload.columns:
                possible = [c for c in df_upload.columns if "default" in c.lower() or "target" in c.lower() or "class" in c.lower()]
                if possible:
                    df_upload = df_upload.rename(columns={possible[0]: "Class"})
                else:
                    st.error("CSV must contain a 'Class' column (or default/target/class).")
                    df_upload = None
            if df_upload is not None:
                df_upload["Class"] = df_upload["Class"].astype(int)
                st.session_state.df = df_upload
                st.session_state.dataset_name = "Custom Upload"
                st.session_state.data_loaded = False
                st.rerun()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    url = st.text_input("URL (raw CSV)", placeholder="https://example.com/data.csv")
    if st.button("Load from URL", key="load_url"):
        if url:
            try:
                df_url = pd.read_csv(url)
                if "Class" not in df_url.columns:
                    possible = [c for c in df_url.columns if "default" in c.lower() or "target" in c.lower() or "class" in c.lower()]
                    if possible:
                        df_url = df_url.rename(columns={possible[0]: "Class"})
                    else:
                        st.error("CSV must contain a 'Class' column.")
                        df_url = None
                if df_url is not None:
                    df_url["Class"] = df_url["Class"].astype(int)
                    st.session_state.df = df_url
                    st.session_state.dataset_name = "URL Import"
                    st.session_state.data_loaded = False
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading URL: {e}")

tabs = [
    "Overview",
    "Data Explorer",
    "Feature Engineering",
    "Model Benchmarks",
    "Scorecard",
    "Explainability",
    "Model Card & About",
]
active_tab = st.sidebar.radio("Go to", tabs, index=(
    tabs.index(st.session_state.get("_last_tab", tabs[0]))
    if st.session_state.get("_last_tab") in tabs else 0
), key="tab_nav")
st.session_state._last_tab = active_tab

st.sidebar.markdown("---")


def auto_load_data():
    """Load default dataset and train models. Called once on startup."""
    if st.session_state.get("_load_attempted", False):
        st.error(
            f"Failed to load {st.session_state.dataset_name}. "
            "Try selecting a different dataset from the sidebar, "
            "or upload a custom CSV file."
        )
        st.session_state.data_loaded = True
        st.session_state.df = make_synthetic_data()
        return
    st.session_state._load_attempted = True
    ds_name = st.session_state.dataset_name
    with st.status(f"Loading {ds_name}...", expanded=True) as status:
        df = load_data(ds_name)
        if df is not None:
            status.update(label=f"{ds_name} loaded ({len(df):,} rows)", state="complete")
            st.session_state.df = df
            st.session_state.dataset_error = None
        else:
            status.update(label="Dataset unavailable — using synthetic data", state="error")
            df = make_synthetic_data()
            st.session_state.df = df
            st.session_state.dataset_error = f"Could not load {ds_name}. Using synthetic demo data."

    with st.status("Training models...", expanded=True) as status:
        try:
            results = train_and_evaluate(df)
            st.session_state.results = results
            st.session_state.y_test = results["y_test"]
            st.session_state.X_test = results["X_test"]
            st.session_state.models = results["models"]
            st.session_state.cv_results = results["cv_results"]
            st.session_state.data_loaded = True
            status.update(label="Models trained", state="complete")
        except Exception as e:
            status.update(label=f"Training failed: {e}", state="error")
            st.session_state.data_loaded = True
    st.rerun()

# ── Auto-load on first visit or dataset change ────────────────────────
if not st.session_state.data_loaded:
    auto_load_data()

df = st.session_state.df
results = st.session_state.results
y_test = st.session_state.y_test
X_test = st.session_state.X_test
models = st.session_state.models
cv_results = st.session_state.cv_results

is_uci = bool(
    st.session_state.dataset_name in DATASET_REGISTRY
    and DATASET_REGISTRY[st.session_state.dataset_name]["loader"] == "load_uci_default"
    and df is not None
    and all(c in df.columns for c in ["LIMIT_BAL", "SEX", "EDUCATION"])
)

if active_tab == tabs[0]:
    st.markdown("<div class='main-header'>Overview \u2014 Credit Risk in Indian Banking</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Probability of Default (PD) estimation for credit card portfolios "
        "is a key component of credit risk management under RBI\u2019s Basel III framework.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.markdown("###  Problem Context")
        st.markdown(
            "Indian banks and NBFCs face three structural challenges in credit card default prediction:"
        )
        bullets = [
            ("**CIBIL / Credit Bureau Data**",
             "India has four major credit bureaus (CIBIL/TransUnion, Experian, Equifax, CRIF High Mark). "
             "Credit scores range from 300\u2013900. Payment history, credit utilization, and inquiry patterns are key predictors."),
            ("**RBI Regulatory Framework**",
             "Basel III implementation by RBI requires banks to estimate PD, LGD, and EAD for capital adequacy under the IRB approach. "
             "NPAs (Non-Performing Assets) are classified as 90+ days overdue per RBI guidelines."),
            ("**Asymmetric Costs**",
             "A default (FN) results in principal loss and NPA provisioning. A false positive (FP) declines a good customer, "
             "losing interest income and relationship value. The cost ratio depends on ticket size and recovery rate."),
        ]
        for title, desc in bullets:
            st.markdown(f"- {title}: {desc}")

    with col2:
        st.markdown("###  Dataset Summary")
        ds_name = st.session_state.dataset_name
        ds_meta = DATASET_REGISTRY.get(ds_name)
        defaults_count = int(df["Class"].sum())
        total = len(df)
        n_feats = len(df.columns) - 1
        source_name = ds_meta["source"] if ds_meta else "Custom upload"
        desc_text = ds_meta["desc"] if ds_meta else ""
        ds_url = ds_meta.get("url", "") if ds_meta else ""
        url_link = f'<br><a href="{ds_url}" target="_blank">View source \u2197</a>' if ds_url else ""
        st.markdown(
            f'<div class="metric-card">'
            f"<b>{ds_name}</b><br><br>"
            f" Accounts: {total:,}<br>"
            f" Defaults: {defaults_count:,} ({df['Class'].mean():.1%})<br>"
            f" Features: {n_feats}<br>"
            f" Source: {source_name}{url_link}"
            f"<br><br>{desc_text}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("###  Key Metrics in Indian Credit Risk")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Default Rate", f"{df['Class'].mean():.1%}",
                  help="Proportion of accounts that defaulted on payment")
    with c2:
        st.metric("Accounts", f"{total:,}",
                  help="Total observations in dataset")
    with c3:
        st.metric("Features", f"{n_feats}",
                  help="Number of feature columns in the dataset")
    with c4:
        st.metric("Regulatory Standard", "RBI Basel III",
                  help="IRB approach for PD estimation")

    st.markdown("###  Techniques Used")
    techs = pd.DataFrame({
        "Technique": [
            "PR-AUC over ROC-AUC",
            "Stratified cross-validation",
            "Probability calibration",
            "Class weighting",
            "Leak-free feature pipeline",
            "SHAP explainability",
            "Gain/Lift charts",
            "Cost-sensitive threshold",
        ],
        "Why": [
            "Honest metric under class imbalance",
            "Accounts for limited sample size",
            "Produces unbiased PD estimates",
            "Addresses asymmetric class distribution",
            "Prevents data leakage in encoding",
            "Regulatory compliance for adverse-action notices",
            "Evaluates ranking performance",
            "Aligns model with P&L impact",
        ],
    })
    st.dataframe(techs, use_container_width=True, hide_index=True)


elif active_tab == tabs[1]:
    if not st.session_state.data_loaded:
        st.error("Data loading failed. Select a different dataset from the sidebar.")
        st.stop()

    ds_name = st.session_state.dataset_name
    st.markdown(f"<div class='main-header'>Data Explorer \u2014 {ds_name}</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Target distribution, feature analysis, and correlations.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if is_uci:
        eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
            "Default Distribution", "Demographics", "Payment History", "Correlations"
        ])
    else:
        eda_tab1, eda_tab2 = st.tabs([
            "Default Distribution", "Feature Analysis"
        ])

    with eda_tab1:
        st.markdown("#### Target Distribution \u2014 Default vs Non-Default")
        col_chart, col_metrics = st.columns([2, 1])
        with col_chart:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            counts = df["Class"].value_counts()
            labels_map = {0: "Non-Default (0)", 1: "Default (1)"}
            bars = ax.bar([labels_map.get(i, str(i)) for i in counts.index], counts.values,
                          color=["#1a56db", "#ef4444"], width=0.5, edgecolor="white")
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                        f"{val:,} ({val/len(df):.1%})", ha="center", va="bottom",
                        fontsize=11, fontweight="bold")
            ax.set_ylabel("Count")
            ax.set_title(f"Target Distribution ({ds_name})", fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig)

        with col_metrics:
            st.markdown(
                f'<div class="metric-card">'
                f"<b>Key Stats</b><br><br>"
                f"Non-Default: {counts.get(0, 0):,}<br>"
                f"Default:  {counts.get(1, 0):,}<br>"
                f"Default Rate: {df['Class'].mean():.1%}<br>"
                f"Total: {len(df):,}<br>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"PR-AUC evaluates ranking performance on the default class "
                f"(baseline accuracy: {max(df['Class'].mean(), 1-df['Class'].mean()):.0%})."
            )

    if is_uci:
        with eda_tab2:
            st.markdown("#### Demographic Analysis")
            demo_choice = st.selectbox("Select demographic feature",
                                       ["SEX", "EDUCATION", "MARRIAGE", "AGE", "LIMIT_BAL"])

            if demo_choice == "SEX":
                df_plot = df.copy()
                df_plot["SEX"] = df_plot["SEX"].map(SEX_MAP).fillna("Unknown")
                fig, ax = plt.subplots(figsize=(8, 4.5))
                default_rates = df_plot.groupby("SEX")["Class"].mean()
                counts_data = df_plot["SEX"].value_counts()
                ax.bar(default_rates.index, default_rates.values,
                       color=["#1a56db", "#ef4444"], edgecolor="white", width=0.4)
                ax.set_ylabel("Default Rate")
                ax.set_title("Default Rate by Gender", fontweight="bold")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                for i, (idx, val) in enumerate(default_rates.items()):
                    ax.text(i, val + 0.005, f"{val:.1%}", ha="center", fontsize=10, fontweight="bold")
                st.pyplot(fig)
                st.caption("Sample: " + ", ".join([f"{k}: {v}" for k, v in counts_data.items()]))

            elif demo_choice == "EDUCATION":
                df_plot = df.copy()
                df_plot["EDUCATION"] = df_plot["EDUCATION"].map(EDU_MAP).fillna("Unknown")
                fig, ax = plt.subplots(figsize=(9, 4.5))
                default_rates = df_plot.groupby("EDUCATION")["Class"].mean().sort_values()
                ax.barh(default_rates.index, default_rates.values,
                        color=["#ef4444" if v > df["Class"].mean() else "#1a56db" for v in default_rates.values],
                        edgecolor="white")
                ax.axvline(df["Class"].mean(), color="black", linestyle="--", alpha=0.5,
                           label=f"Overall ({df['Class'].mean():.1%})")
                ax.set_xlabel("Default Rate")
                ax.set_title("Default Rate by Education Level", fontweight="bold")
                ax.legend()
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                st.pyplot(fig)

            elif demo_choice == "MARRIAGE":
                df_plot = df.copy()
                df_plot["MARRIAGE"] = df_plot["MARRIAGE"].map(MARRIAGE_MAP).fillna("Unknown")
                fig, ax = plt.subplots(figsize=(8, 4.5))
                default_rates = df_plot.groupby("MARRIAGE")["Class"].mean()
                ax.bar(default_rates.index, default_rates.values,
                       color=["#1a56db", "#ef4444", "#64748b"], edgecolor="white", width=0.4)
                ax.set_ylabel("Default Rate")
                ax.set_title("Default Rate by Marital Status", fontweight="bold")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                for i, (idx, val) in enumerate(default_rates.items()):
                    ax.text(i, val + 0.005, f"{val:.1%}", ha="center", fontsize=10, fontweight="bold")
                st.pyplot(fig)

            elif demo_choice == "AGE":
                fig, ax = plt.subplots(figsize=(9, 4.5))
                for label, color, name in [(0, "#1a56db", "Non-Default"), (1, "#ef4444", "Default")]:
                    sns.kdeplot(df[df["Class"] == label]["AGE"], ax=ax,
                                label=name, color=color, fill=True, alpha=0.35)
                ax.set_xlabel("Age")
                ax.set_ylabel("Density")
                ax.set_title("Age Distribution by Default Status", fontweight="bold")
                ax.legend()
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                st.pyplot(fig)
                st.caption("Younger borrowers (20\u201330) tend to have slightly higher default rates in this dataset.")

            elif demo_choice == "LIMIT_BAL":
                fig, ax = plt.subplots(figsize=(9, 4.5))
                for label, color, name in [(0, "#1a56db", "Non-Default"), (1, "#ef4444", "Default")]:
                    sns.kdeplot(df[df["Class"] == label]["LIMIT_BAL"] / 1000,
                                ax=ax, label=name, color=color, fill=True, alpha=0.35)
                ax.set_xlabel("Credit Limit (\u20b9 Thousands)")
                ax.set_ylabel("Density")
                ax.set_title("Credit Limit Distribution by Default Status", fontweight="bold")
                ax.legend()
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                st.pyplot(fig)
                st.caption("Lower credit limits are associated with higher default risk. "
                           "In the Indian context, this correlates with CIBIL score bands.")

        with eda_tab3:
            st.markdown("#### Payment History Analysis")
            st.markdown("Repayment status over 6 months (PAY_0 to PAY_6). "
                        "Negative values indicate no balance or paid in full; positive values indicate delays.")
            pay_col = st.selectbox("Select month", ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
                                   format_func=lambda x: {"PAY_0": "Month 1 (latest)", "PAY_2": "Month 2",
                                                           "PAY_3": "Month 3", "PAY_4": "Month 4",
                                                           "PAY_5": "Month 5", "PAY_6": "Month 6"}[x])
            fig, ax = plt.subplots(figsize=(10, 5))
            grouped = df.groupby(pay_col)["Class"].mean()
            counts_pay = df[pay_col].value_counts().sort_index()
            ax.bar(grouped.index.astype(str), grouped.values,
                   color=["#ef4444" if v > df["Class"].mean() else "#1a56db" for v in grouped.values],
                   edgecolor="white")
            ax.axhline(df["Class"].mean(), color="black", linestyle="--", alpha=0.5,
                       label=f"Overall ({df['Class'].mean():.1%})")
            ax.set_xlabel("Repayment Status (negative = paid, positive = delayed)")
            ax.set_ylabel("Default Rate")
            ax.set_title(f"Default Rate by {pay_col} Repayment Status", fontweight="bold")
            ax.legend()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig)

            st.markdown("#### Average Bill and Payment Amounts")
            bill_means = df[BILL_FEATURES].mean()
            pay_means = df[PAYMENT_AMT_FEATURES].mean()
            fig, ax = plt.subplots(figsize=(10, 4.5))
            x = range(len(BILL_FEATURES))
            ax.plot(x, bill_means.values / 1000, marker="o", color="#1a56db", linewidth=2, label="Avg Bill Amount")
            ax.plot(x, pay_means.values / 1000, marker="s", color="#22c55e", linewidth=2, label="Avg Payment Amount")
            ax.set_xticks(x)
            ax.set_xticklabels(["Month 1", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6"])
            ax.set_ylabel("Amount (\u20b9 Thousands)")
            ax.set_title("Average Bill and Payment Amounts Over 6 Months", fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with eda_tab4:
            st.markdown("#### Feature Correlation with Default")
            st.markdown("Pearson correlation of all features with the default target.")

            corr_data = df[UCI_FEATURES + ["Class"]].corr()["Class"].drop("Class").sort_values()

            fig, ax = plt.subplots(figsize=(9, 10))
            colors = ["#ef4444" if v < 0 else "#1a56db" for v in corr_data.values]
            ax.barh(corr_data.index, corr_data.values, color=colors, edgecolor="white")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Correlation with Default")
            ax.set_title("Feature Correlation with Default Target", fontweight="bold")
            st.pyplot(fig)

            st.caption(
                "PAY_0 (most recent repayment status) has the strongest positive correlation with default. "
                "Higher credit limits (LIMIT_BAL) show negative correlation. "
                "In Indian context, these align with CIBIL score components: payment history and credit utilization."
            )
    else:
        with eda_tab2:
            st.markdown("#### Feature Overview")
            feat_cols = [c for c in df.columns if c != "Class"]
            feat_type = st.selectbox("Select feature to visualise", feat_cols)
            fig, ax = plt.subplots(figsize=(9, 4.5))
            if df[feat_type].nunique() <= 10:
                group_means = df.groupby(feat_type)["Class"].mean()
                ax.bar(group_means.index.astype(str), group_means.values,
                       color=["#ef4444" if v > df["Class"].mean() else "#1a56db" for v in group_means.values],
                       edgecolor="white")
                ax.axhline(df["Class"].mean(), color="black", linestyle="--", alpha=0.5,
                           label=f"Overall ({df['Class'].mean():.1%})")
                ax.set_xlabel(feat_type)
                ax.set_ylabel("Default Rate")
                ax.legend()
            else:
                for label, color, name in [(0, "#1a56db", "Non-Default"), (1, "#ef4444", "Default")]:
                    sns.kdeplot(df[df["Class"] == label][feat_type], ax=ax,
                                label=name, color=color, fill=True, alpha=0.35)
                ax.set_xlabel(feat_type)
                ax.set_ylabel("Density")
            ax.set_title(f"Default Rate by {feat_type}", fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig)

            st.markdown("#### Feature Correlation with Default")
            corr_data = df[[c for c in df.columns if c != "Class"] + ["Class"]].corr()["Class"].drop("Class").sort_values()
            fig, ax = plt.subplots(figsize=(9, max(5, len(corr_data) * 0.35)))
            colors = ["#ef4444" if v < 0 else "#1a56db" for v in corr_data.values]
            ax.barh(corr_data.index, corr_data.values, color=colors, edgecolor="white")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Correlation with Default")
            ax.set_title("Feature Correlation with Default Target", fontweight="bold")
            st.pyplot(fig)


elif active_tab == tabs[2]:
    if not st.session_state.data_loaded:
        st.error("Data loading failed. Select a different dataset from the sidebar.")
        st.stop()
    st.markdown("<div class='main-header'>Feature Engineering \u2014 Leak-Free Pipeline</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Encoding and scaling implemented inside an sklearn Pipeline to prevent data leakage.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown(
        "The UCI dataset contains a mix of categorical features (gender, education, marriage) and numeric features "
        "(age, credit limit, payment amounts, bill amounts). A common error is applying encoders or scalers "
        "on the full dataset before splitting, which leaks test-set information into training."
        "\n\nFeature transformations are wrapped in a `BaseEstimator` + `TransformerMixin` class. "
        "The `.fit()` method learns encoding mappings and scaling parameters **from training data only**, "
        "and `.transform()` applies them to any input."
    )

    st.markdown("### Pipeline Architecture")
    st.markdown(
        "```\n"
        "Pipeline([\n"
        "    ('features', CreditFeatures()),  # Encode categoricals, scale numerics\n"
        "    ('clf',      LightGBM(...)),       # Gradient boosting classifier\n"
        "])\n"
        "```"
    )

    st.markdown("### Feature Set")
    feat_col1, feat_col2 = st.columns(2)
    with feat_col1:
        st.markdown("**Numeric Features (Standard Scaled)**")
        st.markdown("Credit limit, age, bill amounts (6 months), payment amounts (6 months)")
        st.markdown("**Total: 19 numeric**")
    with feat_col2:
        st.markdown("**Categorical Features (Label Encoded)**")
        st.markdown("Gender (Male/Female), Education (Graduate/University/High School/Others), "
                    "Marital Status (Married/Single/Others)")
        st.markdown("**Total: 3 categorical**")

    fe_demo = CreditFeatures().fit(df.drop(columns=["Class"]).head(5000))
    before_cols = len(df.columns) - 1
    after = fe_demo.transform(df.drop(columns=["Class"]).head(5))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Before:** {before_cols} raw features")
        st.dataframe(df.drop(columns=["Class"]).head(3), use_container_width=True)
    with c2:
        st.markdown(f"**After:** {len(after.columns)} features (same count, transformed)")
        st.dataframe(after.head(3), use_container_width=True)

    st.markdown("---")
    st.markdown("**Why this matters:** Inside `cross_val_score`, each fold re-fits the encoding step "
                "on the training partition. Test-fold data does not influence encoding or scaling parameters.")


elif active_tab == tabs[3]:
    if not st.session_state.data_loaded:
        st.error("Data loading failed. Select a different dataset from the sidebar.")
        st.stop()
    st.markdown("<div class='main-header'>Model Benchmarks</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Three classifiers compared using 5-fold stratified cross-validation, "
        "hold-out test metrics, confusion matrices, and McNemar\u2019s test.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("### Cross-Validation (5-Fold Stratified)")
    st.markdown("**Primary metric: PR-AUC** \u2014 measures ranking performance on the default class.")

    cv_df = pd.DataFrame({
        name: {"PR-AUC Mean": v["mean"], "PR-AUC Std": v["std"]}
        for name, v in cv_results.items()
    }).T

    best_model = cv_df["PR-AUC Mean"].idxmax()
    cv_df_styled = cv_df.style.apply(
        lambda row: ["background: #dbeafe" if row.name == best_model else "" for _ in row],
        axis=1
    )
    st.dataframe(cv_df_styled, use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(len(cv_results))
    means = [v["mean"] for v in cv_results.values()]
    stds = [v["std"] for v in cv_results.values()]
    colors_bar = ["#22c55e" if n == best_model else "#1a56db" for n in cv_results.keys()]
    bars = ax.bar(x_pos, means, yerr=stds, color=colors_bar, capsize=6, width=0.55,
                  edgecolor="white", linewidth=1.2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(cv_results.keys()), rotation=25, ha="right")
    ax.set_ylabel("PR-AUC")
    ax.set_title("5-Fold CV: PR-AUC (mean \u00b1 std)", fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, label="Random baseline")
    ax.legend(fontsize=9)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)

    st.info(
        f"**{best_model}** achieved the highest CV PR-AUC "
        f"({cv_results[best_model]['mean']:.4f} \u00b1 {cv_results[best_model]['std']:.4f})."
    )
    st.markdown("---")

    st.markdown("### Hold-Out Test Set")
    rows = []
    for name, m in models.items():
        if m["proba"] is not None:
            yt = y_test
            pred = m["pred"]
            proba = m["proba"]
            cm = confusion_matrix(yt, pred)
            tn, fp, fn, tp = cm.ravel()
            rows.append({
                "Model": name,
                "PR-AUC": f"{average_precision_score(yt, proba):.4f}",
                "ROC-AUC": f"{roc_auc_score(yt, proba):.4f}",
                "Precision": f"{precision_score(yt, pred):.4f}",
                "Recall": f"{recall_score(yt, pred):.4f}",
                "F1": f"{f1_score(yt, pred):.4f}",
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            })
        else:
            rows.append({"Model": name, "PR-AUC": f"{m['pr_auc']:.4f}",
                         "ROC-AUC": "\u2014", "Precision": "\u2014", "Recall": "\u2014",
                         "F1": "\u2014", "TP": "\u2014", "FP": "\u2014", "FN": "\u2014", "TN": "\u2014"})

    test_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(test_df, use_container_width=True)

    st.markdown("### Confusion Matrices")
    conf_cols = st.columns(3)
    for idx, (name, m) in enumerate(models.items()):
        if m["pred"] is not None:
            cm = confusion_matrix(y_test, m["pred"])
            tn, fp, fn, tp = cm.ravel()
            with conf_cols[idx % 3]:
                fig, ax = plt.subplots(figsize=(3.5, 3.5))
                ax.imshow(cm, cmap="Blues", interpolation="nearest")
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                                fontsize=13, fontweight="bold",
                                color="white" if cm[i, j] > cm.max() / 2 else "black")
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["Pred Non-Default", "Pred Default"])
                ax.set_yticklabels(["True Non-Default", "True Default"])
                ax.set_title(f"{name}", fontweight="bold")
                st.pyplot(fig)

    st.markdown("---")
    st.markdown("### McNemar\u2019s Test for Statistical Significance")
    st.markdown(
        "McNemar\u2019s test evaluates whether two classifiers have significantly different error distributions "
        "using discordant pairs where one model is correct and the other is wrong."
    )

    lr_name = "Logistic Regression"
    best_model_name = [n for n in models.keys() if models[n]["pred"] is not None]
    best_model_name = max(best_model_name, key=lambda n: models[n]["pr_auc"])

    if lr_name in models and models[lr_name]["pred"] is not None:
        lr_pred = models[lr_name]["pred"]
        best_pred = models[best_model_name]["pred"]

        both_correct = ((lr_pred == y_test) & (best_pred == y_test)).sum()
        lr_only = ((lr_pred == y_test) & (best_pred != y_test)).sum()
        best_only = ((lr_pred != y_test) & (best_pred == y_test)).sum()
        both_wrong = ((lr_pred != y_test) & (best_pred != y_test)).sum()

        b, c = lr_only, best_only
        stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
        pval = 1 - chi2.cdf(stat, df=1)

        mcnemar_df = pd.DataFrame(
            [[both_correct, best_only], [lr_only, both_wrong]],
            index=[f"{best_model_name} correct", f"{best_model_name} wrong"],
            columns=["LR correct", "LR wrong"],
        )
        st.dataframe(mcnemar_df, use_container_width=True)
        st.markdown(f"**McNemar statistic:** {stat:.2f}  |  **p-value:** {pval:.2e}")
        if pval < 0.05:
            winner = best_model_name if best_only > lr_only else lr_name
            st.success(
                f"Reject H\u2080 at \u03b1=0.05. **{winner}** is statistically significantly better "
                f"({best_only} vs {lr_only} discordant pairs)."
            )
        else:
            st.info("Fail to reject H\u2080 \u2014 classifiers are statistically indistinguishable.")


elif active_tab == tabs[4]:
    if not st.session_state.data_loaded:
        st.error("Data loading failed. Select a different dataset from the sidebar.")
        st.stop()
    st.markdown("<div class='main-header'>Scorecard & Decision Threshold</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Credit score distribution, approval rates by threshold, "
        "gain/lift charts, and cost-sensitive threshold optimization.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    best_name = max(models, key=lambda n: (models[n]["pr_auc"] if models[n]["proba"] is not None else 0))
    best_proba = models[best_name]["proba"]

    if best_proba is not None:
        st.info(f"Using **{best_name}** (best PR-AUC).")

        st.markdown("### Score Distribution by Default Status")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for label, color, name in [(0, "#1a56db", "Non-Default"), (1, "#ef4444", "Default")]:
            sns.kdeplot(best_proba[y_test.values == label], ax=ax,
                        label=name, color=color, fill=True, alpha=0.35)
        ax.set_xlabel("Predicted Default Probability")
        ax.set_ylabel("Density")
        ax.set_title("Score Distribution by Actual Default Status", fontweight="bold")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)

        st.markdown("### Approval Rate vs Default Rate by Threshold")
        thresholds = np.linspace(0.01, 0.99, 100)
        approval_rates = []
        default_rates_at_thresh = []
        fns = []
        fps = []
        for t in thresholds:
            pred = (best_proba >= t).astype(int)
            approval_rates.append((pred == 0).mean())
            approved = pred == 0
            default_rates_at_thresh.append(y_test.values[approved].mean() if approved.sum() > 0 else 0)
            cm = confusion_matrix(y_test.values, pred)
            tn, fp, fn, tp = cm.ravel()
            fns.append(fn)
            fps.append(fp)

        fig, ax1 = plt.subplots(figsize=(10, 4.5))
        ax1.plot(thresholds, approval_rates, color="#1a56db", linewidth=2, label="Approval Rate")
        ax1.set_xlabel("Decision Threshold")
        ax1.set_ylabel("Approval Rate", color="#1a56db")
        ax1.tick_params(axis="y", labelcolor="#1a56db")
        ax2 = ax1.twinx()
        ax2.plot(thresholds, default_rates_at_thresh, color="#ef4444", linewidth=2, label="Default Rate")
        ax2.set_ylabel("Default Rate Among Approved", color="#ef4444")
        ax2.tick_params(axis="y", labelcolor="#ef4444")
        ax1.set_title("Approval Rate and Default Rate by Threshold", fontweight="bold")
        ax1.grid(True, alpha=0.3)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
        st.pyplot(fig)

        st.markdown("### Gain / Lift Chart")
        order = np.argsort(-best_proba)
        y_sorted = y_test.values[order]
        cumulative_defaults = np.cumsum(y_sorted)
        total_defaults = y_sorted.sum()
        pct_population = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
        pct_defaults = cumulative_defaults / total_defaults

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(pct_population, pct_defaults, color="#1a56db", linewidth=2, label="Model")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5, label="Random")
        ax.fill_between(pct_population, pct_defaults, pct_population, alpha=0.15, color="#1a56db")
        ax.set_xlabel("Population (%)")
        ax.set_ylabel("Defaults Captured (%)")
        ax.set_title("Gain Chart \u2014 Cumulative Defaults Captured", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.caption(
            "The gain chart shows the proportion of all defaults captured when reviewing the top-N% of accounts "
            "ranked by predicted default probability. A steeper curve indicates better ranking performance."
        )

        st.markdown("### Misclassification Costs")
        st.markdown(
            "Adjust the cost assumptions to reflect Indian lending economics. "
            "FN cost represents principal loss from a default. FP cost represents interest margin lost from declining a good customer."
        )
        c1, c2 = st.columns(2)
        with c1:
            C_FN = st.number_input("FN Cost (\u20b9, default loss)", min_value=1, value=100000, step=10000,
                                    help="Principal loss from a defaulted account")
        with c2:
            C_FP = st.number_input("FP Cost (\u20b9, missed margin)", min_value=1, value=15000, step=1000,
                                    help="Interest margin lost from declining a good customer")

        costs = [fn * C_FN + fp * C_FP for fn, fp in zip(fns, fps)]
        opt_idx = int(np.argmin(costs))

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(thresholds, costs, color="#ef4444", linewidth=2)
        ax.axvline(thresholds[opt_idx], color="#22c55e", linestyle="--", linewidth=2,
                   label=f"Optimal t = {thresholds[opt_idx]:.3f}")
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Total Cost (\u20b9)")
        ax.set_title("Cost Curve \u2014 Minimising Expected Loss", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Optimal Threshold", f"{thresholds[opt_idx]:.3f}")
        with r2:
            st.metric("Minimum Cost", f"\u20b9{costs[opt_idx]:,.0f}")
        with r3:
            pred_opt = (best_proba >= thresholds[opt_idx]).astype(int)
            cm = confusion_matrix(y_test, pred_opt)
            tn, fp, fn, tp = cm.ravel()
            st.metric("Recall at Optimum", f"{tp / (tp + fn):.1%}",
                      help=f"Caught {tp} / {tp + fn} defaults")
        with r4:
            st.metric("Precision at Optimum", f"{tp / (tp + fp):.1%}",
                      help=f"{tp} real defaults out of {tp + fp} flagged")


elif active_tab == tabs[5]:
    if not st.session_state.data_loaded:
        st.error("Data loading failed. Select a different dataset from the sidebar.")
        st.stop()
    st.markdown("<div class='main-header'>Explainability \u2014 Feature Attribution</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>SHAP-based feature attribution showing which borrower characteristics "
        "drive default probability, supporting RBI\u2019s fair-lending guidelines.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if not HAS_SHAP:
        st.warning(
            "SHAP not installed. Run `pip install shap` to enable this tab. "
            "For Streamlit Cloud deployments, add `shap>=0.44` to requirements.txt"
        )
        st.markdown("SHAP uses cooperative game theory to attribute feature contributions to individual predictions.")
    else:
        best_name = max(models, key=lambda n: (models[n]["pr_auc"] if models[n]["proba"] is not None else 0))

        X_all = df.drop(columns=["Class"])
        y_all = df["Class"]
        X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.25, stratify=y_all, random_state=RANDOM_STATE)

        with st.spinner("Training SHAP-compatible model and computing Shapley values..."):
            if HAS_LGB:
                shap_pipe = Pipeline([
                    ("features", CreditFeatures()),
                    ("clf", lgb.LGBMClassifier(
                        class_weight="balanced", n_estimators=300, learning_rate=0.05,
                        num_leaves=31, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
                    )),
                ])
            else:
                shap_pipe = Pipeline([
                    ("features", CreditFeatures()),
                    ("clf", RandomForestClassifier(
                        class_weight="balanced", n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
                    )),
                ])
            shap_pipe.fit(X_tr, y_tr)

            eng = shap_pipe.named_steps["features"]
            model = shap_pipe.named_steps["clf"]
            X_sample = X_te.sample(min(500, len(X_te)), random_state=RANDOM_STATE)
            X_eng = eng.transform(X_sample)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_eng)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        st.markdown("#### SHAP Summary Plot \u2014 Global Feature Importance")
        st.markdown("Features ranked by mean absolute SHAP value. "
                    "Color indicates feature value (red = high, blue = low).")

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X_eng, feature_names=X_eng.columns.tolist(),
                          show=False, max_display=15, alpha=0.7, color_bar=True)
        plt.title("SHAP Feature Impact on Default Probability", fontsize=12, fontweight="bold")
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("#### Single Prediction \u2014 Feature Contributions")
        st.markdown("Top contributing features for a high-confidence default prediction from the hold-out set.")

        proba_sample = shap_pipe.predict_proba(X_sample)[:, 1]
        default_idx = np.where(y_te.loc[X_sample.index].values == 1)[0]
        if len(default_idx):
            best_default = default_idx[np.argmax(proba_sample[default_idx])]
            contribs = pd.Series(shap_values[best_default], index=X_eng.columns).sort_values(key=abs, ascending=False)

            fig, ax = plt.subplots(figsize=(9, 5))
            top = contribs.head(10)
            colors_waterfall = ["#ef4444" if v > 0 else "#1a56db" for v in top.values]
            ax.barh(top.index[::-1], top.values[::-1], color=colors_waterfall[::-1], edgecolor="white")
            ax.axvline(0, color="black", linewidth=0.6)
            ax.set_xlabel("SHAP Value (impact on default probability log-odds)")
            ax.set_title(f"Top Contributors \u2014 Model Score = {proba_sample[best_default]:.3f}",
                         fontweight="bold")
            st.pyplot(fig)

            st.caption(
                "Positive SHAP values push the prediction toward DEFAULT. "
                "Negative SHAP values push the prediction toward NON-DEFAULT."
            )

        st.markdown("---")
        with st.expander("SHAP Formula \u2014 Shapley Values from Cooperative Game Theory"):
            st.latex(
                r"\phi_i = \sum_{S \subseteq N \setminus \{i\}} "
                r"\frac{|S|!\,(n - |S| - 1)!}{n!} \big( v(S \cup \{i\}) - v(S) \big)"
            )
            st.markdown(
                "Where $\\phi_i$ is the Shapley value for feature $i$, $S$ is a subset of features, "
                "and $v(S)$ is the model\u2019s prediction using only features in $S$."
            )


elif active_tab == tabs[6]:
    if not st.session_state.data_loaded:
        st.error("Data loading failed. Select a different dataset from the sidebar.")
        st.stop()
    st.markdown("<div class='main-header'>Model Card & About</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Documentation covering intended use, limitations, monitoring plan, "
        "and technical details for model governance.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    best_name = max(models, key=lambda n: (models[n]["pr_auc"] if models[n]["proba"] is not None else 0))
    best_pr = models[best_name]["pr_auc"]
    cv_best = max(v["mean"] for v in cv_results.values())
    ds_name = st.session_state.dataset_name
    ds_meta = DATASET_REGISTRY.get(ds_name)
    src_name = ds_meta["source"] if ds_meta else ds_name
    n_feats_mc = len(df.columns) - 1

    model_card = {
        "Model Name": "DefaultRisk-v1",
        "Task": "Binary classification (default vs non-default)",
        "Regulatory Context": "RBI Basel III \u2014 IRB approach for PD estimation",
        "Training Data": {
            "Source": src_name,
            "Rows": f"{len(df):,}",
            "Default Rate": f"{df['Class'].mean():.1%}",
            "Features": f"{n_feats_mc}",
        },
        "Model Architecture": best_name,
        "Primary Metric": "PR-AUC (Average Precision)",
        "Cross-Validation": f"{cv_best:.4f} (5-fold stratified mean)",
        "Hold-Out PR-AUC": f"{best_pr:.4f}",
        "Calibration": "Isotonic regression",
        "Explainability": "SHAP TreeExplainer",
    }

    st.json(model_card, expanded=False)

    st.markdown("---")
    st.markdown("### Intended Use & Limitations")

    c_use, c_lim = st.columns(2)
    with c_use:
        st.markdown("**Intended Use**")
        st.markdown("""
        - Credit card application scoring for Indian banks and NBFCs
        - PD estimation for RBI Basel III capital calculations
        - Portfolio risk monitoring and early warning systems
        - Credit limit management and collection prioritization
        """)

    with c_lim:
        st.markdown("**Limitations**")
        ds_meta_lim = DATASET_REGISTRY.get(st.session_state.dataset_name)
        lim_note = f"Dataset from {ds_meta_lim['country']} ({ds_meta_lim['year']})" if ds_meta_lim else "Dataset source"
        st.markdown(f"""
        - {lim_note} \u2014 Indian demographic mapping is illustrative, not exact
        - No CIBIL/credit bureau scores directly available in this dataset
        - No reject inference \u2014 only approved accounts in training data
        - Feature set may not capture long-term credit behavior
        """)

    st.markdown("---")
    st.markdown("### Monitoring Plan")
    mon_df = pd.DataFrame({
        "Layer": ["Input Drift", "Performance Drift", "Outcome Drift"],
        "Metric": [
            "PSI on score distribution; feature stability index",
            "Monthly PR-AUC back-test; KS statistic",
            "Actual vs predicted default rate by quarter",
        ],
        "Trigger": [
            "PSI > 0.25 on any top-5 feature",
            "PR-AUC drop > 10% from baseline",
            "Default rate deviation > 2\u00d7 std from expected",
        ],
        "Action": ["Retrain model", "Investigate and retrain", "Review with Credit Risk team"],
    })
    st.dataframe(mon_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Tech Stack")
    stack_cols = st.columns(4)
    tech_stack = [
        ("Python 3.11+", "Core language"),
        ("scikit-learn", "Pipeline, models, metrics"),
        ("LightGBM", "Gradient boosting (champion model)"),
        ("SHAP", "Explainability (TreeExplainer)"),
        ("Matplotlib/Seaborn", "Static visualisations"),
        ("Plotly", "Interactive charts"),
        ("Streamlit", "Dashboard framework"),
        ("UCI / OpenML", "Data source"),
    ]
    for i, (name, desc) in enumerate(tech_stack):
        with stack_cols[i % 4]:
            st.markdown(f"**{name}**  \n{desc}")

    st.markdown("---")
    st.markdown("### About This Project")
    st.markdown(
        "**DefaultRisk** is an end-to-end machine learning case study focused on Probability of Default (PD) "
        "modeling for credit card portfolios in the Indian banking context. The project demonstrates:"
    )
    st.markdown("""
    - **Technical depth:** leak-free pipelines, statistical testing, SHAP explainability
    - **Business alignment:** cost-sensitive thresholds, gain/lift charts, scorecard visualization
    - **Regulatory awareness:** RBI Basel III framework, model documentation, monitoring plans
    - **Engineering rigour:** sklearn pipelines, graceful fallbacks, self-contained app

    Designed for data science and ML engineering roles in banking and fintech.
    """)

    st.markdown("---")
    st.markdown(
        "<div class='footer'>"
        "DefaultRisk v1.0  \u00b7  Built with Streamlit  \u00b7  "
        f"Dataset: {st.session_state.dataset_name}  \u00b7  "
        f"Last updated: {datetime.now().strftime('%B %Y')}"
        "</div>",
        unsafe_allow_html=True,
    )


st.sidebar.markdown("---")
if st.session_state.data_loaded:
    ds_short = st.session_state.dataset_name.split("(")[0].strip()
    st.sidebar.caption(
        f"\U0001f3e6 DefaultRisk v1.0  \n"
        f"{ds_short}  \n"
        f"\U0001f4ca {len(df):,} rows  \u00b7  {int(df['Class'].sum())} defaults"
    )
else:
    st.sidebar.caption("\U0001f3e6 DefaultRisk v1.0  \u00b7  Built with Python + Streamlit")
