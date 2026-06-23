# DefaultRisk — Loan Default Prediction

Probability of Default (PD) model for retail banking credit risk, built for the German Credit dataset. Demonstrates production-grade ML pipelines, scorecard visualization, and regulatory-ready explainability.

Built for data science and ML engineering roles in banking and fintech.

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/bravo2024/DefaultRisk-streamlit.git
cd DefaultRisk-streamlit

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app loads the German Credit dataset from OpenML. Click **Load Dataset & Train Models** on any tab to begin.

---

## What This App Shows

| Section | What It Covers |
|---|---|
| **Overview** | Credit risk context, Basel framework, PD modeling fundamentals |
| **Data Explorer** | Default distributions, numeric and categorical feature analysis, correlations |
| **Feature Engineering** | Leak-free encoding pipeline for mixed data types |
| **Model Benchmarks** | LR vs RF vs LightGBM (5-fold CV), confusion matrices, McNemar test |
| **Scorecard** | Score distribution, approval rates, gain/lift chart, cost optimization |
| **Explainability** | SHAP global importance and per-prediction attribution |
| **Model Card** | Documentation, intended use, limitations, monitoring plan |

---

## Key Techniques

- **PR-AUC** — ranking metric appropriate for imbalanced credit data
- **Leak-free Pipeline** — sklearn `BaseEstimator` + `TransformerMixin` prevents data leakage
- **Cost-Sensitive Threshold** — minimises expected loss from FN and FP decisions
- **Probability Calibration** — isotonic regression for unbiased PD estimates
- **SHAP Explainability** — Shapley values for regulatory compliance
- **McNemar Test** — statistical significance between model pairs
- **Gain/Lift Charts** — operational ranking performance evaluation

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11+ | Core language |
| scikit-learn | Pipeline, models, CV, metrics |
| LightGBM | Gradient boosting (champion model) |
| SHAP | TreeExplainer for explainability |
| Streamlit | Interactive dashboard |
| Matplotlib + Seaborn | Static visualisations |
| Plotly | Interactive charts |
| OpenML | Dataset source |

---

## License

MIT

---

## Acknowledgements

- **German Credit Dataset** (UCI Machine Learning Repository)
- Inspired by credit risk modeling practices in retail banking
