# DefaultRisk - Wells Fargo

> Loan default PD model with scorecard explainability.

**Company group:** Fintech / Banking / Quant  
**Task type:** tabular  
**App / framework:** streamlit  
**Dataset:** Lending Club + IEEE-CIS  
**Data links:** https://www.kaggle.com/datasets/wordsforthewise/lending-club

## Quickstart (runs out of the box, no downloads)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py            # trains on synthetic data, writes models/metrics.json
pytest -q                  # smoke test the pipeline
streamlit run app.py        # launches Streamlit dashboard
```

The project ships with a **synthetic data generator** so everything trains immediately. A pure-NumPy model baseline is built in, and stronger libraries (scikit-learn / LightGBM / PyTorch / etc.) are used automatically when installed.

## Use the real dataset
1. Download from the data links above into `data/raw/`.
2. Swap `make_synthetic()` for the `load_real(...)` helper in `src/data.py`.
3. Re-run `python train.py`.

## Repo structure
```
DefaultRisk/
  src/        core ML lib, data, model, evaluate, persist
  train.py    end-to-end training entrypoint
  app.py      streamlit demo
  tests/      pytest smoke test
  data/       put real datasets in data/raw/
  models/     saved model + metrics (gitignored)
  requirements.txt, README.md, LICENSE, Makefile
```

## Next steps to make it portfolio-grade
- Replace synthetic data with the real dataset and re-tune.
- Enable the optional libraries in `requirements.txt` for SOTA models.
- Add cross-validation, hyperparameter search, and CI.
- Deploy the streamlit app (Streamlit Community Cloud / Hugging Face Spaces).

## License
MIT
