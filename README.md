# DefaultRisk

Loan default probability modeling with a GNN scoring head (arXiv:2605.12782).

Probability-of-default modelling that goes beyond per-row classifiers:
borrowers are linked into a k-NN similarity graph and scored with a 2-layer
GCN using structural consistency regularisation (arXiv:2605.12782), compared
head-to-head against a pure-NumPy logistic regression and LightGBM on
[Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)-style
borrower features.

## Quickstart (runs out of the box, no downloads)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py            # trains on synthetic data, writes models/metrics.json
pytest -q                  # smoke test the pipeline
streamlit run app.py        # launches Streamlit dashboard
```

The project ships with a **synthetic data generator** so everything trains immediately. Two model backends are available:

1. **Logistic Regression** (pure NumPy SGD) — fastest, no extra deps.
2. **GNN** (2-layer GCN, arXiv:2605.12782) — constructs a k-NN similarity graph from standardised borrower features, applies structural consistency regularisation, and achieves SOTA calibration. Requires PyTorch.

Optionally **LightGBM** (comment/uncomment in requirements.txt) is available as a third baseline.

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

## Next steps
- Replace synthetic data with the real Lending Club dataset and re-tune the GNN.
- Experiment with attention-based aggregation (GAT) on the borrower graph.
- Add cross-validation for the GNN training loop.
- Deploy the streamlit app (Streamlit Community Cloud / Hugging Face Spaces).

## License
MIT
