# DefaultRisk - Wells Fargo

> Loan default PD model with GNN credit-risk scoring (arXiv:2605.12782).

**Company group:** Fintech / Banking / Quant  
**Task type:** graph (k-NN similarity)  
**App / framework:** streamlit  
**Dataset:** Lending Club + IEEE-CIS  
**Reference:** arXiv:2605.12782 — Structural regularisation for graph-based default prediction  
**Data links:** https://www.kaggle.com/datasets/wordsforthewise/lending-club

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
