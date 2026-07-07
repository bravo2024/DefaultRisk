"""train.py - GNN credit-risk model training entrypoint."""
from src.data import make_synthetic
from src.model import fit_and_evaluate
from src.evaluate import save_metrics, print_report
from src.persist import save_model

def main():
    print("Generating synthetic credit data...")
    data = make_synthetic(n=4000, seed=42)

    print("Training GNN credit-risk model (arXiv:2605.12782)...")
    try:
        model, metrics = fit_and_evaluate(data, verbose=True)
    except ImportError:
        print("  PyTorch not available — falling back to sklearn baseline.")
        from src.model_fallback import fit_and_evaluate as fallback
        model, metrics = fallback(data)

    save_model(model)
    save_metrics(metrics)
    print_report(metrics)
    print(f"\nSaved model -> models/model.pkl and metrics -> models/metrics.json")


if __name__ == "__main__":
    main()
