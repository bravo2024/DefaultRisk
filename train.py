
"""train.py - build data, train, evaluate, persist."""
from src.data import load_real_german_credit
from src.model import fit_and_evaluate
from src.evaluate import save_metrics, print_report
from src.persist import save_model

def main():
    print("Loading real dataset...")
    data=load_real_german_credit()
    print("Training model...")
    model,metrics=fit_and_evaluate(data)
    save_model(model); save_metrics(metrics); print_report(metrics)
    print("\nSaved model -> models/model.pkl and metrics -> models/metrics.json")
if __name__=="__main__": main()
