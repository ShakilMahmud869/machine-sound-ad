"""Anomaly scores and metrics (AUC, pAUC, official harmonic-mean score).

Milestone 2 (autoencoder), reused unchanged in Milestone 3 for a fair comparison.
Run as:
    python -m src.evaluate --config configs/baseline.yaml --machine fan
"""
import argparse


def score_files(model, file_loader, device):
    """Anomaly score per test file = mean reconstruction MSE over its vectors.

    Returns a table (e.g. pandas DataFrame) with filename, domain, label, score.
    TODO: model.eval() + torch.no_grad().
    """
    raise NotImplementedError


def auc(labels, scores):
    """TODO: sklearn.metrics.roc_auc_score."""
    raise NotImplementedError


def pauc_sklearn(labels, scores, max_fpr):
    """roc_auc_score(..., max_fpr=max_fpr). Note: sklearn applies the
    McClish standardisation here — find out what that means."""
    raise NotImplementedError


def pauc_raw(labels, scores, max_fpr):
    """Raw partial area under the ROC curve for FPR in [0, max_fpr],
    computed yourself from sklearn.metrics.roc_curve (interpolate at
    max_fpr), divided by max_fpr. Decide which pAUC you report, and say so
    in the README."""
    raise NotImplementedError


def harmonic_mean(values):
    """Official-style score: harmonic mean of all AUCs and pAUCs."""
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--machine", required=True)
    args = parser.parse_args()
    # TODO:
    #   - load best checkpoint (+ its mean/std), build FileDataset for test
    #   - score files; AUC separately for source and target; pAUC
    #   - append a row to results/baseline.csv, published numbers beside yours
    #   - histogram of scores: normal vs anomalous -> results/
    raise NotImplementedError


if __name__ == "__main__":
    main()
