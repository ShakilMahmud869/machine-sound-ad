"""Training loop — write this one entirely yourself; it's what interviewers ask about.

Milestone 2. Run as:
    python -m src.train --config configs/baseline.yaml --machine fan
"""
import argparse


def set_seed(seed):
    """Seed python `random`, numpy and torch so one command reproduces results."""
    raise NotImplementedError


def split_train_val(file_ids, val_fraction, seed):
    """Split training FILES (not frames) into train / validation lists."""
    raise NotImplementedError


def train_one_epoch(model, loader, optimizer, device):
    """One pass over the training data. Returns mean loss.

    TODO: model.train(); for each batch: forward, MSE(recon, x),
    optimizer.zero_grad(), loss.backward(), optimizer.step().
    """
    raise NotImplementedError


def validate(model, loader, device):
    """Mean validation loss. TODO: model.eval() + torch.no_grad().
    Be able to explain what each of those two changes."""
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--machine", required=True)
    args = parser.parse_args()
    # TODO:
    #   - load config, set seed, pick device (cuda if available)
    #   - split files, compute norm stats on the train split only
    #   - build DataLoaders, model, Adam(lr from config)
    #   - loop over epochs: log train + val loss, save the best checkpoint
    #     (model state + mean/std + config) to checkpoint_dir
    #   - save the loss curve (CSV and/or PNG) to results/
    raise NotImplementedError


if __name__ == "__main__":
    main()
