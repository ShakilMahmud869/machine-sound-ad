"""Models.

Milestone 2: dense autoencoder (published DCASE baseline design).
Milestone 3: small CNN with auxiliary machine-type / attribute classification.
"""
import torch.nn as nn


class AutoEncoder(nn.Module):
    """640 -> 128 -> 128 -> 128 -> 128 -> 8 -> 128 -> 128 -> 128 -> 128 -> 640.

    Every hidden layer: Linear + BatchNorm1d + ReLU. Output layer: plain Linear.

    TODO: __init__(self, d_in, hidden, bottleneck) building self.enc and
    self.dec; forward(x) returns the reconstruction (same shape as x).
    """

    def __init__(self, d_in=640, hidden=128, bottleneck=8):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class CNNClassifier(nn.Module):
    """Milestone 3 — do not start before the baseline is reproduced.

    Input: log-mel spectrogram (batch, 1, n_mels, T) or random crops.
    4 conv blocks (Conv2d + BatchNorm2d + ReLU + pooling)
      -> global average pooling -> 128-d embedding -> linear classifier.

    TODO: forward(x) should return (logits, embedding) so the same model
    is used for training (cross-entropy on logits) and scoring (k-NN on
    embeddings).
    """

    def __init__(self, n_classes, emb_dim=128):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
