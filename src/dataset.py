"""PyTorch Datasets built on the cached features from src/features.py.

Milestone 1, steps 6-7.
"""
from torch.utils.data import Dataset


def compute_norm_stats(train_vectors):
    """Mean and std per input dimension, from TRAINING data only.

    Returns (mean, std), each of shape (640,).
    TODO: store them (e.g. inside the checkpoint) so evaluation reuses
    exactly the same values. Never compute them on test data.
    """
    raise NotImplementedError


class FrameDataset(Dataset):
    """Training dataset: one normalised 640-vector per item.

    TODO:
      - __init__(self, cache_path, file_ids, mean, std): load cached
        features, keep only the files in `file_ids` (train/val split is by
        FILE, so frames of one clip never land in both)
      - __len__: total number of frames
      - __getitem__(i): one (640,) float tensor
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class FileDataset(Dataset):
    """Test dataset: one item per FILE (you score files, not frames).

    __getitem__(i) -> (all normalised vectors of file i: (n_vectors, 640),
                       label 0/1, domain "source"/"target", filename)

    TODO: implement; use batch_size=1 or a custom collate_fn in the DataLoader.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
