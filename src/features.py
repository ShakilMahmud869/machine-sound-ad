"""WAV -> log-mel spectrogram -> stacked frames, cached to disk.

Milestone 1. Run as:
    python -m src.features --config configs/baseline.yaml --machine fan
"""
import argparse


def load_config(path):
    """Read the YAML config into a dict.

    TODO: use yaml.safe_load. (Other modules import this function.)
    """
    raise NotImplementedError


def logmel(wav_path, sr, n_fft, hop_length, n_mels, power=2.0):
    """Return the log-mel spectrogram of one WAV file, in dB.

    Returns: tensor of shape (n_mels, T).

    TODO:
      - load with torchaudio.load; check the file's sample rate equals `sr`
      - torchaudio.transforms.MelSpectrogram on the first channel
      - convert power to dB: 10 * log10(mel + small epsilon)
    Question to be able to answer: what do n_fft and hop_length control?
    """
    raise NotImplementedError


def stack_frames(spec, n_frames):
    """Concatenate `n_frames` neighbouring frames into one vector.

    spec: (n_mels, T)  ->  returns (T - n_frames + 1, n_frames * n_mels)

    TODO: implement with Tensor.unfold (fast version).
    Decide and document the order of values inside each 640-vector
    (frame-major or mel-major) — both versions below must agree.
    """
    raise NotImplementedError


def stack_frames_loop(spec, n_frames):
    """Slow reference version of stack_frames using a plain for-loop.

    Only used in tests/test_features.py to check stack_frames exactly.
    """
    raise NotImplementedError


def parse_filename(path):
    """Extract metadata from a DCASE file name.

    Returns a dict such as {"section": ..., "domain": "source"/"target",
    "label": "normal"/"anomaly", "attributes": {...}}.

    TODO: first look at real file names in notebooks/01_explore.ipynb,
    then write this. Test files' labels come from here.
    """
    raise NotImplementedError


def cache_machine(cfg, machine):
    """Compute features for every train and test WAV of one machine type
    and save them under cfg["paths"]["cache_dir"] as .pt files.

    TODO:
      - one .pt per split (train / test), containing per-file stacked frames
        plus the metadata from parse_filename
      - skip work if the cache already exists
      - assert torch.isfinite(...).all() before saving
    """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--machine", required=True)
    args = parser.parse_args()
    # TODO: load config, call cache_machine, print the resulting shapes
    raise NotImplementedError


if __name__ == "__main__":
    main()
