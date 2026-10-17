# machine-sound-ad

Unsupervised anomalous sound detection for machine condition monitoring, in PyTorch,
evaluated on the DCASE 2025 Task 2 benchmark (AUC / pAUC per machine type).

> **Status:** work in progress — Milestone 1 (data pipeline).
> Results below will only ever contain numbers produced by the code in this repo.

## Result

_TODO (Milestone 5): one-sentence result, e.g. "A CNN trained to classify normal machine
sounds reaches official score X vs Y for the autoencoder baseline."_

## Results table

| Machine | Model | AUC source | AUC target | pAUC | Published baseline (src / tgt) |
|---------|-------|-----------:|-----------:|-----:|-------------------------------:|
| fan     | AE    |            |            |      | 70.96 / 38.75                  |
| valve   | AE    |            |            |      | 63.53 / 67.18                  |

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# Put the DCASE 2025 Task 2 development set (Zenodo record 15097779) under data/
python -m src.features --config configs/baseline.yaml --machine fan   # cache log-mels
python -m src.train    --config configs/baseline.yaml --machine fan
python -m src.evaluate --config configs/baseline.yaml --machine fan
```

## Repository layout

```
configs/baseline.yaml      all hyperparameters (nothing hard-coded)
src/features.py            wav -> log-mel -> stacked frames, cached to disk
src/dataset.py             PyTorch Datasets (frame-level train, file-level test)
src/models.py              autoencoder, CNN classifier
src/train.py               training loop, validation, checkpoints
src/evaluate.py            anomaly scores, AUC, pAUC, results CSV
tests/                     small correctness checks (e.g. unfold vs for-loop)
notebooks/01_explore.ipynb listening + spectrogram plots only
results/                   CSVs + figures that are published
```

## What I learned / limitations

_TODO (Milestone 4–5)._

## License

MIT — see [LICENSE](LICENSE).
