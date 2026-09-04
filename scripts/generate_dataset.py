#!/usr/bin/env python
"""Generates the synthetic click-through dataset used throughout the demo. Run from the repo
root (needs the backend venv active, or run via the backend's python directly):

    python scripts/generate_dataset.py

Writes to data/:
  - train.csv            5,000 rows, labeled, non-drifted - used to train Model A/B.
  - holdout_normal.csv    2,000 rows, labeled, non-drifted - "normal" live traffic wave.
  - holdout_drifted.csv   2,000 rows, labeled, drifted     - "drifted" live traffic wave, used
                          to demonstrate the accuracy-drop -> retraining trigger honestly (the
                          model was never trained on this distribution).
  - batch_100k.csv        100,000 rows, unlabeled - fed to /batch-predict for a real throughput
                          number.

None of these files are committed (see .gitignore) - they're regenerated deterministically
(fixed RNG seeds) by this script.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.ml.label_fn import sample_labels  # noqa: E402
from app.ml.schema import DEVICES, FEATURE_ORDER, ITEM_CATEGORIES  # noqa: E402

DATA_DIR = REPO_ROOT / "data"


def make_features(n: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "user_age": rng.integers(18, 70, n).astype(float),
            "user_tenure_days": rng.integers(0, 2500, n).astype(float),
            "user_avg_session_min": rng.uniform(1, 60, n),
            "user_category_affinity": rng.uniform(0, 1, n),
            "item_price": rng.uniform(5, 300, n),
            "item_popularity": rng.uniform(0, 1, n),
            "item_category": rng.choice(ITEM_CATEGORIES, n),
            "hour_of_day": rng.integers(0, 24, n),
            "history_click_rate": rng.uniform(0, 1, n),
            "device": rng.choice(DEVICES, n),
        }
    )


def make_labeled(n: int, *, seed: int, drift: bool) -> pd.DataFrame:
    df = make_features(n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    df["clicked"] = sample_labels(df, drift=drift, rng=rng)
    return df[[*FEATURE_ORDER, "clicked"]]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train = make_labeled(5_000, seed=100, drift=False)
    train.to_csv(DATA_DIR / "train.csv", index=False)
    print(f"train.csv: {len(train)} rows, click rate {train['clicked'].mean():.3f}")

    holdout_normal = make_labeled(2_000, seed=200, drift=False)
    holdout_normal.to_csv(DATA_DIR / "holdout_normal.csv", index=False)
    print(f"holdout_normal.csv: {len(holdout_normal)} rows, click rate {holdout_normal['clicked'].mean():.3f}")

    holdout_drifted = make_labeled(2_000, seed=300, drift=True)
    holdout_drifted.to_csv(DATA_DIR / "holdout_drifted.csv", index=False)
    print(f"holdout_drifted.csv: {len(holdout_drifted)} rows, click rate {holdout_drifted['clicked'].mean():.3f}")

    batch = make_features(100_000, seed=400)[FEATURE_ORDER]
    batch.to_csv(DATA_DIR / "batch_100k.csv", index=False)
    print(f"batch_100k.csv: {len(batch)} rows (unlabeled)")


if __name__ == "__main__":
    main()
