#!/usr/bin/env python
"""Trains the initial Model A (logistic regression, v2.1) and Model B (gradient boosting,
v2.2) on data/train.csv, registers both in Postgres + MLflow, and creates the 80/20 production
A/B deployment the PRD specifies. Run from the repo root, against a running stack:

    python scripts/train_model.py

Idempotent-ish: re-running trains new rows under the same (name, version) and will fail on the
unique constraint if versions 2.1/2.2 already exist - drop them first (or bump the version
constants below) if you want to retrain from scratch.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# pydantic-settings resolves `env_file=".env"` relative to the process's cwd, not this file's
# location - load it explicitly so this script works the same whether run from the repo root
# or anywhere else, and so it happens before the `app.core.db` import below creates the engine.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(BACKEND_DIR / ".env")

import pandas as pd  # noqa: E402

from app.core.db import SessionLocal  # noqa: E402
from app.core.mlflow_client import configure_mlflow  # noqa: E402
from app.core.models import Deployment  # noqa: E402
from app.ml.training import train_and_register  # noqa: E402

MODEL_NAME = "ctr-recommender"
DATA_DIR = REPO_ROOT / "data"


def main() -> None:
    train_csv = DATA_DIR / "train.csv"
    if not train_csv.exists():
        raise SystemExit(f"{train_csv} not found - run scripts/generate_dataset.py first")

    df = pd.read_csv(train_csv)
    configure_mlflow()

    db = SessionLocal()
    try:
        model_a = train_and_register(
            db, name=MODEL_NAME, version="2.1", algo="logistic_regression", df=df, status="production"
        )
        model_b = train_and_register(
            db, name=MODEL_NAME, version="2.2", algo="gradient_boosting", df=df, status="production"
        )
        db.add_all(
            [
                Deployment(model_id=model_a.id, environment="production", traffic_split="0.8", is_active=True),
                Deployment(model_id=model_b.id, environment="production", traffic_split="0.2", is_active=True),
            ]
        )
        db.commit()

        print(f"Model A ({model_a.version}): {model_a.metrics['accuracy']:.3f} accuracy, id={model_a.id}")
        print(f"Model B ({model_b.version}): {model_b.metrics['accuracy']:.3f} accuracy, id={model_b.id}")
        print("Deployments: 80% -> 2.1, 20% -> 2.2 (environment=production)")
    finally:
        db.close()


if __name__ == "__main__":
    main()
