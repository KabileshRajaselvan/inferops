"""Loads trained model artifacts (sklearn Pipeline joblib files) from the shared artifact
directory into an in-process cache, keyed by model id. The API never talks to MLflow on the
hot path (see README Trade-offs) - it reads `artifact_path` from the Postgres `models` row and
loads straight off disk."""

import logging
import uuid
from pathlib import Path
from threading import Lock

import joblib
import pandas as pd

from app.ml.schema import FEATURE_ORDER, PredictFeatures

logger = logging.getLogger("inferops.registry")


class ModelRegistry:
    def __init__(self) -> None:
        self._cache: dict[uuid.UUID, object] = {}
        self._lock = Lock()

    def get_pipeline(self, model_id: uuid.UUID, artifact_path: str):
        with self._lock:
            pipeline = self._cache.get(model_id)
            if pipeline is not None:
                return pipeline

            path = Path(artifact_path)
            if not path.exists():
                raise FileNotFoundError(f"Model artifact not found at {artifact_path}")

            pipeline = joblib.load(path)
            self._cache[model_id] = pipeline
            logger.info("Loaded model artifact %s for model_id=%s", artifact_path, model_id)
            return pipeline

    def invalidate(self, model_id: uuid.UUID) -> None:
        with self._lock:
            self._cache.pop(model_id, None)


def predict_one(pipeline, features: PredictFeatures) -> float:
    """Returns P(click) in [0, 1]."""
    row = pd.DataFrame([features.model_dump()])[FEATURE_ORDER]
    return float(pipeline.predict_proba(row)[0, 1])


def predict_many(pipeline, rows: pd.DataFrame) -> "pd.Series[float]":
    ordered = rows[FEATURE_ORDER]
    return pd.Series(pipeline.predict_proba(ordered)[:, 1], index=rows.index)


registry = ModelRegistry()
