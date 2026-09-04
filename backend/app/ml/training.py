"""Shared training/registration logic used both by the initial `scripts/train_model.py` and
by the evaluator's automatic retraining trigger, so the two paths can never drift apart."""

import logging
from pathlib import Path
from typing import Literal

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.models import Model
from app.ml.drift import histogram
from app.ml.schema import CATEGORICAL_FEATURES, FEATURE_ORDER, NUMERIC_FEATURES

logger = logging.getLogger("inferops.training")

Algorithm = Literal["logistic_regression", "gradient_boosting"]

_ALGO_FACTORIES = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=42),
}
_FRAMEWORK = "scikit-learn"


def build_pipeline(algo: Algorithm) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("classifier", _ALGO_FACTORIES[algo]())])


def _feature_baseline(train_df: pd.DataFrame) -> dict:
    return {feature: histogram(train_df[feature].tolist()) for feature in NUMERIC_FEATURES}


def train_and_register(
    db: Session,
    *,
    name: str,
    version: str,
    algo: Algorithm,
    df: pd.DataFrame,
    label_col: str = "clicked",
    status: str = "production",
    trigger_reason: str | None = None,
) -> Model:
    """Trains `algo` on `df`, logs to MLflow, saves a joblib artifact, and inserts (without
    committing) a `models` row. Caller is responsible for `db.commit()`."""

    settings = get_settings()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_col])

    pipeline = build_pipeline(algo)
    pipeline.fit(train_df[FEATURE_ORDER], train_df[label_col])

    y_pred = pipeline.predict(test_df[FEATURE_ORDER])
    y_proba = pipeline.predict_proba(test_df[FEATURE_ORDER])[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(test_df[label_col], y_pred), 4),
        "precision": round(precision_score(test_df[label_col], y_pred, zero_division=0), 4),
        "recall": round(recall_score(test_df[label_col], y_pred, zero_division=0), 4),
        "f1": round(f1_score(test_df[label_col], y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(test_df[label_col], y_proba), 4),
        "training_rows": len(train_df),
        "test_rows": len(test_df),
    }
    metrics["algorithm"] = algo
    if trigger_reason:
        metrics["retrain_trigger_reason"] = trigger_reason

    artifact_dir = Path(settings.model_artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{name}-{version}.joblib"
    joblib.dump(pipeline, artifact_path)
    # Always store forward slashes: this path gets written on Windows (local `python
    # scripts/train_model.py` runs) and read back inside Linux containers, where a literal
    # backslash in a relative path is just a filename character, not a separator.
    artifact_path_str = artifact_path.as_posix()

    mlflow_run_id = None
    try:
        with mlflow.start_run(run_name=f"{name}-{version}") as run:
            mlflow.log_param("algorithm", algo)
            mlflow.log_param("model_name", name)
            mlflow.log_param("version", version)
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, int | float)})
            mlflow.sklearn.log_model(pipeline, artifact_path="model", registered_model_name=name)
            mlflow_run_id = run.info.run_id
    except Exception:
        logger.exception("MLflow logging failed; continuing without tracking (artifact still saved to disk)")

    model_row = Model(
        name=name,
        version=version,
        model_type="binary_classifier",
        framework=_FRAMEWORK,
        artifact_path=artifact_path_str,
        input_shape={"features": FEATURE_ORDER},
        output_shape={"labels": ["no_click", "click"]},
        metrics={**metrics, "feature_baseline": _feature_baseline(train_df)},
        mlflow_run_id=mlflow_run_id,
        status=status,
    )
    db.add(model_row)
    db.flush()
    return model_row
