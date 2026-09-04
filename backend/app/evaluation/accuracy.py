from datetime import UTC, datetime, timedelta

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.core.models import Inference, Model


def recent_accuracy(db: Session, model_id, *, lookback_hours: int) -> tuple[float | None, int]:
    """Returns (accuracy, labeled_count) over labeled inferences in the lookback window.
    accuracy is None if there isn't enough labeled feedback yet to say anything meaningful."""

    since = datetime.now(UTC) - timedelta(hours=lookback_hours)
    is_correct = Inference.predicted_output["label"].astext == Inference.actual_output["label"].astext

    row = db.execute(
        select(
            func.count().label("labeled_count"),
            func.avg(case((is_correct, 1.0), else_=0.0)).label("accuracy"),
        ).where(
            Inference.model_id == model_id,
            Inference.actual_output.isnot(None),
            Inference.created_at >= since,
        )
    ).first()

    if row is None or row.labeled_count == 0:
        return None, 0
    return float(row.accuracy), int(row.labeled_count)


def build_training_frame(db: Session, model_name: str):
    """Pulls all labeled feedback for a model name (across versions) into a DataFrame shaped
    like the original synthetic training set, for use as retraining data."""

    import pandas as pd

    from app.ml.schema import FEATURE_ORDER

    rows = db.execute(
        select(Inference.input_features, Inference.actual_output).where(
            Inference.model_name == model_name, Inference.actual_output.isnot(None)
        )
    ).all()

    records = []
    for input_features, actual_output in rows:
        record = {feature: input_features[feature] for feature in FEATURE_ORDER}
        record["clicked"] = 1 if actual_output["label"] == "click" else 0
        records.append(record)

    return pd.DataFrame.from_records(records)


def model_name_for(db: Session, model_id) -> str | None:
    model = db.get(Model, model_id)
    return model.name if model else None
