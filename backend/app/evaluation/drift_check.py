from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.models import DataDriftMetric, Inference, Model
from app.metrics import data_drift_score
from app.ml.drift import histogram, kl_divergence
from app.ml.schema import NUMERIC_FEATURES


def check_drift(db: Session, model: Model, *, lookback_hours: int) -> dict[str, float]:
    """Compares each numeric feature's recent live distribution against the training-time
    baseline stored on the model row, records a data_drift_metrics row per feature, and
    returns {feature_name: drift_score}."""

    baseline = model.metrics.get("feature_baseline", {})
    if not baseline:
        return {}

    since = datetime.now(UTC) - timedelta(hours=lookback_hours)
    rows = db.execute(
        select(Inference.input_features).where(Inference.model_id == model.id, Inference.created_at >= since)
    ).scalars().all()

    if len(rows) < 20:
        return {}

    scores: dict[str, float] = {}
    for feature in NUMERIC_FEATURES:
        expected = baseline.get(feature)
        if expected is None:
            continue
        live_values = [row[feature] for row in rows if feature in row]
        if not live_values:
            continue

        score = kl_divergence(expected, live_values)
        scores[feature] = score

        db.add(
            DataDriftMetric(
                model_id=model.id,
                feature_name=feature,
                expected_distribution=expected,
                # Finite edges for the persisted/displayed distribution (JSON has no
                # infinity token, unlike the +-inf edges kl_divergence() uses internally to
                # correctly score out-of-range drift - see app/ml/drift.py).
                actual_distribution=histogram(live_values, bin_edges=expected["bin_edges"]),
                drift_score=round(score, 4),
            )
        )
        data_drift_score.labels(model_name=model.name, feature=feature).set(score)

    return scores
