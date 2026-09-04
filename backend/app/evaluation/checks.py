import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.models import Deployment, Model
from app.evaluation.accuracy import recent_accuracy
from app.evaluation.drift_check import check_drift
from app.evaluation.retrain import has_recent_retraining_job, maybe_retrain
from app.metrics import model_accuracy

logger = logging.getLogger("inferops.checks")

MIN_LABELED_FOR_ACCURACY_CHECK = 30


def run_checks_for_model(db: Session, model_name: str) -> dict:
    settings = get_settings()

    active_model_ids = db.execute(
        select(Deployment.model_id)
        .join(Model)
        .where(Model.name == model_name, Deployment.is_active.is_(True), Deployment.environment == "production")
    ).scalars().all()

    results: dict = {"model_name": model_name, "versions": []}

    for model_id in set(active_model_ids):
        model = db.get(Model, model_id)
        if model is None:
            continue

        version_result: dict = {"version": model.version, "model_id": str(model.id)}

        drift_scores = check_drift(db, model, lookback_hours=settings.drift_lookback_hours)
        version_result["drift"] = drift_scores

        accuracy, labeled_count = recent_accuracy(db, model.id, lookback_hours=settings.accuracy_lookback_hours)
        version_result["live_accuracy"] = accuracy
        version_result["labeled_count"] = labeled_count
        baseline_accuracy = model.metrics.get("accuracy")
        version_result["baseline_accuracy"] = baseline_accuracy

        if accuracy is not None:
            model_accuracy.labels(model_name=model.name, version=model.version).set(accuracy)

        retrain_triggered = False
        if (
            accuracy is not None
            and baseline_accuracy is not None
            and labeled_count >= MIN_LABELED_FOR_ACCURACY_CHECK
        ):
            drop = baseline_accuracy - accuracy
            if drop > settings.accuracy_drop_threshold:
                if has_recent_retraining_job(db, model.id):
                    version_result["retrain_skipped_reason"] = "cooldown"
                else:
                    retrain_triggered = True
                    job = maybe_retrain(db, model, accuracy_drop=drop)
                    version_result["retraining_job"] = {"id": str(job.id), "status": job.status}

        version_result["retrain_triggered"] = retrain_triggered
        results["versions"].append(version_result)

    db.commit()
    return results
