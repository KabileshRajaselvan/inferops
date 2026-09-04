import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.models import Model, RetrainingJob
from app.evaluation.accuracy import build_training_frame
from app.metrics import retraining_jobs_total
from app.ml.training import train_and_register

logger = logging.getLogger("inferops.retrain")

MIN_LABELED_ROWS_FOR_RETRAIN = 150


def has_recent_retraining_job(db: Session, model_id) -> bool:
    """True if a retraining job for this model started within the cooldown window - guards
    against the evaluator's timer loop (and/or an on-demand /evaluate call landing in the same
    window) re-triggering retraining on every single pass while the accuracy regression that
    caused it hasn't been resolved yet (i.e. the retrained candidate hasn't been promoted to a
    deployment). Without this, a sustained regression spawns a new job - and a new `-rtN`
    model version - every evaluation pass indefinitely."""

    cooldown = timedelta(minutes=get_settings().retraining_cooldown_minutes)
    since = datetime.now(UTC) - cooldown
    recent = db.execute(
        select(RetrainingJob.id).where(RetrainingJob.model_id == model_id, RetrainingJob.started_at >= since).limit(1)
    ).scalar_one_or_none()
    return recent is not None


def _next_retrain_version(db: Session, base_version: str, model_name: str) -> str:
    count = db.execute(
        select(func.count()).where(Model.name == model_name, Model.version.like(f"{base_version}-rt%"))
    ).scalar_one()
    return f"{base_version}-rt{count + 1}"


def maybe_retrain(db: Session, model: Model, *, accuracy_drop: float) -> RetrainingJob:
    """Always records a retraining_jobs row (so the trigger is auditable even when skipped for
    lack of data); only actually retrains when there's enough freshly labeled feedback."""

    trigger_reason = f"accuracy_drop_{accuracy_drop:.4f}"
    job = RetrainingJob(
        model_id=model.id,
        trigger_reason=trigger_reason,
        status="running",
        started_at=datetime.now(UTC),
    )
    db.add(job)
    db.flush()
    retraining_jobs_total.labels(trigger_reason="accuracy_drop").inc()

    training_df = build_training_frame(db, model.name)
    if len(training_df) < MIN_LABELED_ROWS_FOR_RETRAIN or training_df["clicked"].nunique() < 2:
        job.status = "skipped_insufficient_data"
        job.completed_at = datetime.now(UTC)
        logger.warning(
            "Retraining triggered for %s but only %d labeled rows available (need %d); skipping",
            model.name,
            len(training_df),
            MIN_LABELED_ROWS_FOR_RETRAIN,
        )
        return job

    algo = model.metrics.get("algorithm", "logistic_regression")

    try:
        # A SAVEPOINT, not the outer transaction: if train_and_register's insert fails (e.g. a
        # version-name collision from a near-simultaneous retrain), only this nested block rolls
        # back - the `job` row above (and anything already done for other models in this same
        # /evaluate call) survives, instead of the whole call dying with a PendingRollbackError.
        with db.begin_nested():
            new_version = _next_retrain_version(db, model.version, model.name)
            new_model = train_and_register(
                db,
                name=model.name,
                version=new_version,
                algo=algo,
                df=training_df,
                status="staging",
                trigger_reason=trigger_reason,
            )
        job.status = "completed"
        job.new_version = new_model.version
        job.new_model_id = new_model.id
    except Exception:
        logger.exception("Retraining failed for %s", model.name)
        job.status = "failed"
    finally:
        job.completed_at = datetime.now(UTC)

    return job
