import asyncio
import logging

from sqlalchemy import select

from app.config import get_settings
from app.core.db import SessionLocal
from app.core.mlflow_client import configure_mlflow
from app.core.models import Model
from app.evaluation.checks import run_checks_for_model
from app.evaluation.partitions import ensure_partitions

logger = logging.getLogger("inferops.evaluator")


def run_once() -> dict:
    settings = get_settings()
    db = SessionLocal()
    try:
        ensure_partitions(db, lookahead_days=settings.partition_lookahead_days)

        model_names = db.execute(select(Model.name).distinct()).scalars().all()
        summary = {"checked": []}
        for name in model_names:
            result = run_checks_for_model(db, name)
            summary["checked"].append(result)
        return summary
    finally:
        db.close()


async def run_forever() -> None:
    settings = get_settings()
    configure_mlflow()
    logger.info("Evaluator loop starting, interval=%ss", settings.evaluator_interval_seconds)
    while True:
        try:
            summary = await asyncio.to_thread(run_once)
            logger.info("Evaluation pass complete: %d model(s) checked", len(summary["checked"]))
        except Exception:
            logger.exception("Evaluation pass failed")
        await asyncio.sleep(settings.evaluator_interval_seconds)
