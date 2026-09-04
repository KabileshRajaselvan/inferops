import io
import time
import uuid
from datetime import UTC, date, datetime

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_producer, get_registry
from app.core.api_base import APIModel
from app.core.db import get_db
from app.core.kafka import KafkaProducerClient
from app.core.models import Deployment, Model
from app.metrics import batch_predict_rows_total
from app.ml.registry import ModelRegistry, predict_many
from app.ml.schema import FEATURE_ORDER

router = APIRouter(prefix="/api/v1", tags=["batch-predict"])


class BatchPredictSummary(APIModel):
    model_name: str
    model_version: str
    deployment_id: str
    rows: int
    click_rate: float
    total_latency_ms: int
    rows_per_second: float


def _current_champion_deployment(db: Session, model_name: str) -> Deployment:
    deployment = (
        db.execute(
            select(Deployment)
            .join(Model)
            .where(Model.name == model_name, Deployment.environment == "production", Deployment.is_active.is_(True))
            .order_by(Deployment.traffic_split.desc())
        )
        .scalars()
        .first()
    )
    if deployment is None:
        raise HTTPException(status_code=404, detail=f"No active production deployment for '{model_name}'")
    return deployment


@router.post("/batch-predict", response_model=BatchPredictSummary)
async def batch_predict(
    model_name: str,
    file: UploadFile = File(..., description="CSV with one row per prediction, columns matching the feature schema"),
    db: Session = Depends(get_db),
    producer: KafkaProducerClient = Depends(get_producer),
    model_registry: ModelRegistry = Depends(get_registry),
) -> BatchPredictSummary:
    start = time.perf_counter()

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"CSV missing required columns: {missing}")
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV has no rows")

    deployment = _current_champion_deployment(db, model_name)
    model_row = db.get(Model, deployment.model_id)
    if model_row is None:
        raise HTTPException(status_code=500, detail="Deployment references a missing model")

    pipeline = model_registry.get_pipeline(model_row.id, model_row.artifact_path)
    scores = predict_many(pipeline, df)
    labels = scores.apply(lambda s: "click" if s >= 0.5 else "no_click")

    today = date.today().isoformat()
    now = datetime.now(UTC).isoformat()
    events = [
        {
            "id": str(uuid.uuid4()),
            "inference_date": today,
            "model_id": str(model_row.id),
            "model_name": model_row.name,
            "model_version": model_row.version,
            "input_features": row[FEATURE_ORDER].to_dict(),
            "predicted_output": {"score": round(float(scores.loc[idx]), 4), "label": labels.loc[idx]},
            "latency_ms": None,
            "created_at": now,
        }
        for idx, row in df.iterrows()
    ]
    await producer.publish_many(events)

    total_latency_ms = int((time.perf_counter() - start) * 1000)
    batch_predict_rows_total.inc(len(df))

    return BatchPredictSummary(
        model_name=model_row.name,
        model_version=model_row.version,
        deployment_id=str(deployment.id),
        rows=len(df),
        click_rate=round(float((labels == "click").mean()), 4),
        total_latency_ms=total_latency_ms,
        rows_per_second=round(len(df) / max(total_latency_ms / 1000, 1e-6), 1),
    )
