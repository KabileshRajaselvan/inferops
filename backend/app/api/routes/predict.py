import time
import uuid
from datetime import UTC, date, datetime

from fastapi import APIRouter, Depends, HTTPException
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_producer, get_redis, get_registry
from app.config import get_settings
from app.core.db import get_db
from app.core.kafka import KafkaProducerClient
from app.core.models import Deployment, Model
from app.metrics import cache_hits_total, cache_misses_total, prediction_latency_ms, predictions_total
from app.ml.ab import select_deployment
from app.ml.cache_keys import hash_features
from app.ml.registry import ModelRegistry, predict_one
from app.ml.schema import Prediction, PredictRequest, PredictResponse

router = APIRouter(prefix="/api/v1", tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    db: Session = Depends(get_db),
    redis: Redis = Depends(get_redis),
    producer: KafkaProducerClient = Depends(get_producer),
    model_registry: ModelRegistry = Depends(get_registry),
) -> PredictResponse:
    start = time.perf_counter()
    cache_key = hash_features(request.model_name, request.features)

    cached = await redis.get(cache_key)
    if cached is not None:
        cache_hits_total.inc()
        response = PredictResponse.model_validate_json(cached)
        response.latency_ms = int((time.perf_counter() - start) * 1000)
        response.cache_hit = True
        return response

    cache_misses_total.inc()

    deployments = (
        db.execute(
            select(Deployment)
            .join(Model)
            .where(Model.name == request.model_name, Deployment.environment == "production")
            .options(selectinload(Deployment.model))
        )
        .scalars()
        .all()
    )
    choice = select_deployment(list(deployments))
    if choice is None:
        raise HTTPException(status_code=404, detail=f"No active production deployment for '{request.model_name}'")

    model_row = db.get(Model, choice.model_id)
    if model_row is None:
        raise HTTPException(status_code=500, detail="Deployment references a missing model")

    pipeline = model_registry.get_pipeline(model_row.id, model_row.artifact_path)
    score = predict_one(pipeline, request.features)
    label = "click" if score >= 0.5 else "no_click"

    latency_ms = int((time.perf_counter() - start) * 1000)
    inference_id = str(uuid.uuid4())

    response = PredictResponse(
        inference_id=inference_id,
        prediction=Prediction(score=round(score, 4), label=label),
        model_version=model_row.version,
        latency_ms=latency_ms,
        deployment_id=str(choice.deployment_id),
        cache_hit=False,
    )

    settings = get_settings()
    await redis.set(cache_key, response.model_dump_json(), ex=settings.cache_ttl_seconds)

    predictions_total.labels(model_name=model_row.name, model_version=model_row.version).inc()
    prediction_latency_ms.labels(model_name=model_row.name).observe(latency_ms)

    await producer.publish_inference_event(
        {
            "id": inference_id,
            "inference_date": date.today().isoformat(),
            "model_id": str(model_row.id),
            "model_name": model_row.name,
            "model_version": model_row.version,
            "input_features": request.features.model_dump(),
            "predicted_output": response.prediction.model_dump(),
            "latency_ms": latency_ms,
            "created_at": datetime.now(UTC).isoformat(),
        }
    )

    return response
