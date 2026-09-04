from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.core.api_base import APIModel
from app.core.db import get_db
from app.core.models import DataDriftMetric, Deployment, Inference, Model, RetrainingJob

router = APIRouter(prefix="/api/v1/models", tags=["models"])


class ModelSummary(APIModel):
    id: str
    name: str
    version: str
    model_type: str
    framework: str
    status: str
    metrics: dict
    mlflow_run_id: str | None
    created_at: datetime

    @classmethod
    def from_row(cls, row: Model) -> "ModelSummary":
        return cls(
            id=str(row.id),
            name=row.name,
            version=row.version,
            model_type=row.model_type,
            framework=row.framework,
            status=row.status,
            metrics={k: v for k, v in row.metrics.items() if k != "feature_baseline"},
            mlflow_run_id=row.mlflow_run_id,
            created_at=row.created_at,
        )


class DailyMetric(APIModel):
    day: date
    prediction_count: int
    avg_latency_ms: float | None
    p95_latency_ms: float | None
    click_rate: float | None
    accuracy: float | None
    labeled_count: int


class DriftPoint(APIModel):
    feature_name: str
    drift_score: float
    checked_at: datetime


class DeploymentInfo(APIModel):
    id: str
    model_version: str
    traffic_split: float
    is_active: bool


class RetrainingJobInfo(APIModel):
    id: str
    trigger_reason: str
    status: str
    new_version: str | None
    started_at: datetime | None
    completed_at: datetime | None


class ModelMetricsResponse(APIModel):
    model_name: str
    start_date: date
    end_date: date
    daily: list[DailyMetric]
    drift: list[DriftPoint]
    deployments: list[DeploymentInfo]
    retraining_jobs: list[RetrainingJobInfo]


@router.get("", response_model=list[ModelSummary])
def list_models(db: Session = Depends(get_db)) -> list[ModelSummary]:
    rows = db.execute(select(Model).order_by(Model.name, Model.created_at.desc())).scalars().all()
    return [ModelSummary.from_row(r) for r in rows]


@router.get("/{name}", response_model=list[ModelSummary])
def get_model_versions(name: str, db: Session = Depends(get_db)) -> list[ModelSummary]:
    rows = db.execute(select(Model).where(Model.name == name).order_by(Model.created_at.desc())).scalars().all()
    if not rows:
        raise HTTPException(status_code=404, detail=f"No model registered under '{name}'")
    return [ModelSummary.from_row(r) for r in rows]


@router.get("/{name}/metrics", response_model=ModelMetricsResponse)
def get_model_metrics(
    name: str,
    start_date: date = Query(default_factory=lambda: date.today() - timedelta(days=7)),
    end_date: date = Query(default_factory=date.today),
    db: Session = Depends(get_db),
) -> ModelMetricsResponse:
    model_ids = db.execute(select(Model.id).where(Model.name == name)).scalars().all()
    if not model_ids:
        raise HTTPException(status_code=404, detail=f"No model registered under '{name}'")

    is_click = Inference.predicted_output["label"].astext == "click"
    is_correct = Inference.predicted_output["label"].astext == Inference.actual_output["label"].astext

    daily_rows = db.execute(
        select(
            Inference.inference_date.label("day"),
            func.count().label("prediction_count"),
            func.avg(Inference.latency_ms).label("avg_latency_ms"),
            func.percentile_cont(0.95).within_group(Inference.latency_ms.asc()).label("p95_latency_ms"),
            func.avg(case((is_click, 1.0), else_=0.0)).label("click_rate"),
            func.avg(case((is_correct, 1.0), else_=0.0)).filter(Inference.actual_output.isnot(None)).label(
                "accuracy"
            ),
            func.count().filter(Inference.actual_output.isnot(None)).label("labeled_count"),
        )
        .where(
            Inference.model_id.in_(model_ids),
            Inference.inference_date >= start_date,
            Inference.inference_date <= end_date,
        )
        .group_by(Inference.inference_date)
        .order_by(Inference.inference_date)
    ).all()

    daily = [
        DailyMetric(
            day=r.day,
            prediction_count=r.prediction_count,
            avg_latency_ms=round(r.avg_latency_ms, 2) if r.avg_latency_ms is not None else None,
            p95_latency_ms=round(r.p95_latency_ms, 2) if r.p95_latency_ms is not None else None,
            click_rate=round(r.click_rate, 4) if r.click_rate is not None else None,
            accuracy=round(r.accuracy, 4) if r.accuracy is not None else None,
            labeled_count=r.labeled_count,
        )
        for r in daily_rows
    ]

    drift_rows = db.execute(
        select(DataDriftMetric)
        .where(DataDriftMetric.model_id.in_(model_ids))
        .order_by(DataDriftMetric.checked_at.desc())
        .limit(50)
    ).scalars().all()
    drift = [
        DriftPoint(feature_name=d.feature_name, drift_score=float(d.drift_score), checked_at=d.checked_at)
        for d in drift_rows
    ]

    deployment_rows = db.execute(
        select(Deployment, Model.version)
        .join(Model)
        .where(Model.id.in_(model_ids), Deployment.environment == "production")
    ).all()
    deployments = [
        DeploymentInfo(
            id=str(dep.id), model_version=version, traffic_split=float(dep.traffic_split), is_active=dep.is_active
        )
        for dep, version in deployment_rows
    ]

    job_rows = db.execute(
        select(RetrainingJob).where(RetrainingJob.model_id.in_(model_ids)).order_by(RetrainingJob.started_at.desc())
    ).scalars().all()
    retraining_jobs = [
        RetrainingJobInfo(
            id=str(j.id),
            trigger_reason=j.trigger_reason,
            status=j.status,
            new_version=j.new_version,
            started_at=j.started_at,
            completed_at=j.completed_at,
        )
        for j in job_rows
    ]

    return ModelMetricsResponse(
        model_name=name,
        start_date=start_date,
        end_date=end_date,
        daily=daily,
        drift=drift,
        deployments=deployments,
        retraining_jobs=retraining_jobs,
    )
