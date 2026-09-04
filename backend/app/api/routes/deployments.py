import uuid
from datetime import datetime
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.api_base import APIModel
from app.core.db import get_db
from app.core.models import Deployment, Model

router = APIRouter(prefix="/api/v1/deployments", tags=["deployments"])


class DeploymentOut(APIModel):
    id: str
    model_id: str
    model_name: str
    model_version: str
    environment: str
    traffic_split: float
    is_active: bool
    deployed_at: datetime

    @classmethod
    def from_row(cls, dep: Deployment, model: Model) -> "DeploymentOut":
        return cls(
            id=str(dep.id),
            model_id=str(dep.model_id),
            model_name=model.name,
            model_version=model.version,
            environment=dep.environment,
            traffic_split=float(dep.traffic_split),
            is_active=dep.is_active,
            deployed_at=dep.deployed_at,
        )


class CreateDeploymentRequest(APIModel):
    model_id: str
    environment: str = "production"
    traffic_split: float = Field(ge=0, le=1)


class UpdateDeploymentRequest(APIModel):
    traffic_split: float | None = Field(default=None, ge=0, le=1)
    is_active: bool | None = None


@router.get("", response_model=list[DeploymentOut])
def list_deployments(model_name: str | None = None, db: Session = Depends(get_db)) -> list[DeploymentOut]:
    stmt = select(Deployment, Model).join(Model)
    if model_name:
        stmt = stmt.where(Model.name == model_name)
    rows = db.execute(stmt.order_by(Deployment.deployed_at.desc())).all()
    return [DeploymentOut.from_row(dep, model) for dep, model in rows]


@router.post("", response_model=DeploymentOut, status_code=201)
def create_deployment(payload: CreateDeploymentRequest, db: Session = Depends(get_db)) -> DeploymentOut:
    try:
        model_uuid = uuid.UUID(payload.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="model_id must be a UUID") from exc

    model = db.get(Model, model_uuid)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {payload.model_id} not found")

    deployment = Deployment(
        model_id=model.id,
        environment=payload.environment,
        traffic_split=Decimal(str(payload.traffic_split)),
        is_active=True,
    )
    db.add(deployment)
    db.commit()
    db.refresh(deployment)
    return DeploymentOut.from_row(deployment, model)


@router.patch("/{deployment_id}", response_model=DeploymentOut)
def update_deployment(deployment_id: str, payload: UpdateDeploymentRequest, db: Session = Depends(get_db)) -> DeploymentOut:
    try:
        dep_uuid = uuid.UUID(deployment_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="deployment_id must be a UUID") from exc

    deployment = db.get(Deployment, dep_uuid)
    if deployment is None:
        raise HTTPException(status_code=404, detail=f"Deployment {deployment_id} not found")

    if payload.traffic_split is not None:
        deployment.traffic_split = Decimal(str(payload.traffic_split))
    if payload.is_active is not None:
        deployment.is_active = payload.is_active

    db.commit()
    db.refresh(deployment)
    model = db.get(Model, deployment.model_id)
    return DeploymentOut.from_row(deployment, model)
