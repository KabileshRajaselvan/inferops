import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, Numeric, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base


class Model(Base):
    """A trained model version in the registry. `metrics` JSONB holds training/eval metrics
    plus the per-feature baseline histogram used later for drift comparison."""

    __tablename__ = "models"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_models_name_version"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50))
    framework: Mapped[str] = mapped_column(String(50))
    artifact_path: Mapped[str] = mapped_column(String(500))
    input_shape: Mapped[dict] = mapped_column(JSONB, default=dict)
    output_shape: Mapped[dict] = mapped_column(JSONB, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    status: Mapped[str] = mapped_column(String(50), default="staging")

    deployments: Mapped[list["Deployment"]] = relationship(back_populates="model", cascade="all, delete-orphan")


class Deployment(Base):
    """An A/B test variant: a model version live at some traffic_split within an environment."""

    __tablename__ = "deployments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("models.id", ondelete="CASCADE"))
    environment: Mapped[str] = mapped_column(String(50), default="production")
    traffic_split: Mapped[Decimal] = mapped_column(Numeric(3, 2), default=Decimal("1.0"))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    deployed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    model: Mapped["Model"] = relationship(back_populates="deployments")


class Inference(Base):
    """Event log of every prediction. Partitioned by inference_date (see the initial Alembic
    migration) - the partition key must be part of the primary key in Postgres, hence the
    composite PK. Rows normally arrive via the Kafka consumer, not written directly by the API."""

    __tablename__ = "inferences"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    inference_date: Mapped[date] = mapped_column(Date, primary_key=True)
    model_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    model_name: Mapped[str] = mapped_column(String(255))
    model_version: Mapped[str] = mapped_column(String(50))
    input_features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    predicted_output: Mapped[dict] = mapped_column(JSONB, nullable=False)
    actual_output: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DataDriftMetric(Base):
    __tablename__ = "data_drift_metrics"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("models.id", ondelete="CASCADE"))
    feature_name: Mapped[str] = mapped_column(String(255))
    expected_distribution: Mapped[dict] = mapped_column(JSONB)
    actual_distribution: Mapped[dict] = mapped_column(JSONB)
    drift_score: Mapped[Decimal] = mapped_column(Numeric(6, 4))
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RetrainingJob(Base):
    __tablename__ = "retraining_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("models.id", ondelete="CASCADE"))
    trigger_reason: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50), default="pending")
    new_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    new_model_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("models.id"), nullable=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
