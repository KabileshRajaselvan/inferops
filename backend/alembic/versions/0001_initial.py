"""initial schema: models, deployments, inferences (partitioned), data_drift_metrics, retraining_jobs

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-04
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "models",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("model_type", sa.String(50), nullable=True),
        sa.Column("framework", sa.String(50), nullable=True),
        sa.Column("artifact_path", sa.String(500), nullable=True),
        sa.Column("input_shape", postgresql.JSONB, server_default="{}"),
        sa.Column("output_shape", postgresql.JSONB, server_default="{}"),
        sa.Column("metrics", postgresql.JSONB, server_default="{}"),
        sa.Column("mlflow_run_id", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("status", sa.String(50), server_default="staging"),
        sa.UniqueConstraint("name", "version", name="uq_models_name_version"),
    )

    op.create_table(
        "deployments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("model_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("models.id", ondelete="CASCADE"), nullable=False),
        sa.Column("environment", sa.String(50), server_default="production"),
        sa.Column("traffic_split", sa.Numeric(3, 2), server_default="1.0"),
        sa.Column("is_active", sa.Boolean, server_default=sa.true()),
        sa.Column("deployed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_deployments_model", "deployments", ["model_id"])

    # Partitioned by inference_date - the partition key must be part of every unique
    # constraint (including the PK) in Postgres, hence the composite primary key. No FK to
    # `models` here: high-volume, Kafka-consumer-written event tables are deliberately kept
    # loosely coupled from the registry so a transient lag/ordering issue on the consumer side
    # can never block an insert.
    op.create_table(
        "inferences",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("inference_date", sa.Date, nullable=False),
        sa.Column("model_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("input_features", postgresql.JSONB, nullable=False),
        sa.Column("predicted_output", postgresql.JSONB, nullable=False),
        sa.Column("actual_output", postgresql.JSONB, nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id", "inference_date"),
        postgresql_partition_by="RANGE (inference_date)",
    )
    op.create_index("idx_inferences_model_date", "inferences", ["model_id", sa.text("inference_date DESC")])
    op.create_index("idx_inferences_model_created", "inferences", ["model_id", "created_at"])
    op.create_index("idx_inferences_id", "inferences", ["id"])

    # Safety-net partition: catches any row whose inference_date falls outside the partitions
    # the evaluator/consumer have proactively created (see app/evaluation/partitions.py).
    op.execute("CREATE TABLE inferences_default PARTITION OF inferences DEFAULT")

    op.create_table(
        "data_drift_metrics",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("model_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("models.id", ondelete="CASCADE"), nullable=False),
        sa.Column("feature_name", sa.String(255), nullable=False),
        sa.Column("expected_distribution", postgresql.JSONB, nullable=False),
        sa.Column("actual_distribution", postgresql.JSONB, nullable=False),
        sa.Column("drift_score", sa.Numeric(6, 4), nullable=False),
        sa.Column("checked_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_drift_model_checked", "data_drift_metrics", ["model_id", "checked_at"])

    op.create_table(
        "retraining_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("model_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("models.id", ondelete="CASCADE"), nullable=False),
        sa.Column("trigger_reason", sa.String(255), nullable=False),
        sa.Column("status", sa.String(50), server_default="pending"),
        sa.Column("new_version", sa.String(50), nullable=True),
        sa.Column("new_model_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("models.id"), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_retraining_jobs_model", "retraining_jobs", ["model_id"])


def downgrade() -> None:
    op.drop_table("retraining_jobs")
    op.drop_table("data_drift_metrics")
    op.execute("DROP TABLE IF EXISTS inferences_default")
    op.drop_table("inferences")
    op.drop_table("deployments")
    op.drop_table("models")
