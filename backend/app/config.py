from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    # Database
    database_url: str = "postgresql+psycopg://inferops:inferops@localhost:5432/inferops"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 300

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_inference_topic: str = "inference-events"
    kafka_consumer_group: str = "inferops-consumer"
    consumer_batch_size: int = 500
    consumer_flush_interval_seconds: float = 2.0

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "ctr-recommender"

    # Model artifacts (shared volume in Docker; local dir when run from repo root)
    model_artifact_dir: str = "./model-artifacts"

    # Evaluator
    evaluator_interval_seconds: int = 60
    drift_lookback_hours: int = 24
    accuracy_lookback_hours: int = 24
    accuracy_drop_threshold: float = 0.02
    retraining_cooldown_minutes: int = 15
    drift_score_threshold: float = 0.15
    partition_lookahead_days: int = 3
    evaluator_metrics_port: int = 9100

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
