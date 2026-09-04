"""Shared test fixtures.

Unit tests never touch a database. Integration tests use a real Postgres (and, where relevant,
a real Redis / Kafka) rather than mocks - they read DATABASE_URL / REDIS_URL / KAFKA_BOOTSTRAP_
SERVERS and skip automatically if the corresponding service isn't reachable, so `pytest` still
runs clean on a laptop with nothing running, while CI (which starts real service containers)
runs everything.
"""

import os

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://inferops:inferops@localhost:5432/inferops_test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
# Isolated from the "inferops-consumer" group the real `consumer` service uses, so running
# tests against a live `docker compose up` stack doesn't compete for partition assignment with
# it (which would send a test's messages to the *live* service's database instead of the
# test's, and make test_consumer.py hang waiting for a row that never arrives there).
os.environ.setdefault("KAFKA_CONSUMER_GROUP", "inferops-consumer-test")
os.environ.setdefault("MODEL_ARTIFACT_DIR", os.path.join(os.path.dirname(__file__), "_artifacts"))

import numpy as np
import pandas as pd
import pytest

from app.ml.label_fn import sample_labels
from app.ml.schema import DEVICES, FEATURE_ORDER, ITEM_CATEGORIES


def make_synthetic_frame(n: int = 300, *, drift: bool = False, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "user_age": rng.integers(18, 70, n).astype(float),
            "user_tenure_days": rng.integers(0, 2000, n).astype(float),
            "user_avg_session_min": rng.uniform(1, 60, n),
            "user_category_affinity": rng.uniform(0, 1, n),
            "item_price": rng.uniform(5, 300, n),
            "item_popularity": rng.uniform(0, 1, n),
            "item_category": rng.choice(ITEM_CATEGORIES, n),
            "hour_of_day": rng.integers(0, 24, n),
            "history_click_rate": rng.uniform(0, 1, n),
            "device": rng.choice(DEVICES, n),
        }
    )
    df["clicked"] = sample_labels(df, drift=drift, rng=rng)
    return df[[*FEATURE_ORDER, "clicked"]]


@pytest.fixture(scope="session")
def database_url() -> str:
    return os.environ["DATABASE_URL"]


@pytest.fixture(scope="session")
def db_engine(database_url: str):
    from sqlalchemy import create_engine, text

    engine = create_engine(database_url)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip(f"No reachable Postgres at {database_url}; skipping integration tests.")

    from alembic import command
    from alembic.config import Config

    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alembic_cfg = Config(os.path.join(backend_dir, "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(alembic_cfg, "head")

    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    from sqlalchemy import text
    from sqlalchemy.orm import sessionmaker

    session_factory = sessionmaker(bind=db_engine, future=True)
    session = session_factory()
    yield session
    session.rollback()
    for table in ("retraining_jobs", "data_drift_metrics", "inferences", "deployments", "models"):
        session.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
    session.commit()
    session.close()


@pytest.fixture(scope="session")
def redis_url() -> str:
    return os.environ["REDIS_URL"]


@pytest.fixture
def redis_client(redis_url: str):
    import redis as redis_sync

    try:
        sync_client = redis_sync.from_url(redis_url, decode_responses=True)
        sync_client.ping()
    except Exception:
        pytest.skip(f"No reachable Redis at {redis_url}; skipping.")

    from redis.asyncio import Redis

    client = Redis.from_url(redis_url, decode_responses=True)
    yield client

    # Flush/close via the sync client, not the async one: the async client's connection can end
    # up bound to an event loop owned by FastAPI's TestClient (via its anyio portal), which is
    # already closed by the time this fixture tears down - reusing it from a fresh asyncio.run()
    # here raises "Event loop is closed" deep in its transport instead of cleaning up.
    sync_client.flushdb()
    sync_client.close()


class FakeProducer:
    """Records events in memory instead of talking to Kafka - used for API-route tests that
    exercise real Postgres/Redis but don't need to exercise the Kafka pipeline itself (that's
    covered separately by tests/integration/test_consumer.py against a real broker)."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    async def publish_inference_event(self, event: dict) -> None:
        self.events.append(event)

    async def publish_many(self, events: list[dict], **_kwargs) -> None:
        self.events.extend(events)


@pytest.fixture
def fake_producer() -> FakeProducer:
    return FakeProducer()


@pytest.fixture
def trained_model(db_session):
    """Trains a small real (not mocked) model against synthetic data and registers it, for
    tests that exercise the actual prediction/evaluation pipeline end to end."""

    from app.ml.training import train_and_register

    df = make_synthetic_frame(n=300, seed=1)
    model = train_and_register(
        db_session, name="ctr-recommender", version="2.1-test", algo="logistic_regression", df=df, status="production"
    )
    db_session.commit()
    return model


@pytest.fixture
def active_deployment(db_session, trained_model):
    from app.core.models import Deployment

    deployment = Deployment(model_id=trained_model.id, environment="production", traffic_split="1.0", is_active=True)
    db_session.add(deployment)
    db_session.commit()
    db_session.refresh(deployment)
    return deployment


@pytest.fixture
def client(db_session, redis_client, fake_producer):
    from fastapi.testclient import TestClient

    from app.api.deps import get_producer, get_redis
    from app.core.db import get_db
    from app.main import app

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_redis] = lambda: redis_client
    app.dependency_overrides[get_producer] = lambda: fake_producer

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
