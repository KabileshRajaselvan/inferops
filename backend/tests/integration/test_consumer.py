"""Exercises the real Kafka -> consumer -> Postgres pipeline end to end (not the Postgres-only
`db_session` fixture in isolation) - skips if no Kafka broker is reachable. This is the one
place the Kafka design decision (see README Trade-offs) actually gets tested for real, since
the API-level tests stub the producer to keep those tests focused on request/response behavior."""

import asyncio
import uuid
from datetime import UTC, date, datetime

import pytest
from sqlalchemy import select

from app.config import get_settings
from app.core.kafka import KafkaProducerClient


@pytest.fixture
async def kafka_producer():
    settings = get_settings()
    producer = KafkaProducerClient()
    try:
        await asyncio.wait_for(producer.start(), timeout=5)
    except Exception:
        pytest.skip(f"No reachable Kafka at {settings.kafka_bootstrap_servers}; skipping.")
    yield producer
    await producer.stop()


async def test_published_events_land_in_postgres_via_the_consumer(db_session, kafka_producer, trained_model):
    from app.consumer import run_forever
    from app.core.models import Inference

    event_id = str(uuid.uuid4())
    event = {
        "id": event_id,
        "inference_date": date.today().isoformat(),
        "model_id": str(trained_model.id),
        "model_name": trained_model.name,
        "model_version": trained_model.version,
        "input_features": {"user_age": 30},
        "predicted_output": {"score": 0.7, "label": "click"},
        "latency_ms": 15,
        "created_at": datetime.now(UTC).isoformat(),
    }
    await kafka_producer.publish_inference_event(event)

    consumer_task = asyncio.create_task(run_forever())
    try:
        # Generous budget: a fresh consumer group join (coordinator discovery + rebalance)
        # can itself take several seconds on top of the 2s flush interval, especially the
        # first time this group id is used.
        for _ in range(40):  # poll for up to ~40s
            await asyncio.sleep(1.0)
            row = db_session.execute(select(Inference).where(Inference.id == uuid.UUID(event_id))).scalar_one_or_none()
            if row is not None:
                break
        else:
            row = None
    finally:
        consumer_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer_task

    assert row is not None, "event was not consumed into Postgres within the timeout"
    assert row.model_name == trained_model.name
    assert row.predicted_output == {"score": 0.7, "label": "click"}
