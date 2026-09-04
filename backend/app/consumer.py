"""Entrypoint for the `consumer` service: `python -m app.consumer`.

Reads inference events off the Kafka topic and micro-batches them into bulk inserts against
the partitioned `inferences` table - this is what keeps 100K-row batch-predict runs from
hammering Postgres with one insert per row (see README Trade-offs, decision on Kafka vs.
direct DB writes)."""

import asyncio
import logging
import uuid
from datetime import date, datetime

from sqlalchemy.dialects.postgresql import insert

from app.config import get_settings
from app.core.db import SessionLocal
from app.core.kafka import build_consumer
from app.core.models import Inference
from app.evaluation.partitions import ensure_partitions

logging.basicConfig(level=get_settings().log_level)
logger = logging.getLogger("inferops.consumer")


def _prepare_row(event: dict) -> dict:
    return {
        "id": uuid.UUID(event["id"]),
        "inference_date": date.fromisoformat(event["inference_date"]),
        "model_id": uuid.UUID(event["model_id"]),
        "model_name": event["model_name"],
        "model_version": event["model_version"],
        "input_features": event["input_features"],
        "predicted_output": event["predicted_output"],
        "latency_ms": event.get("latency_ms"),
        "created_at": datetime.fromisoformat(event["created_at"]),
    }


def flush_batch(events: list[dict]) -> None:
    """Bulk-inserts a batch of events, idempotently: Kafka only guarantees at-least-once
    delivery (a redelivered message after a missed offset commit is expected, not exceptional),
    so a duplicate `id` is silently ignored rather than aborting the whole batch's insert."""

    if not events:
        return
    rows = [_prepare_row(e) for e in events]
    db = SessionLocal()
    try:
        stmt = insert(Inference).values(rows).on_conflict_do_nothing(index_elements=["id", "inference_date"])
        db.execute(stmt)
        db.commit()
        logger.info("Flushed %d inference events to Postgres", len(rows))
    except Exception:
        db.rollback()
        logger.exception("Failed to flush batch of %d events", len(rows))
    finally:
        db.close()


async def run_forever() -> None:
    settings = get_settings()

    db = SessionLocal()
    try:
        ensure_partitions(db, lookahead_days=settings.partition_lookahead_days)
    finally:
        db.close()

    consumer = build_consumer()
    await consumer.start()
    logger.info("Kafka consumer started on topic=%s", settings.kafka_inference_topic)

    buffer: list[dict] = []
    loop = asyncio.get_event_loop()
    last_flush = loop.time()

    try:
        while True:
            try:
                msg = await asyncio.wait_for(consumer.getone(), timeout=settings.consumer_flush_interval_seconds)
                buffer.append(msg.value)
            except TimeoutError:
                pass

            now = loop.time()
            due = buffer and (
                len(buffer) >= settings.consumer_batch_size
                or now - last_flush >= settings.consumer_flush_interval_seconds
            )
            if due:
                await asyncio.to_thread(flush_batch, buffer)
                await consumer.commit()
                buffer = []
                last_flush = now
    finally:
        if buffer:
            await asyncio.to_thread(flush_batch, buffer)
            await consumer.commit()
        await consumer.stop()


if __name__ == "__main__":
    asyncio.run(run_forever())
