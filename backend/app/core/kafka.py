import asyncio
import json
import logging
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from app.config import get_settings

logger = logging.getLogger("inferops.kafka")


async def _await_all(pending: list) -> None:
    if not pending:
        return
    try:
        await asyncio.gather(*pending)
    except Exception:
        logger.exception("Failed to publish one or more batched inference events to Kafka")


def _serialize(value: dict[str, Any]) -> bytes:
    return json.dumps(value, default=str).encode("utf-8")


def _deserialize(value: bytes) -> dict[str, Any]:
    return json.loads(value.decode("utf-8"))


class KafkaProducerClient:
    """Thin wrapper so the API can publish inference events fire-and-forget without
    blocking the request on broker acknowledgement of every single message."""

    def __init__(self) -> None:
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        settings = get_settings()
        self._producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=_serialize,
            acks=1,
            linger_ms=20,
        )
        await self._producer.start()

    async def stop(self) -> None:
        if self._producer is not None:
            await self._producer.stop()

    async def publish_inference_event(self, event: dict[str, Any]) -> None:
        if self._producer is None:
            logger.warning("Kafka producer not started; dropping event")
            return
        settings = get_settings()
        try:
            await self._producer.send(settings.kafka_inference_topic, value=event)
        except Exception:
            logger.exception("Failed to publish inference event to Kafka")

    async def publish_many(self, events: list[dict[str, Any]], *, chunk_size: int = 2000) -> None:
        """Publishes a large batch of events without awaiting each send individually - lets
        the producer's internal batching (linger_ms) do the work, only syncing every
        `chunk_size` sends so pending futures don't grow unbounded for very large batches."""

        if self._producer is None:
            logger.warning("Kafka producer not started; dropping %d events", len(events))
            return
        settings = get_settings()
        pending = []
        for event in events:
            pending.append(self._producer.send(settings.kafka_inference_topic, value=event))
            if len(pending) >= chunk_size:
                await _await_all(pending)
                pending = []
        await _await_all(pending)


def build_consumer() -> AIOKafkaConsumer:
    settings = get_settings()
    return AIOKafkaConsumer(
        settings.kafka_inference_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group,
        value_deserializer=_deserialize,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
    )


producer_client = KafkaProducerClient()
