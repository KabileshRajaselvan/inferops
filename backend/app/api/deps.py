from redis.asyncio import Redis

from app.core.kafka import KafkaProducerClient, producer_client
from app.core.redis_client import get_redis as _get_redis
from app.ml.registry import ModelRegistry, registry


def get_redis() -> Redis:
    return _get_redis()


def get_producer() -> KafkaProducerClient:
    return producer_client


def get_registry() -> ModelRegistry:
    return registry
