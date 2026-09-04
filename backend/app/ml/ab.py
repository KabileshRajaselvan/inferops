import random
import uuid
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class DeploymentChoice:
    deployment_id: uuid.UUID
    model_id: uuid.UUID


def select_deployment(deployments: list) -> "DeploymentChoice | None":
    """Weighted-random pick across active deployments by traffic_split. Deployments is a list
    of ORM Deployment rows (duck-typed here to keep this pure/unit-testable)."""

    active = [d for d in deployments if d.is_active]
    if not active:
        return None

    weights = [float(d.traffic_split) for d in active]
    total = sum(weights)
    if total <= 0:
        chosen = active[0]
    else:
        normalized = [w / total for w in weights]
        chosen = random.choices(active, weights=normalized, k=1)[0]

    return DeploymentChoice(deployment_id=chosen.id, model_id=chosen.model_id)


def normalize_split(split: Decimal | float) -> Decimal:
    return Decimal(str(round(float(split), 2)))
