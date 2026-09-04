import uuid
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal

from app.ml.ab import normalize_split, select_deployment


@dataclass
class FakeDeployment:
    id: uuid.UUID
    model_id: uuid.UUID
    traffic_split: Decimal
    is_active: bool = True


def test_select_deployment_returns_none_when_no_active_deployments():
    assert select_deployment([]) is None
    assert select_deployment([FakeDeployment(uuid.uuid4(), uuid.uuid4(), Decimal("1.0"), is_active=False)]) is None


def test_select_deployment_skips_inactive():
    active = FakeDeployment(uuid.uuid4(), uuid.uuid4(), Decimal("1.0"), is_active=True)
    inactive = FakeDeployment(uuid.uuid4(), uuid.uuid4(), Decimal("1.0"), is_active=False)
    for _ in range(20):
        choice = select_deployment([active, inactive])
        assert choice.deployment_id == active.id


def test_select_deployment_roughly_matches_traffic_split():
    model_a = uuid.uuid4()
    model_b = uuid.uuid4()
    deployment_a = FakeDeployment(uuid.uuid4(), model_a, Decimal("0.8"))
    deployment_b = FakeDeployment(uuid.uuid4(), model_b, Decimal("0.2"))

    counts = Counter()
    n = 5000
    for _ in range(n):
        choice = select_deployment([deployment_a, deployment_b])
        counts[choice.model_id] += 1

    ratio_a = counts[model_a] / n
    assert 0.75 < ratio_a < 0.85


def test_normalize_split_rounds_to_two_decimals():
    assert normalize_split(0.123456) == Decimal("0.12")
    assert normalize_split(Decimal("0.8")) == Decimal("0.8")
