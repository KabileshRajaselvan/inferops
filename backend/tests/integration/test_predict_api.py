"""Predict API tests against a real Postgres + Redis. Skips automatically if either is
unreachable (see tests/conftest.py). Kafka is stubbed here via fake_producer - the real
Kafka -> consumer -> Postgres path is covered separately in test_consumer.py."""

SAMPLE_FEATURES = {
    "user_age": 34,
    "user_tenure_days": 220,
    "user_avg_session_min": 12.5,
    "user_category_affinity": 0.62,
    "item_price": 49.99,
    "item_popularity": 0.71,
    "item_category": "electronics",
    "hour_of_day": 14,
    "history_click_rate": 0.35,
    "device": "mobile",
}


def test_predict_returns_a_scored_prediction(client, active_deployment, trained_model):
    response = client.post(
        "/api/v1/predict", json={"model_name": "ctr-recommender", "features": SAMPLE_FEATURES}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0.0 <= body["prediction"]["score"] <= 1.0
    assert body["prediction"]["label"] in ("click", "no_click")
    assert body["model_version"] == trained_model.version
    assert body["deployment_id"] == str(active_deployment.id)
    assert body["cache_hit"] is False


def test_predict_second_identical_request_is_a_cache_hit(client, active_deployment, trained_model, fake_producer):
    first = client.post("/api/v1/predict", json={"model_name": "ctr-recommender", "features": SAMPLE_FEATURES})
    assert first.status_code == 200
    events_after_first = len(fake_producer.events)

    second = client.post("/api/v1/predict", json={"model_name": "ctr-recommender", "features": SAMPLE_FEATURES})
    assert second.status_code == 200
    assert second.json()["cache_hit"] is True
    assert second.json()["prediction"] == first.json()["prediction"]
    # cache hits never re-publish an inference event
    assert len(fake_producer.events) == events_after_first


def test_predict_publishes_an_inference_event_on_cache_miss(client, active_deployment, trained_model, fake_producer):
    response = client.post("/api/v1/predict", json={"model_name": "ctr-recommender", "features": SAMPLE_FEATURES})
    assert response.status_code == 200

    assert len(fake_producer.events) == 1
    event = fake_producer.events[0]
    assert event["model_name"] == "ctr-recommender"
    assert event["id"] == response.json()["inference_id"]


def test_predict_returns_404_when_no_deployment_exists(client):
    response = client.post("/api/v1/predict", json={"model_name": "unknown-model", "features": SAMPLE_FEATURES})
    assert response.status_code == 404


def test_predict_rejects_invalid_features(client, active_deployment):
    bad_features = {**SAMPLE_FEATURES, "item_category": "not-a-real-category"}
    response = client.post("/api/v1/predict", json={"model_name": "ctr-recommender", "features": bad_features})
    assert response.status_code == 422


def test_predict_splits_traffic_across_two_deployments_roughly_as_configured(db_session, client, trained_model):
    from app.core.models import Deployment
    from app.ml.training import train_and_register
    from tests.conftest import make_synthetic_frame

    model_b = train_and_register(
        db_session,
        name="ctr-recommender",
        version="2.2-test",
        algo="gradient_boosting",
        df=make_synthetic_frame(n=300, seed=2),
        status="production",
    )
    db_session.add_all(
        [
            Deployment(model_id=trained_model.id, environment="production", traffic_split="0.8", is_active=True),
            Deployment(model_id=model_b.id, environment="production", traffic_split="0.2", is_active=True),
        ]
    )
    db_session.commit()

    versions_seen = []
    for i in range(200):
        features = {**SAMPLE_FEATURES, "hour_of_day": i % 24}  # vary to avoid cache hits
        response = client.post("/api/v1/predict", json={"model_name": "ctr-recommender", "features": features})
        assert response.status_code == 200
        versions_seen.append(response.json()["model_version"])

    share_a = versions_seen.count(trained_model.version) / len(versions_seen)
    assert 0.65 < share_a < 0.95
