"""Exercises the honest accuracy/drift/retraining loop: feedback is recorded the same way a
real deployment would receive it (POST /feedback against a specific inference_id), and
/evaluate runs the same checks the evaluator's background loop runs."""

import uuid
from datetime import UTC, date, datetime

import pytest

from app.evaluation.retrain import MIN_LABELED_ROWS_FOR_RETRAIN
from app.ml.schema import PredictFeatures
from tests.conftest import make_synthetic_frame

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


def _insert_inference(db_session, model, *, predicted_label: str, actual_label: str | None, features: dict):
    from app.core.models import Inference

    inference = Inference(
        id=uuid.uuid4(),
        inference_date=date.today(),
        model_id=model.id,
        model_name=model.name,
        model_version=model.version,
        input_features=features,
        predicted_output={"score": 0.9 if predicted_label == "click" else 0.1, "label": predicted_label},
        actual_output={"label": actual_label} if actual_label else None,
        latency_ms=10,
        created_at=datetime.now(UTC),
    )
    db_session.add(inference)
    return inference


def test_feedback_records_ground_truth_on_the_matching_inference(db_session, client, trained_model):
    inference = _insert_inference(
        db_session, trained_model, predicted_label="click", actual_label=None, features=SAMPLE_FEATURES
    )
    db_session.commit()

    response = client.post(
        "/api/v1/feedback", json={"inference_id": str(inference.id), "actual_label": "click"}
    )
    assert response.status_code == 200
    assert response.json()["recorded"] is True

    db_session.refresh(inference)
    assert inference.actual_output == {"label": "click"}


def test_feedback_for_unknown_inference_returns_404(client):
    response = client.post(
        "/api/v1/feedback", json={"inference_id": str(uuid.uuid4()), "actual_label": "click"}
    )
    assert response.status_code == 404


def test_evaluate_computes_accuracy_from_labeled_feedback(db_session, client, trained_model, active_deployment):
    # 15 correct, 15 incorrect => live accuracy ~0.5, well below whatever the model scored on
    # its own holdout set (never < 0 drop, since accuracy_lookback picks these up).
    for _i in range(15):
        _insert_inference(
            db_session, trained_model, predicted_label="click", actual_label="click", features=SAMPLE_FEATURES
        )
    for _i in range(15):
        _insert_inference(
            db_session, trained_model, predicted_label="click", actual_label="no_click", features=SAMPLE_FEATURES
        )
    db_session.commit()

    response = client.post(f"/api/v1/models/{trained_model.name}/evaluate")
    assert response.status_code == 200
    result = response.json()

    version_result = next(v for v in result["versions"] if v["version"] == trained_model.version)
    assert version_result["labeled_count"] == 30
    assert version_result["live_accuracy"] == pytest.approx(0.5, abs=0.01)


def test_evaluate_triggers_retraining_when_accuracy_drop_exceeds_threshold_and_enough_data(
    db_session, client, trained_model, active_deployment
):
    training_df = make_synthetic_frame(n=MIN_LABELED_ROWS_FOR_RETRAIN + 20, seed=42)
    for _, row in training_df.iterrows():
        # model_dump() through the pydantic schema normalizes numpy int64/float64 (from the
        # synthetic frame) into plain JSON-serializable Python types.
        features = PredictFeatures(**row.drop("clicked").to_dict()).model_dump()
        actual_label = "click" if row["clicked"] == 1 else "no_click"
        # Deliberately predict the *opposite* of the true label so live accuracy is ~0,
        # guaranteeing a drop far past the 2% threshold regardless of the model's own
        # holdout accuracy.
        predicted_label = "no_click" if row["clicked"] == 1 else "click"
        _insert_inference(
            db_session, trained_model, predicted_label=predicted_label, actual_label=actual_label, features=features
        )
    db_session.commit()

    response = client.post(f"/api/v1/models/{trained_model.name}/evaluate")
    assert response.status_code == 200
    result = response.json()

    version_result = next(v for v in result["versions"] if v["version"] == trained_model.version)
    assert version_result["retrain_triggered"] is True
    assert version_result["retraining_job"]["status"] == "completed"

    from app.core.models import Model, RetrainingJob

    jobs = db_session.query(RetrainingJob).filter_by(model_id=trained_model.id).all()
    assert len(jobs) == 1
    assert jobs[0].status == "completed"
    assert jobs[0].new_version is not None

    new_model = db_session.query(Model).filter_by(id=jobs[0].new_model_id).one()
    assert new_model.status == "staging"
    assert new_model.name == trained_model.name


def test_evaluate_skips_retraining_when_not_enough_labeled_data(db_session, client, trained_model, active_deployment):
    for _i in range(10):
        _insert_inference(
            db_session, trained_model, predicted_label="click", actual_label="no_click", features=SAMPLE_FEATURES
        )
    db_session.commit()

    response = client.post(f"/api/v1/models/{trained_model.name}/evaluate")
    assert response.status_code == 200
    result = response.json()

    version_result = next(v for v in result["versions"] if v["version"] == trained_model.version)
    # Fewer than MIN_LABELED_FOR_ACCURACY_CHECK (30) labeled rows -> no accuracy verdict yet.
    assert version_result["retrain_triggered"] is False
