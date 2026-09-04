import io

from tests.conftest import make_synthetic_frame


def _csv_bytes(n: int = 50) -> bytes:
    df = make_synthetic_frame(n=n, seed=9).drop(columns=["clicked"])
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def test_batch_predict_scores_every_row(client, active_deployment, trained_model):
    response = client.post(
        "/api/v1/batch-predict",
        params={"model_name": "ctr-recommender"},
        files={"file": ("batch.csv", _csv_bytes(50), "text/csv")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rows"] == 50
    assert body["model_version"] == trained_model.version
    assert 0.0 <= body["click_rate"] <= 1.0
    assert body["rows_per_second"] > 0


def test_batch_predict_publishes_one_event_per_row(client, active_deployment, fake_producer):
    client.post(
        "/api/v1/batch-predict",
        params={"model_name": "ctr-recommender"},
        files={"file": ("batch.csv", _csv_bytes(25), "text/csv")},
    )
    assert len(fake_producer.events) == 25


def test_batch_predict_rejects_csv_missing_columns(client, active_deployment):
    bad_csv = b"user_age,user_tenure_days\n30,100\n"
    response = client.post(
        "/api/v1/batch-predict",
        params={"model_name": "ctr-recommender"},
        files={"file": ("bad.csv", bad_csv, "text/csv")},
    )
    assert response.status_code == 400


def test_batch_predict_returns_404_for_unknown_model(client):
    response = client.post(
        "/api/v1/batch-predict",
        params={"model_name": "unknown-model"},
        files={"file": ("batch.csv", _csv_bytes(5), "text/csv")},
    )
    assert response.status_code == 404
