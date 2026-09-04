def test_create_and_list_deployment(client, trained_model):
    response = client.post(
        "/api/v1/deployments",
        json={"model_id": str(trained_model.id), "environment": "production", "traffic_split": 1.0},
    )
    assert response.status_code == 201, response.text
    created = response.json()
    assert created["model_version"] == trained_model.version
    assert created["is_active"] is True

    listing = client.get("/api/v1/deployments", params={"model_name": "ctr-recommender"})
    assert listing.status_code == 200
    assert any(d["id"] == created["id"] for d in listing.json())


def test_update_deployment_traffic_split(client, active_deployment):
    response = client.patch(f"/api/v1/deployments/{active_deployment.id}", json={"traffic_split": 0.5})
    assert response.status_code == 200
    assert response.json()["traffic_split"] == 0.5


def test_deactivate_deployment(client, active_deployment):
    response = client.patch(f"/api/v1/deployments/{active_deployment.id}", json={"is_active": False})
    assert response.status_code == 200
    assert response.json()["is_active"] is False


def test_create_deployment_for_missing_model_returns_404(client):
    response = client.post(
        "/api/v1/deployments",
        json={"model_id": "00000000-0000-0000-0000-000000000000", "environment": "production", "traffic_split": 1.0},
    )
    assert response.status_code == 404
