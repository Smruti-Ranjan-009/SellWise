from fastapi.testclient import TestClient

import app


client = TestClient(app.app)


def test_health_reports_serving_artifacts() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["forecast_available"] is True
    assert response.json()["metrics_available"] is True


def test_catalog_lists_stores() -> None:
    response = client.get("/catalog")

    assert response.status_code == 200
    assert response.json()["stores"]


def test_metrics_contains_overall_score() -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert isinstance(response.json()["WRMSSE_final"], (int, float))


def test_forecast_returns_selected_item_horizon() -> None:
    catalog_response = client.get("/catalog", params={"store_id": "CA_1"})
    assert catalog_response.status_code == 200
    item_id = catalog_response.json()["items"][0]
    department_id = "_".join(item_id.split("_")[:2])

    response = client.get(
        "/forecasts",
        params={
            "store_id": "CA_1",
            "department_id": department_id,
            "item_id": item_id,
        },
    )

    assert response.status_code == 200
    assert response.json()["total_series"] == 1
    assert len(response.json()["data"]) == 28