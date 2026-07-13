"""Tests basicos para API REST de negocio."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload.get("status") == "ok"
    assert "model_ready" in payload


def test_predict_conversion_endpoint():
    payload = {
        "threshold": 0.34,
        "records": [
            {
                "age": 30,
                "total_spent": 500,
                "avg_order_value": 80,
                "last_3_month_purchase_freq": 3,
                "total_visits": 20,
                "pages_per_session": 4,
                "support_tickets": 1,
                "satisfaction_score": 4,
            }
        ],
    }
    response = client.post("/predict/conversion", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert "conversion_score" in data["results"][0]


def test_campaign_targets_endpoint():
    response = client.get("/campaign/targets?threshold=0.34&top_n=10")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert data["top_n"] == 10


def run_tests() -> bool:
    tests = [
        test_health_endpoint,
        test_predict_conversion_endpoint,
        test_campaign_targets_endpoint,
    ]
    passed = 0
    failed = 0

    for fn in tests:
        try:
            fn()
            print(f"[OK] {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {fn.__name__}: {exc}")
            failed += 1

    print(f"RESULTADOS API: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    raise SystemExit(0 if run_tests() else 1)
