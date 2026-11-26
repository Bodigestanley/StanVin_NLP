from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_metadata():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "ReturnIQ"

def test_classes_are_fixed_four():
    response = client.get("/api/classes")
    assert response.status_code == 200
    assert response.json()["classes"] == ["Sizing", "Damaged Item", "Wrong Product", "Late Delivery"]

def test_health_shape():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "database_connected" in payload
    assert "tfidf_model_available" in payload
    assert "distilbert_model_available" in payload

def test_predict_rejects_short_text():
    response = client.post("/api/predict", json={"text":"a","model":"tfidf"})
    assert response.status_code == 422
