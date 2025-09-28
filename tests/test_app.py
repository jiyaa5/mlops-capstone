from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict", json={"area": 1200})
    assert response.status_code == 200
    assert "prediction" in response.json()
