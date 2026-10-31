from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

# Проверяем работу сервера
def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# Проверяем обработку неправильного запроса
def test_invalid_data():
    response = client.post("/predict", json={"AGE": 24})

    assert response.status_code == 422
    
# Проверяем предсказание модели
def test_predict():
    data = {
        "LIMIT_BAL": 20000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 24,
        "PAY_0": 2,
        "PAY_2": 2,
        "PAY_3": -1,
        "PAY_4": -1,
        "PAY_5": -2,
        "PAY_6": -2,
        "BILL_AMT1": 3913,
        "BILL_AMT2": 3102,
        "BILL_AMT3": 689,
        "BILL_AMT4": 0,
        "BILL_AMT5": 0,
        "BILL_AMT6": 0,
        "PAY_AMT1": 0,
        "PAY_AMT2": 689,
        "PAY_AMT3": 0,
        "PAY_AMT4": 0,
        "PAY_AMT5": 0,
        "PAY_AMT6": 0
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200
    assert response.json()["default_prediction"] == 1
    assert abs(response.json()["default_probability"] - 0.5561) < 0.001