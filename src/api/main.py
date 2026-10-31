import joblib
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Данные клиента
class ClientData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int

    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int

    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

@app.get("/health")
def health():
    return {"status": "ok"}

# Предсказываем дефолт клиента
@app.post("/predict")
def predict(data: ClientData):
    model = joblib.load("models/credit_model.pkl")

    df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return {
        "default_prediction": int(prediction),
        "default_probability": float(probability)
    }