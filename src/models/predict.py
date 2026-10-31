import joblib
import pandas as pd

# Загружаем модель из DVC
model = joblib.load("models/credit_model.pkl")

# Загружаем данные
df = pd.read_csv("data/raw/UCI_Credit_Card.csv")

# Берём первого клиента
X = df.drop(columns=["ID", "default.payment.next.month"])
client = X.iloc[[0]]

# Получаем прогноз
prediction = model.predict(client)
probability = model.predict_proba(client)

print("Прогноз:", prediction[0])
print("Вероятность дефолта:", probability[0][1])
