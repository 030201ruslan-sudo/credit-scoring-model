
import pandas as pd
import requests

# Загружаем данные
df = pd.read_csv("data/raw/UCI_Credit_Card.csv")

# Выбираем 10 клиентов
clients = df.tail(10)

# Отправляем данные в API
for i in range(len(clients)):
    client = clients.iloc[i]

    data = client.drop(
        labels=["ID", "default.payment.next.month"]
    ).to_dict()

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=data,
        timeout=10
    )

    print("Клиент:", i + 1)
    print("HTTP:", response.status_code)

    if response.status_code == 200:
        print("Прогноз:", response.json()["default_prediction"])
        print("Вероятность:", response.json()["default_probability"])
