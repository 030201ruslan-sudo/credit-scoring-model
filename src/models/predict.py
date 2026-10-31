import pandas as pd
import mlflow
import mlflow.sklearn

# Подключаемся к MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Загружаем сохранённую модель
model = mlflow.sklearn.load_model(
    "models:/m-40e2f659a8c34cfca83bd72f57c19dfe"
)

# Загружаем данные
df = pd.read_csv("data/raw/UCI_Credit_Card.csv")

# Отделяем признаки
X = df.drop(columns=["ID", "default.payment.next.month"])

X = df.drop(columns=["ID", "default.payment.next.month"])

# Получаем прогноз для первого клиента
prediction = model.predict(X.head(1))
probability = model.predict_proba(X.head(1))[:, 1]

print("Предсказанный класс:", prediction[0])
print("Вероятность дефолта:", round(probability[0], 4))