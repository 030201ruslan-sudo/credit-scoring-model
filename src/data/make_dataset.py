
import pandas as pd
from sklearn.model_selection import train_test_split

# Загружаем исходные данные
df = pd.read_csv("data/raw/UCI_Credit_Card.csv")

# Разделяем данные на две выборки
train, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["default.payment.next.month"]
)

# Сохраняем подготовленные данные
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("Обучающая выборка:", train.shape)
print("Тестовая выборка:", test.shape)
