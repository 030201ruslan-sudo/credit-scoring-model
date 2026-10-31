
import pandas as pd
from scipy.stats import ks_2samp

# Загружаем данные
df = pd.read_csv("data/raw/UCI_Credit_Card.csv")

# Разделяем данные
reference = df.iloc[:24000]
current = df.iloc[24000:]

# Признаки для проверки
columns = ["AGE", "LIMIT_BAL", "BILL_AMT1", "PAY_AMT1", "PAY_0"]

results = []

# Проверяем распределения признаков
for column in columns:
    statistic, p_value = ks_2samp(reference[column], current[column])

    if p_value < 0.05:
        status = "Drift"
    else:
        status = "No Drift"

    results.append({
        "feature": column,
        "ks_statistic": round(statistic, 4),
        "p_value": p_value,
        "status": status
    })

# Создаём таблицу и сохраняем отчёт
report = pd.DataFrame(results)

print(report)

report.to_csv("data/processed/drift_report.csv", index=False)

print("\nОтчёт сохранён в data/processed/drift_report.csv")
