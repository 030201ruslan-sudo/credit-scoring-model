
import pandas as pd
from scipy.stats import ks_2samp
from src.monitoring.psi import calculate_psi

# Загружаем данные
df = pd.read_csv("data/raw/UCI_Credit_Card.csv")

reference = df.iloc[:24000]
current = df.iloc[24000:]

columns = ["AGE", "LIMIT_BAL", "BILL_AMT1", "PAY_AMT1", "PAY_0"]

# Проверяем признаки
results = []

for column in columns:
    statistic, p_value = ks_2samp(reference[column], current[column])
    psi = calculate_psi(reference[column], current[column])

    if p_value < 0.05:
        status = "Drift"
    else:
        status = "No Drift"

    results.append({
        "feature": column,
        "ks_statistic": round(statistic, 4),
        "p_value": p_value,
        "psi": round(psi, 4),
        "status": status
    })

report = pd.DataFrame(results)

print(report)

report.to_csv("data/processed/drift_report.csv", index=False)

print("\nОтчёт сохранён в data/processed/drift_report.csv")
