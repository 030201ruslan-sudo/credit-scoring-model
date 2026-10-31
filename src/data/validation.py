import pandas as pd
import pandera.pandas as pa

schema = pa.DataFrameSchema({
    "ID": pa.Column(int, nullable=False, unique=True),
    "LIMIT_BAL": pa.Column(float, pa.Check.gt(0), nullable=False),
    "AGE": pa.Column(int, pa.Check.in_range(18, 100), nullable=False),
    "EDUCATION": pa.Column(int, pa.Check.isin([0, 1, 2, 3, 4, 5, 6])),
    "MARRIAGE": pa.Column(int, pa.Check.isin([0, 1, 2, 3])),
    "default.payment.next.month": pa.Column(int, pa.Check.isin([0, 1]))
})

if __name__ == "__main__":
    df = pd.read_csv("data/raw/UCI_Credit_Card.csv")
    schema.validate(df)
    print("Проверка данных пройдена успешно!")
