import pandas as pd
import pytest
from pandera.errors import SchemaError
from src.data.validation import schema


# Создаем небольшую таблицу для тестирования
df = pd.DataFrame({
    "ID": [1, 2],
    "LIMIT_BAL": [20000.0, 50000.0],
    "AGE": [25, 40],
    "EDUCATION": [1, 2],
    "MARRIAGE": [1, 2],
    "default.payment.next.month": [0, 1]
})


def test_valid_data():
    schema.validate(df)


def test_invalid_age():
    df_test = df.copy()
    df_test.loc[0, "AGE"] = 150

    with pytest.raises(SchemaError):
        schema.validate(df_test)