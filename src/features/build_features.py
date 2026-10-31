def build_features(df):
    df = df.copy()

    pay_columns = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    df["late_payments"] = (df[pay_columns] > 0).sum(axis=1)

    df["credit_utilization"] = df["BILL_AMT1"] / df["LIMIT_BAL"]

    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    return df