import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
from src.features.build_features import build_features
from sklearn.preprocessing import FunctionTransformer

# Настраиваем MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Credit_Default_Prediction")


# Загружаем подготовленные данные
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

# Отделяем признаки и целевую переменную
X_train = train.drop(columns=["ID", "default.payment.next.month"])
y_train = train["default.payment.next.month"]

X_test = test.drop(columns=["ID", "default.payment.next.month"])
y_test = test["default.payment.next.month"]

print("Обучающая выборка:", X_train.shape)
print("Тестовая выборка:", X_test.shape)

# Разделяем признаки по типам
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

numeric_features = X_train.drop(columns=categorical_features).columns.tolist()

numeric_features = numeric_features + ["late_payments", "credit_utilization"]

# Создаем предобработку
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Создаем Pipeline
model = Pipeline([
    ("features", FunctionTransformer(build_features)),
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# Подбираем гиперпараметры
param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10]
}

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Используем лучшую модель
model = grid_search.best_estimator_

# Сохраняем обученную модель
joblib.dump(model, "models/credit_model.pkl")

print("Модель сохранена в models/credit_model.pkl")

print("Лучшие параметры:", grid_search.best_params_)
print("Лучший ROC-AUC на кросс-валидации:", round(grid_search.best_score_, 4))

print("Модель успешно обучена!")

# Получаем предсказания на тестовой выборке
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Рассчитываем метрики
auc = roc_auc_score(y_test, probabilities)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

# Сохраняем результаты эксперимента
with mlflow.start_run(run_name="Logistic_Regression"):
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("C", grid_search.best_params_["classifier__C"])
    mlflow.log_param("cv", 3)

    mlflow.log_metric("roc_auc", auc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Сохраняем ROC-кривую
    RocCurveDisplay.from_predictions(y_test, probabilities)

    plt.title("ROC-кривая Logistic Regression")
    plt.grid()

    mlflow.log_figure(plt.gcf(), "roc_curve.png")
    plt.close()
    
    # Сохраняем обученную модель
    mlflow.sklearn.log_model(
        sk_model=model,
        name="credit_default_model",
        serialization_format="cloudpickle"
    )
    


print("Результаты сохранены в MLflow!")

print("ROC-AUC:", round(auc, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
