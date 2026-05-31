import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


DATA_PATH = "data/processed/model_data.csv"
MODEL_PATH = "models/best_model.pkl"
METRICS_PATH = "reports/model_comparison.csv"


def load_model_data(path=DATA_PATH):
    return pd.read_csv(path)


def train_and_evaluate_model(model_name, model, x_train, x_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        mlflow.log_param("model_name", model_name)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.sklearn.log_model(model, model_name)

        return metrics, model


def main():
    df = load_model_data()

    x = df.drop(columns=["CustomerId", "is_high_risk"])
    y = df["is_high_risk"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results = []
    best_model = None
    best_model_name = None
    best_f1 = -1

    for model_name, model in models.items():
        metrics, trained_model = train_and_evaluate_model(
            model_name,
            model,
            x_train,
            x_test,
            y_train,
            y_test
        )

        row = {"model": model_name}
        row.update(metrics)
        results.append(row)

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model = trained_model
            best_model_name = model_name

    results_df = pd.DataFrame(results)
    results_df.to_csv(METRICS_PATH, index=False)

    joblib.dump(best_model, MODEL_PATH)

    print("Task 5 completed successfully.")
    print("Best model:", best_model_name)
    print(results_df)


if __name__ == "__main__":
    main()