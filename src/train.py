"""Train and evaluate a logistic regression on Titanic passenger data."""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.get_data import DATA_PATH

NUMERIC = ["age", "fare", "sibsp", "parch"]
CATEGORICAL = ["pclass", "sex"]
TARGET = "survived"


def main():
    with DATA_PATH.open(encoding="utf-8") as data_file:
        columns = []
        for line in data_file:
            if line.startswith("@attribute "):
                columns.append(line.split("'", 2)[1])
            elif line.strip().lower() == "@data":
                break
        data = pd.read_csv(data_file, names=columns, na_values="?")

    features = data[NUMERIC + CATEGORICAL]
    labels = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                NUMERIC,
            ),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    model = make_pipeline(preprocess, LogisticRegression())
    model.fit(x_train, y_train)
    metrics = {
        "accuracy": accuracy_score(y_test, model.predict(x_test)),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
    }

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    with (artifacts / "model.pkl").open("wb") as model_file:
        pickle.dump(model, model_file)
    metrics_json = json.dumps(metrics, indent=2, allow_nan=False)
    (artifacts / "metrics.json").write_text(metrics_json + "\n", encoding="utf-8")
    print(metrics_json)
    print("Saved artifacts/model.pkl and artifacts/metrics.json")


if __name__ == "__main__":
    main()
