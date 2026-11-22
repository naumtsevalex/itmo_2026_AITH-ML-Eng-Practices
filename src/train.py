"""Train and evaluate a logistic regression on Titanic passenger data."""

import json
import pickle
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.prepare import CATEGORICAL, NUMERIC, TARGET, read_titanic

SPLIT_SEED = 42
TEST_SIZE = 0.2


def split_data(data):
    features = data[NUMERIC + CATEGORICAL]
    labels = data[TARGET]
    return train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=labels
    )


def build_model(c=1.0):
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
    return make_pipeline(preprocess, LogisticRegression(C=c))


def evaluate(model, x_test, y_test):
    return {
        "accuracy": accuracy_score(y_test, model.predict(x_test)),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]),
    }


def main():
    x_train, x_test, y_train, y_test = split_data(read_titanic())
    model = build_model()
    model.fit(x_train, y_train)
    metrics = {
        **evaluate(model, x_test, y_test),
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
