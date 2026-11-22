"""Compare three logistic regressions on the same Titanic split."""

import hashlib
import subprocess
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import mlflow.sklearn
import pandas as pd

from src.train import SPLIT_SEED, TEST_SIZE, build_model, evaluate, split_data

WORK_DIR = Path("artifacts/hw-3")
DATA_PATH = WORK_DIR / "titanic.csv"
DATA_VERSION = "data-v1"
DATA_SHA256 = "9f5c950ded31ac97c6dcaea04579fe64371173e01f1d9a474a712355496c732a"
TRACKING_URI = "sqlite:///artifacts/hw-3/mlflow.db"
EXPERIMENT = "Titanic HW3"
MODEL_PACKAGES = ("mlflow", "pandas", "scikit-learn", "skops")
VARIANTS = [
    ("baseline", 1.0),
    ("stronger-regularization", 0.1),
    ("weaker-regularization", 10.0),
]


def main():
    if not DATA_PATH.is_file():
        raise SystemExit(
            f"Missing {DATA_PATH}. Fetch data-v1 using the README commands."
        )
    if hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() != DATA_SHA256:
        raise SystemExit(f"{DATA_PATH} does not match {DATA_VERSION}.")

    x_train, x_test, y_train, y_test = split_data(pd.read_csv(DATA_PATH))
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = (
        experiment.experiment_id
        if experiment
        else client.create_experiment(
            EXPERIMENT, (WORK_DIR / "runs").resolve().as_uri()
        )
    )

    for name, c in VARIANTS:
        with mlflow.start_run(experiment_id=experiment_id, run_name=name) as run:
            model = build_model(c)
            model.fit(x_train, y_train)
            metrics = evaluate(model, x_test, y_test)
            mlflow.set_tags(
                {
                    "data_version": DATA_VERSION,
                    "data_sha256": DATA_SHA256,
                    "git_commit": git_commit,
                }
            )
            mlflow.log_params(
                {
                    "estimator": "LogisticRegression",
                    "C": c,
                    "split_seed": SPLIT_SEED,
                    "test_size": TEST_SIZE,
                    "train_rows": len(x_train),
                    "test_rows": len(x_test),
                }
            )
            mlflow.log_metrics(metrics)
            with TemporaryDirectory() as tmp:
                model_path = Path(tmp) / "model"
                mlflow.sklearn.save_model(
                    model,
                    model_path,
                    serialization_format="skops",
                    skops_trusted_types=["numpy.dtype"],
                    pip_requirements=[
                        f"{pkg}=={version(pkg)}" for pkg in MODEL_PACKAGES
                    ],
                )
                mlflow.log_artifacts(model_path, "model")
            print(
                f"{name}: accuracy={metrics['accuracy']:.4f}, "
                f"ROC-AUC={metrics['roc_auc']:.4f}, run={run.info.run_id}"
            )


if __name__ == "__main__":
    main()
