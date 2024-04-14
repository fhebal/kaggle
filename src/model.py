import logging
import uuid

import mlflow
import pandas as pd
import yaml
from mlflow import log_metrics, log_params
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (  # noqa E501
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.utils.sklearn import build_model, build_preprocessor

unused_model = LogisticRegression()
run_id = str(uuid.uuid4())
logger = logging.getLogger(__name__)


def load_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def run_pipeline(config_path):
    mlflow.start_run(run_name="Baseline Model Experiment")
    config = load_config(config_path)
    mlflow.log_artifact(config_path)

    params = {
        "model_type": config["model"]["type"],
        "numeric_features": config["preprocessing"]["numeric_features"],
        "categorical_features": config["preprocessing"]["categorical_features"], # noqa E501
        "text_features": config["preprocessing"]["text_features"],
    }
    log_params(params)

    # Data
    df_train = pd.read_csv(config["data"]["train_data_path"])
    X_train = df_train.drop(config["data"]["target"], axis=1)
    y_train = df_train[config["data"]["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42
        )

    # Pipeline
    preprocessor = build_preprocessor(config['preprocessing'])
    pipeline = build_model(config['model'], preprocessor)
    pipeline.fit(X_train, y_train)

    # Inference
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    log_metrics(
        {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="macro"),
            "recall": recall_score(y_test, y_pred, average="macro"),
            "auc": roc_auc_score(y_test, y_pred_proba, average="macro"),
        }
    )

    # End the MLflow experiment
    mlflow.end_run()


if __name__ == "__main__":
    run_pipeline("config/logreg.yaml")
