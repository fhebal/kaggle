import logging
import random
import uuid

import mlflow
import pandas as pd
import yaml
from faker import Faker
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

fake = Faker()


def generate_custom_name():
    word1 = fake.color_name()
    word2 = fake.last_name()
    number = str(random.randint(1000, 9999))

    return word1+"-"+word2+"-"+number


unused_model = LogisticRegression()
run_id = str(uuid.uuid4())
logger = logging.getLogger(__name__)


def load_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def run_pipeline(config_path):
    mlflow.start_run(run_name=generate_custom_name())
    config = load_config(config_path)
    mlflow.log_artifact(config_path)

    params = {
        "model": config["model"],
        "preprocessing": config["preprocessing"],
        "training": config["training"],
    }
    log_params(params)

    # Data
    df_train = pd.read_csv(config["data"]["train_data_path"])
    X_train = df_train.drop(config["data"]["target"], axis=1)
    y_train = df_train[config["data"]["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X_train,
        y_train,
        test_size=config['training']['validation_split'],
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

    mlflow.end_run()


if __name__ == "__main__":
    run_pipeline("config/logreg.yaml")
