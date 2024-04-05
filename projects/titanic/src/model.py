import time
import uuid
from io import StringIO

import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (  # noqa
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # GridSearchCV,
from sklearn.pipeline import Pipeline
from src.utils.kaggle import check_submission_status, submit_to_kaggle
from src.utils.sklearn import build_preprocessor

import mlflow
from mlflow import log_metrics, log_params

unused_model = LogisticRegression()


def load_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def build_model(config, preprocessor):
    model_class = getattr(
        __import__("sklearn.linear_model", fromlist=[config["model"]["type"]]),
        config["model"]["type"],
    )
    model = model_class(**config["model"]["parameters"])
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", model)]
    )  # noqa E501

    return pipeline


def run_pipeline(config_path):

    # Start an MLflow experiment
    mlflow.start_run(run_name="Baseline Model Experiment")

    # Load the configuration
    config = load_config(config_path)

    # Log the entire configuration file
    # mlflow.log_artifact(config_path, "config")

    log_params(
        {
            "model_type": config["model"]["type"],
            "preprocessor_numeric_features": len(
                config["preprocessing"]["numeric_features"]
            ),
            "preprocessor_categorical_features": len(
                config["preprocessing"]["categorical_features"]
            ),
            "preprocessor_text_features": len(
                config["preprocessing"]["text_features"]
            ),  # noqa: E501
        }
    )

    # Continue as before to build preprocessor, model, load data, etc.
    preprocessor = build_preprocessor(config)
    pipeline = build_model(config, preprocessor)
    df_train = pd.read_csv(config["data"]["train_data_path"])
    X_train = df_train.drop(
        config["data"]["target"], axis=1
    )  # Adjust based on your config structure
    y_train = df_train[
        config["data"]["target"]
    ]  # Adjust based on your config structure
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # Train and evaluate the model as before
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    validation = pd.read_csv("data/test.csv")
    validation["Survived"] = pipeline.predict(validation)
    validation[["PassengerId", "Survived"]].to_csv(
        "data/submission.csv", index=False
    )  # noqa: E501

    # Generate a unique message using UUID
    run_id = uuid.uuid4()
    submit_to_kaggle(
        "titanic", "data/submission.csv", f"My automated submission {run_id}"
    )

    time.sleep(10)

    # Check the submission status
    kaggle_result = check_submission_status("titanic")

    run_id_str = str(run_id)
    df = pd.read_csv(StringIO(kaggle_result))

    # Filter the DataFrame for your submission using the run_id
    filtered_df = df[df["description"].str.contains(run_id_str, na=False)]

    # Assuming the filtered DataFrame is not empty, extract the AUC metric
    if not filtered_df.empty:
        auc_metric = filtered_df.iloc[0]["publicScore"]
        print(f"AUC Metric for run_id {run_id_str}: {auc_metric}")
    else:
        print("No matching submission found.")

    # Log metrics
    log_metrics(
        {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="macro"),
            "recall": recall_score(y_test, y_pred, average="macro"),
            "auc": roc_auc_score(y_test, y_pred_proba, average="macro"),
            "kaggle": auc_metric,
        }
    )

    # End the MLflow experiment
    mlflow.end_run()


# Example usage
run_pipeline("config/sk_baseline.yaml")
