#!/usr/bin/env python3
"""
Train a Logistic Regression on the Iris dataset, log with mlflow, and save the model.

Usage:
    python src/train_model.py --timestamp 20251020123456
"""
import argparse
import datetime
import os
import pickle
import mlflow
from joblib import dump
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split

def main(timestamp: str):
    # --- prepare data ---
    X, y = load_iris(return_X_y=True)

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.30, random_state=42, shuffle=True
    )

    # Save test split so evaluate_model.py can reuse it
    if not os.path.exists("data"):
        os.makedirs("data")
    with open("data/test_X.pickle", "wb") as fx:
        pickle.dump(test_X, fx)
    with open("data/test_y.pickle", "wb") as fy:
        pickle.dump(test_y, fy)

    # --- mlflow setup ---
    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "Iris"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"
    # create_experiment raises if exists, but timestamp ensures unique name
    experiment_id = mlflow.create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{dataset_name}_LogReg"):
        mlflow.log_param("dataset", "iris")
        mlflow.log_param("n_samples", X.shape[0])
        mlflow.log_param("n_features", X.shape[1])

        # --- train model ---
        clf = LogisticRegression(max_iter=500, random_state=0)
        clf.fit(train_X, train_y)

        # --- metrics on test set ---
        preds = clf.predict(test_X)
        acc = accuracy_score(test_y, preds)
        f1 = f1_score(test_y, preds, average="weighted")
        prec = precision_score(test_y, preds, average="weighted")

        mlflow.log_metrics({
            "accuracy": acc,
            "f1_weighted": f1,
            "precision_weighted": prec
        })

        # Ensure models/ folder exists locally for human inspection (workflow moves the model later)
        if not os.path.exists("models"):
            os.makedirs("models")

        # Save the model using the same naming convention expected by the GitHub Actions
        model_version = f"model_{timestamp}_dt_model"  # keep same pattern so workflow scripts keep working
        model_filename = f"{model_version}.joblib"
        dump(clf, model_filename)
        print(f"Saved model to: {model_filename}")
        # Optionally also save a copy under models/ for local listing
        dump(clf, os.path.join("models", model_filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    main(args.timestamp)
