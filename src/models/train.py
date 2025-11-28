import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_model(data_path, model_out):
    logging.info("Loading processed dataset...")
    df = pd.read_csv(data_path)

    X = df["text"]
    y = df["label"]

    logging.info("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    logging.info("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)

    logging.info(f"Accuracy: {acc}")
    logging.info(f"Precision: {prec}")
    logging.info(f"Recall: {rec}")

    # MLflow logging
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_param("max_iter", 200)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)

    joblib.dump({"model": model, "vectorizer": vectorizer}, model_out)
    mlflow.sklearn.log_model(model, "model")

    logging.info(f"Model saved to {model_out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train spam model")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()
    with mlflow.start_run():
        train_model(args.data, args.out)
