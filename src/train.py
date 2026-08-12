"""Train Iris classification models and save the trained pipelines."""

import os
from typing import Dict, Tuple
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_iris_dataset, validate_dataset
from src.preprocessing import build_preprocessing_pipeline, encode_target
from src.evaluate import build_model_comparison_table


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def ensure_model_dir() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)


def train_models() -> Tuple[Dict[str, Pipeline], Dict[str, pd.DataFrame], pd.Series, object, Dict[str, dict], object, str]:
    """Train and compare multiple Iris classification models."""
    df, X, y, target_names = load_iris_dataset()
    report = validate_dataset(df)

    y_encoded, label_encoder = encode_target(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42,
    )

    models = {
        "Logistic Regression": Pipeline(
            [("scaler", build_preprocessing_pipeline()),
             ("classifier", LogisticRegression(
                 solver="lbfgs",
                 max_iter=500,
                 random_state=42,
             ))]
        ),
        "Decision Tree": Pipeline(
            [("scaler", build_preprocessing_pipeline()),
             ("classifier", DecisionTreeClassifier(
                 max_depth=5,
                 min_samples_leaf=3,
                 random_state=42,
             ))]
        ),
        "Random Forest": Pipeline(
            [("scaler", build_preprocessing_pipeline()),
             ("classifier", RandomForestClassifier(
                 n_estimators=100,
                 random_state=42,
             ))]
        ),
    }

    results = {}
    predictions = {}
    trained_models = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        predictions[name] = y_pred
        trained_models[name] = pipeline

    comparison = build_model_comparison_table(predictions, y_test.to_numpy(), target_names)
    best_model_name = comparison.sort_values(by="Accuracy", ascending=False).iloc[0]["Model"]
    best_pipeline = trained_models[best_model_name]

    ensure_model_dir()
    joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "best_model_pipeline.pkl"))
    joblib.dump(trained_models["Logistic Regression"], os.path.join(MODEL_DIR, "logistic_regression_pipeline.pkl"))
    joblib.dump(trained_models["Decision Tree"], os.path.join(MODEL_DIR, "decision_tree_pipeline.pkl"))
    joblib.dump(trained_models["Random Forest"], os.path.join(MODEL_DIR, "random_forest_pipeline.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(trained_models["Logistic Regression"].named_steps["scaler"], os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(comparison, os.path.join(MODEL_DIR, "model_comparison.pkl"))

    metrics = comparison.set_index("Model").to_dict(orient="index")
    metrics = {name: {k: float(v) for k, v in values.items()} for name, values in metrics.items()}

    return trained_models, {"comparison": comparison}, y_test, label_encoder, metrics, report, best_model_name


def main() -> None:
    trained_models, outputs, y_test, label_encoder, metrics, report, best_model_name = train_models()
    print("\nTraining complete.")
    print("Best model:", best_model_name)
    print("\nModel comparison:")
    print(outputs["comparison"].to_string(index=False))
    print("\nDataset validation completed.")


if __name__ == "__main__":
    main()
