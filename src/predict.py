"""Command-line prediction script for Iris species using a saved pipeline."""

import argparse
import os
import sys
import numpy as np
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


def load_pipeline(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Saved model pipeline not found: {path}")
    return joblib.load(path)


def load_label_encoder():
    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError("Saved label encoder not found. Run training first.")
    return joblib.load(LABEL_ENCODER_PATH)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Iris flower species from numeric measurements.")
    parser.add_argument("sepal_length", type=float, help="Sepal length in centimeters")
    parser.add_argument("sepal_width", type=float, help="Sepal width in centimeters")
    parser.add_argument("petal_length", type=float, help="Petal length in centimeters")
    parser.add_argument("petal_width", type=float, help="Petal width in centimeters")
    parser.add_argument("--model", type=str, default="best_model_pipeline.pkl",
                        choices=["best_model_pipeline.pkl", "logistic_regression_pipeline.pkl", "decision_tree_pipeline.pkl", "random_forest_pipeline.pkl"],
                        help="Choose the saved model pipeline file.")
    return parser.parse_args()


def validate_inputs(values: np.ndarray) -> None:
    if np.any(values <= 0):
        raise ValueError("All flower measurements must be greater than zero.")
    if np.any(values > 10):
        raise ValueError("Measurements above 10 cm are outside the typical Iris range.")


def main() -> None:
    args = parse_arguments()
    inputs = np.array([args.sepal_length, args.sepal_width, args.petal_length, args.petal_width], dtype=float).reshape(1, -1)
    validate_inputs(inputs)

    pipeline_path = os.path.join(MODEL_DIR, args.model)
    pipeline = load_pipeline(pipeline_path)

    import pandas as pd
    input_df = pd.DataFrame([
        {
            "sepal_length": args.sepal_length,
            "sepal_width": args.sepal_width,
            "petal_length": args.petal_length,
            "petal_width": args.petal_width,
        }
    ])

    prediction = pipeline.predict(input_df)[0]
    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(input_df)[0]

    label_encoder = load_label_encoder()
    predicted_species = label_encoder.inverse_transform([int(prediction)])[0]

    print("\nIris Flower Prediction")
    print("---------------------")
    print(f"Input measurements: Sepal Length={args.sepal_length}, Sepal Width={args.sepal_width}, Petal Length={args.petal_length}, Petal Width={args.petal_width}")
    print(f"Model: {args.model.replace('_pipeline.pkl', '').replace('_', ' ').title()}")
    print(f"Predicted species: {predicted_species.title()}")
    if probabilities is not None:
        print("Prediction probabilities:")
        for idx, prob in enumerate(probabilities):
            species_name = label_encoder.inverse_transform([idx])[0]
            print(f"  {species_name.title()}: {prob:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
