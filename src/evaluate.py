"""Evaluation helpers for classification metrics and visualization."""

from typing import Any, Dict, List
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def build_model_comparison_table(
    predictions: Dict[str, np.ndarray],
    y_test: np.ndarray,
    labels: List[str],
) -> pd.DataFrame:
    """Build a comparison table for models using multiclass metrics."""
    rows = []
    for name, y_pred in predictions.items():
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, average="macro", zero_division=0),
            }
        )
    return pd.DataFrame(rows).round(4)


def create_classification_report(y_test: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> str:
    """Generate a classification report string for the selected model."""
    return classification_report(y_test, y_pred, target_names=labels, zero_division=0)


def create_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> pd.DataFrame:
    """Create a confusion matrix DataFrame with labels."""
    matrix = confusion_matrix(y_test, y_pred)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def get_feature_importance_values(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """Get feature importance values for tree-based models or coefficients for linear models."""
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        source = "Feature importance"
    elif hasattr(model, "coef_"):
        importance = np.mean(np.abs(model.coef_), axis=0)
        source = "Coefficient magnitude"
    else:
        raise ValueError("Model does not expose feature importance or coefficients.")
    return pd.DataFrame(
        {"feature": feature_names, "importance": importance, "source": source}
    ).sort_values(by="importance", ascending=False)


def identify_misclassifications(
    y_true: np.ndarray, y_pred: np.ndarray, X: pd.DataFrame, labels: List[str]
) -> pd.DataFrame:
    """Return a DataFrame listing misclassified samples with actual/predicted labels and feature values.

    Parameters
    - y_true: ground-truth numeric labels (np.ndarray)
    - y_pred: predicted numeric labels (np.ndarray)
    - X: feature DataFrame aligned with y_true/y_pred
    - labels: list mapping label index -> label name
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if X is None:
        # Return an empty DataFrame with placeholder columns if no features provided
        return pd.DataFrame(columns=["actual", "predicted"])

    # Reset index to align rows and make selection simple
    X_reset = X.reset_index(drop=True).copy()

    # Boolean mask for misclassified rows
    mask = y_true != y_pred

    if not mask.any():
        return pd.DataFrame(columns=(["actual", "predicted"] + list(X_reset.columns)))

    mis_X = X_reset.loc[mask].copy()
    actual_labels = [labels[int(i)] for i in y_true[mask]]
    pred_labels = [labels[int(i)] for i in y_pred[mask]]

    mis_X.insert(0, "predicted", pred_labels)
    mis_X.insert(0, "actual", actual_labels)

    return mis_X.reset_index(drop=True)
