"""Data loading and validation for Iris flower classification."""

from dataclasses import dataclass
from typing import Tuple
import pandas as pd
from sklearn.datasets import load_iris


@dataclass
class DatasetReport:
    samples: int
    features: int
    class_distribution: pd.Series
    missing_values: pd.Series
    duplicate_count: int
    summary: pd.DataFrame
    dtypes: pd.Series


def load_iris_dataset() -> Tuple[pd.DataFrame, pd.Series, list]:
    """Load the Iris dataset and return a cleaned DataFrame, target series, and class names."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]
    df["species"] = iris.target_names[iris.target]
    target_names = list(iris.target_names)
    X = df.drop(columns=["species"])
    y = df["species"].copy()
    return df, X, y, target_names


def validate_dataset(df: pd.DataFrame) -> DatasetReport:
    """Validate the dataset and return a report with checks for missing values and duplicates."""
    class_distribution = df["species"].value_counts().sort_index()
    missing_values = df.isna().sum()
    duplicate_count = df.duplicated().sum()
    summary = df.describe().transpose()
    dtypes = df.dtypes

    return DatasetReport(
        samples=df.shape[0],
        features=df.shape[1] - 1,
        class_distribution=class_distribution,
        missing_values=missing_values,
        duplicate_count=duplicate_count,
        summary=summary,
        dtypes=dtypes,
    )


def show_dataset_overview(df: pd.DataFrame) -> None:
    """Print a dataset overview to the console."""
    report = validate_dataset(df)
    print("\n=== Iris Dataset Overview ===")
    print(f"Total samples: {report.samples}")
    print(f"Feature columns: {report.features}")
    print("\nClass distribution:")
    print(report.class_distribution.to_string())
    print("\nMissing values:")
    print(report.missing_values.to_string())
    print(f"\nDuplicate rows: {report.duplicate_count}")
    print("\nData types:")
    print(report.dtypes.to_string())
    print("\nStatistical summary:")
    print(report.summary)
