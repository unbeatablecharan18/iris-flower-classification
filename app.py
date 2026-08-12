"""Streamlit dashboard for Iris flower species prediction and model insights."""

import os
import joblib
import pandas as pd
import seaborn as sns
import plotly.express as px
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError

from src.data_loader import validate_dataset
from src.evaluate import (
    create_classification_report,
    create_confusion_matrix,
    get_feature_importance_values,
    identify_misclassifications,
)
from src.preprocessing import validate_measurements

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression_pipeline.pkl",
    "Decision Tree": "decision_tree_pipeline.pkl",
    "Random Forest": "random_forest_pipeline.pkl",
    "Best Model": "best_model_pipeline.pkl",
}


def load_pipeline(model_filename: str):
    model_path = os.path.join(MODEL_DIR, model_filename)
    if not os.path.exists(model_path):
        st.error("Saved model pipeline not found. Run training first.")
        st.stop()
    return joblib.load(model_path)


def load_iris_dataframe() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = FEATURE_NAMES + ["species"]
    df["species"] = iris.target_names[iris.target]
    return df


def prepare_test_data(df: pd.DataFrame):
    from sklearn.model_selection import train_test_split

    label_mapping = {name: idx for idx, name in enumerate(sorted(df["species"].unique()))}
    y_encoded = df["species"].map(label_mapping)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURE_NAMES],
        y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42,
    )
    return X_test, y_test, list(sorted(label_mapping.keys()))


def get_model_predictions(pipeline, X_test):
    y_pred = pipeline.predict(X_test)
    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)
    return y_pred, probabilities


def display_top_navigation():
    st.markdown("## 🌸 Iris Species Predictor")
    st.markdown("#### Machine Learning Classification Dashboard")
    st.markdown(
        "A polished Iris classification dashboard with dataset validation, EDA, model comparison, misclassification analysis, and prediction."
    )
    st.markdown("---")

    top_right = st.columns([3, 1])
    with top_right[1]:
        st.metric("Dataset", "Iris")
        st.metric("Models", "3")

    tabs = st.tabs(
        [
            "Overview",
            "EDA",
            "Model Performance",
            "Confusion Matrix",
            "Prediction",
            "Model Insights",
            "About Project",
        ]
    )
    return tabs


def display_overview(df):
    st.header("🏠 Project Overview")
    st.markdown(
        "This Iris classification system evaluates the sepal and petal measurements to predict flower species using interpretable machine learning models."
    )
    dataset_info = st.columns(4)
    dataset_info[0].metric("Samples", df.shape[0])
    dataset_info[1].metric("Features", df.shape[1] - 1)
    dataset_info[2].metric("Species", df["species"].nunique())
    dataset_info[3].metric("Duplicates", int(df.duplicated().sum()))

    st.subheader("Dataset snapshot")
    st.dataframe(df.head(), width="stretch")
    with st.expander("Show last 5 rows"):
        st.dataframe(df.tail(), width="stretch")

    validation = validate_dataset(df)
    st.subheader("Data validation")
    st.write(validation.summary)
    st.write(validation.missing_values)
    st.write(f"Duplicate rows: {validation.duplicate_count}")


def display_eda(df):
    st.header("📊 Exploratory Data Analysis")
    st.subheader("Class distribution")
    fig = px.histogram(df, x="species", title="Iris Species Distribution", color="species")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Feature distributions")
    features = FEATURE_NAMES
    fig = px.histogram(df, x=features[0], color="species", barmode="overlay", opacity=0.7, title="Sepal Length Distribution by Species")
    st.plotly_chart(fig, width="stretch")
    fig = px.histogram(df, x=features[2], color="species", barmode="overlay", opacity=0.7, title="Petal Length Distribution by Species")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Pairwise feature relationships")
    pairplot = sns.pairplot(df, hue="species", corner=True, diag_kind="kde", plot_kws={"alpha": 0.7})
    st.pyplot(pairplot.fig)

    st.subheader("Correlation matrix")
    corr = df[FEATURE_NAMES].corr()
    corr_fig = px.imshow(corr, text_auto=True, title="Feature Correlation Heatmap", color_continuous_scale="Blues")
    st.plotly_chart(corr_fig, width="stretch")


def display_model_performance(comparison, df):
    st.header("🤖 Model Performance")
    st.write(comparison)
    st.markdown("The table above compares model performance using accuracy, precision, recall, and F1 score for multiclass classification.")

    best_pipeline = load_pipeline(MODEL_OPTIONS["Best Model"])
    X_test, y_test, labels = prepare_test_data(df)
    y_pred, _ = get_model_predictions(best_pipeline, X_test)
    report_text = create_classification_report(y_test.to_numpy(), y_pred, labels)

    st.subheader("Classification Report — Best Model")
    st.text(report_text)

    st.subheader("Misclassified samples — Best Model")
    mis_df = identify_misclassifications(y_test.to_numpy(), y_pred, X_test, labels)
    if mis_df.empty:
        st.write("No misclassifications on the held-out test set.")
    else:
        st.dataframe(mis_df, width="stretch")


def load_comparison() -> pd.DataFrame:
    comparison_path = os.path.join(MODEL_DIR, "model_comparison.pkl")
    if not os.path.exists(comparison_path):
        st.error("Model comparison results not found. Run training first.")
        st.stop()
    return joblib.load(comparison_path)


def display_confusion_matrices(df, model_files):
    st.header("🔍 Confusion Matrix")
    X = df[FEATURE_NAMES]
    y = df["species"]
    label_mapping = {name: idx for idx, name in enumerate(sorted(y.unique()))}
    y_encoded = y.map(label_mapping)
    X_train, X_test, y_train, y_test = None, None, None, None

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42,
    )

    for model_name, model_file in model_files.items():
        pipeline = load_pipeline(model_file)
        y_pred = pipeline.predict(X_test)
        matrix = create_confusion_matrix(y_test.to_numpy(), y_pred, sorted(label_mapping.keys()))
        st.subheader(model_name)
        st.write(matrix)
        st.markdown(
            "The confusion matrix above shows how often each true species was predicted correctly or misclassified by the model."
        )


def display_prediction_interface():
    st.header("🌸 Flower Prediction")
    model_choice = st.selectbox("Choose prediction model", list(MODEL_OPTIONS.keys()), index=3)
    with st.form("prediction_form"):
        sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
        sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
        petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
        petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
        submitted = st.form_submit_button("Predict Flower Species")

    if submitted:
        values = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width,
        }
        is_valid, message = validate_measurements(values)
        if not is_valid:
            st.error(message)
            return

        model_file = MODEL_OPTIONS[model_choice]
        pipeline = load_pipeline(model_file)
        data = pd.DataFrame([values])
        prediction = pipeline.predict(data)[0]
        probability = None
        species_names = ["setosa", "versicolor", "virginica"]
        if hasattr(pipeline, "predict_proba"):
            probability = pipeline.predict_proba(data)[0]

        st.markdown("### Prediction Result")
        st.write(f"**Predicted species:** {species_names[int(prediction)].title()}")
        if probability is not None:
            st.write("**Prediction probabilities:**")
            prob_df = pd.DataFrame({"Species": [name.title() for name in species_names], "Probability": probability})
            st.table(prob_df.set_index("Species"))
            confidence = float(probability[int(prediction)])
            st.success(f"Confidence: {confidence:.2f}")

        st.markdown("### Measurement summary")
        st.write(values)
        st.markdown(
            f"The measurements are most consistent with the learned characteristics of {species_names[int(prediction)].title()}."
        )


def display_model_insights():
    st.header("💡 Model Insights")
    st.markdown("Decision trees and logistic regression provide complementary interpretability for Iris classification.")
    pipeline = load_pipeline(MODEL_OPTIONS["Decision Tree"])
    tree = pipeline.named_steps["classifier"]
    importance_df = get_feature_importance_values(tree, FEATURE_NAMES)
    st.subheader("Decision Tree Feature Importance")
    st.write(importance_df[["feature", "importance"]].set_index("feature"))
    fig = px.bar(importance_df, x="feature", y="importance", title="Decision Tree Feature Importance")
    st.plotly_chart(fig, width="stretch")

    logistic_pipeline = load_pipeline(MODEL_OPTIONS["Logistic Regression"])
    lr_coef = logistic_pipeline.named_steps["classifier"].coef_
    coef_mean = lr_coef.mean(axis=0)
    coef_df = pd.DataFrame({"feature": FEATURE_NAMES, "coefficient_magnitude": abs(coef_mean)})
    st.subheader("Logistic Regression Coefficients")
    st.write(coef_df.set_index("feature"))
    fig2 = px.bar(coef_df, x="feature", y="coefficient_magnitude", title="Logistic Regression Coefficient Magnitudes")
    st.plotly_chart(fig2, width="stretch")


def display_about():
    st.header("📋 About Project")
    st.markdown(
        "This portfolio-ready application covers the full machine learning workflow, from data validation to model deployment."
    )
    st.markdown(
        "**Tech stack:** Python, pandas, scikit-learn, Streamlit, seaborn, plotly, joblib."
    )
    st.markdown(
        "**Key features:** clean data validation, model comparison, performance metrics, interactive prediction, and saved model pipelines."
    )


def main():
    st.set_page_config(page_title="Iris Flower Classification", page_icon="🌸", layout="wide")
    df = load_iris_dataframe()
    tabs = display_top_navigation()
    tab_overview, tab_eda, tab_perf, tab_confusion, tab_prediction, tab_insights, tab_about = tabs

    with tab_overview:
        display_overview(df)
    with tab_eda:
        display_eda(df)
    with tab_perf:
        comparison = load_comparison()
        display_model_performance(comparison, df)
    with tab_confusion:
        display_confusion_matrices(df, MODEL_OPTIONS)
    with tab_prediction:
        display_prediction_interface()
    with tab_insights:
        display_model_insights()
    with tab_about:
        display_about()


if __name__ == "__main__":
    try:
        main()
    except NotFittedError:
        st.error("Model pipelines must be trained and saved before using the dashboard.")
    except Exception as exc:
        st.error(f"An unexpected error occurred: {exc}")
