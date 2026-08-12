# Iris Flower Classification

## Project Overview

This project builds a professional Iris flower classification system using the Iris dataset from scikit-learn. It includes a complete machine learning workflow with data validation, exploratory data analysis, preprocessing, model training, evaluation, model persistence, CLI prediction, and a polished Streamlit dashboard.

<p align="center">
  <img src="images/project-overview.jpeg" width="95%">
</p>

## Problem Statement

Predict the species of an Iris flower based on four measurements:

* Sepal length
* Sepal width
* Petal length
* Petal width

The species labels are:

* Iris Setosa
* Iris Versicolor
* Iris Virginica

## Objectives

* Load and validate the Iris dataset
* Perform exploratory data analysis with visualizations
* Train and compare Logistic Regression, Decision Tree, and Random Forest models
* Evaluate models using accuracy, precision, recall, and F1 score
* Save model pipelines with Joblib
* Provide both CLI and Streamlit-based prediction interfaces

## Dataset

The dataset is loaded directly from scikit-learn and includes 150 Iris flower samples with 4 numerical features and a target species label.

## Technologies Used

* Python 3
* pandas
* NumPy
* scikit-learn
* matplotlib
* seaborn
* plotly
* Streamlit
* joblib

## Machine Learning Workflow

1. Data loading and validation
2. Exploratory data analysis
3. Preprocessing and label encoding
4. Train/test split with stratification
5. Model training with Logistic Regression, Decision Tree, and Random Forest
6. Evaluation and comparison
7. Save pipelines and use them for predictions

## Exploratory Data Analysis

The project includes dataset overview metrics, class distribution charts, feature distributions, pairwise relationship plots, and correlation heatmaps.

<p align="center">
  <img src="images/exploratory-data-analysis.jpeg" width="95%">
</p>

### Feature Relationships

The feature relationships across the Iris species are explored using pairwise visualizations to understand how the four measurements differ between classes.

<p align="center">
  <img src="images/exploratory-data-analysis-2.png" width="90%">
</p>

## Models Used

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

## Model Comparison

The trained models are compared using dynamic metrics generated from test data.

<p align="center">
  <img src="images/model-performance.jpeg" width="95%">
</p>

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 score
* Confusion matrices
* Classification report

## Confusion Matrix

Confusion matrices are used to evaluate the classification performance of the trained models and identify correctly classified and misclassified Iris samples.

<p align="center">
  <img src="images/confusion-matrix.jpeg" width="95%">
</p>

## Feature Importance

Feature importance is visualized for the Decision Tree model, and logistic regression coefficients are shown for interpretability.

## Prediction System

* `app.py` provides an interactive Streamlit dashboard
* `src/predict.py` provides a CLI prediction script

<p align="center">
  <img src="images/prediction-with-result.jpeg" width="95%">
</p>

## Project Structure

```text
iris-flower-classification/
├── data/
├── models/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
```

## Installation

1. Clone the repository or copy the project folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

### Train models and save pipelines

```bash
python -m src.train
```

### Make a CLI prediction

```bash
python src/predict.py 5.1 3.5 1.4 0.2 --model best_model_pipeline.pkl
```

### Run the Streamlit app

```bash
streamlit run app.py
```

## Example Prediction

```bash
python src/predict.py 5.1 3.5 1.4 0.2
```

Expected output:

* Predicted species: Setosa
* Probabilities for all species if supported by the selected model

## Results

The model training produced the following results:

* Decision Tree achieved the best test accuracy: 0.9667
* Logistic Regression test accuracy: 0.9333
* Random Forest test accuracy: 0.9000

## Future Improvements

* Add hyperparameter tuning using GridSearchCV
* Add cross-validation consistency checks
* Export the Streamlit app to a Docker container
* Add unit tests for preprocessing and prediction code

## Author

Iris Flower Classification — Intelligent Species Prediction System

