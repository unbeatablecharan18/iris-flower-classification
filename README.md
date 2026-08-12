# Iris Flower Classification

## Project Overview

This project builds a professional Iris flower classification system using the Iris dataset from scikit-learn. It includes a complete machine learning workflow with data validation, exploratory data analysis, preprocessing, model training, evaluation, model persistence, CLI prediction, and a polished Streamlit dashboard.

<img width="1824" height="966" alt="Project-Overview" src="https://github.com/user-attachments/assets/d954e371-a1fe-4642-8484-b5fcfed9d79c" />



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

<img width="1814" height="906" alt="exploratory Data Analysis" src="https://github.com/user-attachments/assets/9f1b208c-e155-4bfc-85cc-d2d407f47b96" />


### Feature Relationships

The pairplot provides a visual comparison of the four numerical features across the three Iris species. It helps identify patterns, relationships, and separation between the classes.

<img width="1460" height="1290" alt="exploratory Data Analysis-2" src="https://github.com/user-attachments/assets/586acec6-4610-43d1-b2d1-08734c30e02b" />

## Models Used

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

## Model Comparison

The trained models are compared using dynamic metrics generated from test data.

<img width="1833" height="958" alt="model performance" src="https://github.com/user-attachments/assets/a5c6355e-3597-4b98-96f5-1d774e770d35" />

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 score
* Confusion matrices
* Classification report

## Confusion Matrix

Confusion matrices show how often each Iris species was classified correctly and where the models made classification errors.

<img width="1823" height="930" alt="confusion matrix" src="https://github.com/user-attachments/assets/bbb990bb-2b81-44cd-bdc1-13d10e3ae566" />

## Feature Importance

Feature importance is visualized for the Decision Tree model, and logistic regression coefficients are shown for interpretability.

## Prediction System

* `app.py` provides an interactive Streamlit dashboard
* `src/predict.py` provides a CLI prediction script

The dashboard allows users to enter sepal and petal measurements and receive a predicted Iris species along with prediction probabilities and confidence.

<img width="1837" height="936" alt="prediction with result" src="https://github.com/user-attachments/assets/2ddef68a-9e80-43ca-97aa-e071dd011059" />


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
├── Project-Overview.jpeg
├── confusion matrix.jpeg
├── exploratory Data Analysis-2.png
├── exploratory Data Analysis.jpeg
├── model performance.jpeg
└── prediction with result.jpeg
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

* **Decision Tree:** 0.9667 test accuracy
* **Logistic Regression:** 0.9333 test accuracy
* **Random Forest:** 0.9000 test accuracy

The Decision Tree achieved the best overall test accuracy among the three evaluated models.

## Future Improvements

* Add hyperparameter tuning using GridSearchCV
* Add cross-validation consistency checks
* Export the Streamlit app to a Docker container
* Add unit tests for preprocessing and prediction code

## Author

**Iris Flower Classification — Intelligent Species Prediction System**
