# Diabetes Readmission Prediction Using Decision Trees

## Overview

This project uses machine learning to predict **early hospital readmission of diabetic patients within 30 days of discharge**. The analysis is based on a clinical dataset containing **101,766 patient encounters and 50 attributes**, covering demographic information, admission details, diagnoses, laboratory tests, medication history, and healthcare utilization.

The project focuses on identifying patient and hospital-encounter characteristics associated with early readmission and building an interpretable **Decision Tree classifier** for binary prediction.

## Objectives

* Analyze patterns associated with diabetic patient readmission.
* Explore relationships between clinical and hospital-utilization variables and readmission.
* Transform categorical and diagnostic variables into machine-learning-compatible representations.
* Address class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**.
* Train and evaluate a Decision Tree classification model.
* Identify the features that contribute most strongly to the model's predictions.

## Dataset

The dataset contains **101,766 encounters with 50 variables**. Features include:

* Demographics: race, gender, age
* Admission information: admission type, admission source, discharge disposition
* Hospital utilization: time in hospital, outpatient visits, emergency visits, inpatient visits
* Clinical information: diagnoses, laboratory procedures, number of diagnoses
* Medication information: medication counts and diabetes medications
* Diabetes-specific measurements: glucose serum and HbA1c results
* Target variable: `readmitted`

The original target contains three outcomes: `NO`, `>30`, and `<30`. For binary classification, **`<30` was treated as the positive class (early readmission)**, while `NO` and `>30` were treated as the negative class.

## Exploratory Data Analysis

The analysis investigates relationships between readmission and variables such as:

* Number of medications
* Glucose serum test results
* HbA1c results
* Healthcare service utilization
* Patient demographics
* Hospital admission characteristics

For example, the notebook examines medication usage and service utilization across readmission categories to identify potential patterns in patient outcomes.

## Data Preprocessing

The preprocessing pipeline includes:

1. Converting the target variable into a binary classification problem.
2. Processing diagnosis codes and creating hierarchical diagnostic representations.
3. Handling missing/unknown diagnostic values.
4. Encoding categorical variables using one-hot encoding.
5. Converting relevant diagnostic features into numerical representations.
6. Balancing the training data using SMOTE.

Categorical variables including gender, admission type, discharge disposition, admission source, glucose results, HbA1c results, diagnosis categories, and race were encoded for model training.

### Class Balancing

The processed dataset exhibited a substantial class imbalance, with **5,071 early-readmission cases versus 54,635 non-early-readmission cases**. SMOTE was used to synthetically oversample the minority class, producing balanced classes of **54,635 samples each** before model training.

## Model

A **Decision Tree Classifier** was selected because of its interpretability and ability to model nonlinear relationships between clinical variables.

The final configuration used:

```text
DecisionTreeClassifier(
    criterion='entropy',
    max_depth=28,
    min_samples_split=10
)
```

The model was trained using an 80/20 train-test split.

## Results

The Decision Tree achieved the following performance on the test set:

| Metric    |    Score |
| --------- | -------: |
| Accuracy  | **0.90** |
| Precision | **0.91** |
| Recall    | **0.89** |

These results indicate that the model was able to distinguish early readmission cases from non-early-readmission cases with relatively strong precision and recall on the evaluated test set.

## Feature Importance

The Decision Tree's built-in feature importance scores were used to identify the **10 most influential features** in the model. This provides an interpretable view of which patient, clinical, and hospital-utilization variables contributed most to the classification process.

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **imbalanced-learn / SMOTE**
* **Google Colab**

## Project Workflow

```text
Clinical Dataset
       ↓
Exploratory Data Analysis
       ↓
Data Cleaning & Preprocessing
       ↓
Diagnostic Feature Engineering
       ↓
Categorical Encoding
       ↓
Binary Readmission Target
       ↓
SMOTE Class Balancing
       ↓
Train-Test Split
       ↓
Decision Tree Classifier
       ↓
Performance Evaluation
       ↓
Feature Importance Analysis
```

## Key Takeaway

The project demonstrates an end-to-end machine learning workflow for a healthcare prediction problem, combining **exploratory data analysis, feature engineering, categorical encoding, class-imbalance handling, interpretable classification, and feature-importance analysis** to predict early diabetic patient readmission.
