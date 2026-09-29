# Diabetes Readmission Prediction Using Machine Learning

## Overview

This project applies machine learning techniques to predict **early hospital readmission of diabetic patients within 30 days of discharge**. The analysis uses the UCI Diabetes 130-US Hospitals dataset, containing **101,766 hospital encounters and 50 attributes** covering demographic information, admission details, diagnoses, laboratory procedures, medications, and healthcare utilization.

The project focuses on developing an interpretable **Decision Tree classifier** for binary readmission prediction and comparing its performance with a **K-Nearest Neighbors (KNN)** classifier.

## Objectives

* Analyze patterns associated with early readmission among diabetic patients.
* Explore relationships between clinical, demographic, medication, and hospital-utilization variables.
* Clean and transform categorical and diagnostic variables for machine learning.
* Engineer hierarchical diagnostic and interaction features.
* Address substantial class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**.
* Train and evaluate Decision Tree and KNN classification models.
* Analyze feature importance to identify variables contributing to Decision Tree predictions.
* Evaluate models using accuracy, precision, recall, and ROC-AUC.

## Dataset

The dataset contains **101,766 hospital encounters with 50 attributes**.

Major feature groups include:

* **Demographics:** race, gender, age
* **Admission information:** admission type, admission source, discharge disposition
* **Hospital utilization:** time in hospital, outpatient visits, emergency visits, inpatient visits
* **Clinical information:** diagnoses, laboratory procedures, number of diagnoses
* **Medication information:** medication counts and diabetes medications
* **Diabetes-specific measurements:** glucose serum and HbA1c results
* **Target variable:** `readmitted`

The original `readmitted` variable contains three outcomes:

* `NO` — no readmission
* `>30` — readmission after 30 days
* `<30` — readmission within 30 days

For this project, the problem was converted into a binary classification task:

* **1:** `<30` — early readmission
* **0:** `NO` or `>30` — no early readmission

## Exploratory Data Analysis

Exploratory analysis was performed to investigate relationships between readmission and several clinical and hospital-utilization variables.

The analysis included:

* Distribution of early readmission cases
* Patient age and readmission
* Gender and readmission
* Race and readmission
* Number of medications
* Time spent in hospital
* Number of laboratory procedures
* Glucose serum test results
* HbA1c test results
* Diabetes medication usage
* Healthcare service utilization

These analyses were used to understand patterns in the data before model development.

## Data Preprocessing

The preprocessing pipeline consisted of several stages.

### 1. Data Cleaning

Features with large amounts of missing or unknown information, including `weight`, `payer_code`, and `medical_specialty`, were removed.

Records containing unknown values in key diagnostic, demographic, or admission-related fields were also filtered.

### 2. Diagnostic Feature Engineering

The original diagnosis codes were transformed into hierarchical diagnostic categories.

The diagnosis variables were grouped into broader clinical categories using ICD-9 code ranges, creating features such as:

* `level1_diag1`
* `level1_diag2`
* `level1_diag3`
* `level2_diag1`
* `level2_diag2`
* `level2_diag3`

This reduced the dimensionality of the raw diagnosis codes while retaining clinically meaningful diagnostic group information.

### 3. Admission Feature Transformation

Admission type, discharge disposition, and admission source categories were consolidated into fewer groups to reduce the number of categorical levels.

### 4. Medication Feature Engineering

Medication-related variables were transformed into numerical representations.

Additional features were created to capture:

* Number of medication changes
* Number of diabetes medications
* Overall medication utilization

### 5. Healthcare Utilization

A `service_utilization` feature was initially created from outpatient, emergency, and inpatient visit counts.

Highly skewed utilization variables were also examined, with log transformations applied where appropriate.

### 6. Interaction Features

Several interaction features were engineered to capture relationships between clinical and hospital-utilization variables.

Examples include:

* `num_medications | time_in_hospital`
* `num_medications | num_lab_procedures`
* `time_in_hospital | num_lab_procedures`
* `age | number_diagnoses`
* `change | num_medications`
* `num_medications | number_diagnoses`

These features allow the models to capture combined effects between variables rather than considering each variable independently.

### 7. Categorical Encoding

Categorical variables were converted into numerical representations using binary encoding and one-hot encoding.

Encoded variables included:

* Gender
* Race
* Admission type
* Discharge disposition
* Admission source
* Glucose serum results
* HbA1c results
* Diagnostic categories

## Class Imbalance

Early readmission represented a relatively small proportion of the dataset, resulting in substantial class imbalance.

The minority class was therefore oversampled using **SMOTE (Synthetic Minority Over-sampling Technique)**.

SMOTE generated synthetic observations for the early-readmission class to produce a balanced training dataset.

> **Important:** SMOTE was used to address class imbalance during model development. Model performance should therefore be interpreted using class-specific metrics such as precision, recall, and ROC-AUC rather than accuracy alone.

## Models

Two classification approaches were evaluated:

### Decision Tree

A Decision Tree classifier was selected because it provides an interpretable model structure and can capture nonlinear relationships between features.

The final configuration used:

```text
DecisionTreeClassifier(
    criterion='entropy',
    max_depth=28,
    min_samples_split=10
)
```

### K-Nearest Neighbors

A KNN classifier was also implemented as a comparative model.

Hyperparameters were evaluated using `GridSearchCV`.

The selected configuration was:

```text
{
    'n_neighbors': 3,
    'p': 2,
    'weights': 'uniform'
}
```

## Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* ROC-AUC

These metrics provide a more complete view of performance, particularly because the target variable is highly imbalanced.

## Results

### Decision Tree

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **81.73%** |
| Precision | **10.54%** |
| Recall    | **13.54%** |
| ROC-AUC   | **0.5112** |

### KNN

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **82.64%** |
| Precision | **10.51%** |
| Recall    | **12.15%** |
| ROC-AUC   | **0.5267** |

The results demonstrate the importance of evaluating class-specific performance rather than relying solely on overall accuracy. Although overall accuracy was above 80%, the relatively low precision, recall, and ROC-AUC indicate that predicting the minority early-readmission class remained challenging.

## Decision Tree Feature Importance

The Decision Tree's built-in feature importance scores were used to identify the most influential features.

The top features included:

| Feature                                  | Importance |
| ---------------------------------------- | ---------: |
| `number_inpatient_log1p`                 |   0.201157 |
| `change`                                 |   0.096470 |
| `num_medications \| num_lab_procedures`  |   0.040417 |
| `change \| num_medications`              |   0.036509 |
| `time_in_hospital \| num_lab_procedures` |   0.033127 |
| `age \| number_diagnoses`                |   0.032338 |
| `gender`                                 |   0.031935 |
| `num_lab_procedures`                     |   0.027609 |
| `num_medications \| time_in_hospital`    |   0.026941 |
| `num_medications \| number_diagnoses`    |   0.026505 |

`number_inpatient_log1p` had the highest Decision Tree feature importance, followed by `change` and several engineered interaction features.

These importance scores describe the variables used most by the fitted Decision Tree; they should not be interpreted as evidence that a feature independently causes readmission.

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **imbalanced-learn / SMOTE**
* **SciPy**
* **Google Colab**

## Project Workflow

```text
Diabetes Clinical Dataset
          ↓
Exploratory Data Analysis
          ↓
Data Cleaning
          ↓
Diagnostic Feature Engineering
          ↓
Medication & Utilization Features
          ↓
Interaction Feature Engineering
          ↓
Categorical Encoding
          ↓
Binary Readmission Target
          ↓
Class Imbalance Analysis
          ↓
SMOTE
          ↓
Train-Test Split
          ↓
Decision Tree + KNN
          ↓
Model Evaluation
          ↓
Feature Importance Analysis
```

## Key Takeaways

This project demonstrates an end-to-end machine learning workflow for a healthcare classification problem, including:

* Exploratory data analysis
* Clinical feature engineering
* Diagnostic code transformation
* Categorical encoding
* Interaction feature creation
* Class-imbalance handling with SMOTE
* Decision Tree classification
* KNN classification
* Hyperparameter tuning
* Model evaluation using multiple classification metrics
* Feature importance analysis

The analysis also highlights an important consideration in healthcare machine learning: **high overall accuracy does not necessarily indicate strong minority-class prediction performance when the target is highly imbalanced**. Precision, recall, and ROC-AUC therefore provide important additional information when evaluating early readmission prediction.

## Project Structure

```text
Diabetes-Readmission-Prediction/
│
├── README.md
├── diabetic_data.csv
├── diabetes_readmission_prediction.ipynb
└── requirements.txt
```

## Disclaimer

This project is intended for **educational and machine learning research purposes**. The predictions and feature-importance results should not be used for clinical decision-making or patient treatment without appropriate clinical validation.
