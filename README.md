# Diabetes Hospital Readmission Prediction

## Overview

This project applies machine learning to predict **early hospital readmission within 30 days of discharge** among patients with diabetes.

The project uses the **Diabetes 130-US hospitals for years 1999–2008** dataset and develops an end-to-end machine learning pipeline involving data cleaning, exploratory data analysis, diagnostic feature engineering, categorical encoding, interaction features, class-imbalance handling, model training, hyperparameter tuning, and model evaluation.

The final modelling dataset contains **67,580 patient encounters** and **71 model features**.

Two classification approaches were evaluated:

* **Decision Tree Classifier**
* **K-Nearest Neighbors (KNN)** with hyperparameter tuning

The project also uses Decision Tree feature importance to examine which variables contributed most strongly to model predictions.

---

## Objectives

The main objectives of the project are to:

* Explore factors associated with early hospital readmission among diabetic patients.
* Perform exploratory analysis of demographic, clinical, medication, and hospital-utilization variables.
* Clean and preprocess healthcare data for machine learning.
* Engineer diagnostic and interaction features.
* Convert categorical variables into numerical representations.
* Address class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**.
* Develop an interpretable Decision Tree classifier.
* Develop and tune a KNN classifier.
* Compare model performance using classification metrics and ROC-AUC.
* Analyze Decision Tree feature importance.

---

## Dataset

The project is based on the **Diabetes 130-US hospitals for years 1999–2008** dataset.

The original dataset contains:

* **101,766 patient encounters**
* **50 original attributes**
* Data collected from **130 US hospitals**
* Patient demographic, admission, diagnostic, medication, laboratory, and hospital-utilization information.

The original dataset contains three readmission outcomes:

| Original Value | Meaning                                  |
| -------------- | ---------------------------------------- |
| `NO`           | No readmission within the defined period |
| `>30`          | Readmission after 30 days                |
| `<30`          | Readmission within 30 days               |

For this project, the target was converted into a binary classification problem:

* `1` → **Readmitted within 30 days**
* `0` → **Not readmitted within 30 days**

The final processed dataset used for modelling contains:

**67,580 patient records and 71 model features.**

---

## Features

The dataset contains information from several categories.

### Demographic Features

* Age
* Gender
* Race

### Admission and Hospital Information

* Admission type
* Admission source
* Discharge disposition
* Time spent in hospital
* Number of procedures
* Number of laboratory procedures
* Number of medications

### Healthcare Utilization

* Number of emergency visits
* Number of inpatient visits
* Number of outpatient visits

### Diagnostic Information

* Primary diagnosis
* Secondary diagnoses
* Tertiary diagnoses
* Hierarchical diagnostic categories
* Number of diagnoses

### Medication Information

The dataset contains information about several diabetes medications, including:

* Metformin
* Repaglinide
* Nateglinide
* Glimepiride
* Glipizide
* Glyburide
* Pioglitazone
* Rosiglitazone
* Acarbose
* Tolazamide
* Insulin
* Combination medications

### Diabetes-Specific Measurements

* HbA1c results
* Maximum glucose serum results
* Diabetes medication changes

---

## Exploratory Data Analysis

Exploratory analysis was performed to investigate relationships between readmission and important clinical and hospital-utilization variables.

The analysis included visualizations examining:

* Distribution of readmission outcomes
* Time spent in hospital
* Patient age
* Race
* Gender
* Number of medications
* Diabetes medication prescription
* Healthcare service utilization
* Glucose serum test results
* HbA1c results
* Number of laboratory procedures

Kernel density plots and categorical count plots were used to explore differences between patients who were and were not readmitted within 30 days.

---

## Data Preprocessing

The preprocessing pipeline consisted of several stages.

### 1. Target Transformation

The original three-category readmission variable was converted into a binary target:

```text
0 → Not readmitted within 30 days
1 → Readmitted within 30 days
```

### 2. Duplicate Patient Handling

Duplicate encounters belonging to the same patient were handled to create the final modelling population.

### 3. Missing-Value Handling

Missing and unknown categorical values were processed before model training.

### 4. Diagnostic Feature Engineering

Diagnosis codes were transformed into hierarchical diagnostic categories to make the clinical information more suitable for machine learning.

### 5. Categorical Encoding

Categorical variables were converted into numerical representations using encoding techniques such as one-hot encoding.

Encoded variables included:

* Gender
* Race
* Admission type
* Discharge disposition
* Admission source
* Glucose serum result
* HbA1c result
* Diagnostic categories

### 6. Log Transformations

Highly skewed healthcare-utilization variables were transformed using logarithmic transformations where appropriate.

For example:

```text
number_inpatient_log1p
number_outpatient_log1p
number_emergency_log1p
```

### 7. Interaction Features

Additional interaction features were created to capture relationships between clinical and hospital-utilization variables.

Examples include:

```text
num_medications | time_in_hospital
num_medications | num_lab_procedures
time_in_hospital | num_lab_procedures
num_medications | number_diagnoses
age | number_diagnoses
change | num_medications
```

---

## Class Imbalance

The final modelling dataset remained highly imbalanced.

The target distribution was:

| Readmission |      Count |
| ----------- | ---------: |
| `0`         |     61,451 |
| `1`         |      6,129 |
| **Total**   | **67,580** |

This imbalance means that a model could achieve relatively high accuracy by predominantly predicting the majority class.

Therefore, **SMOTE (Synthetic Minority Over-sampling Technique)** was incorporated into the modelling workflow to improve representation of the minority class during model training.

Because early readmission is the minority class, metrics such as **precision, recall, F1-score, and ROC-AUC** were considered alongside accuracy.

---

## Machine Learning Models

Two classification models were evaluated.

### 1. Decision Tree

A Decision Tree classifier was used because of its interpretability and ability to capture nonlinear relationships between features.

The model was configured using:

```python
DecisionTreeClassifier(
    criterion="entropy",
    max_depth=28,
    min_samples_split=10
)
```

Decision Tree feature importance was subsequently used to identify the variables that contributed most strongly to the model's predictions.

---

### 2. K-Nearest Neighbors

A KNN classifier was also developed as a comparison model.

Because KNN is distance-based, feature scaling was performed before model training.

Hyperparameters were tuned using **GridSearchCV** with ROC-AUC as the model-selection metric.

The search considered:

```text
n_neighbors: 3, 5, 7
weights: uniform, distance
p: 1, 2
```

The best configuration identified by the search was:

```text
n_neighbors = 5
p = 2
weights = uniform
```

---

## Model Evaluation

The final models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

These metrics provide a more complete evaluation than accuracy alone, particularly because the target variable is highly imbalanced.

---

## Results

### Decision Tree

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **0.8212** |
| Precision | **0.1111** |
| Recall    | **0.1387** |
| F1-score  | **0.1234** |
| ROC-AUC   | **0.5205** |

### KNN

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **0.8273** |
| Precision | **0.1226** |
| Recall    | **0.1468** |
| F1-score  | **0.1336** |
| ROC-AUC   | **0.5481** |

### Best KNN Parameters

```text
{
    'knn__n_neighbors': 5,
    'knn__p': 2,
    'knn__weights': 'uniform'
}
```

The KNN model achieved a ROC-AUC of **0.5481**, while the Decision Tree achieved **0.5205** on the evaluated test set.

The relatively low ROC-AUC values indicate that the models had limited ability to distinguish early readmission cases from non-early-readmission cases. This is an important limitation of the current modelling approach and is reported rather than relying on accuracy alone.

---

## Decision Tree Feature Importance

The ten features with the highest Decision Tree feature-importance scores were:

| Rank | Feature                                | Importance |
| ---: | -------------------------------------- | ---------: |
|    1 | `number_inpatient_log1p`               |   0.204426 |
|    2 | `change`                               |   0.081522 |
|    3 | `num_medications\|num_lab_procedures`  |   0.038743 |
|    4 | `gender_1`                             |   0.032886 |
|    5 | `time_in_hospital\|num_lab_procedures` |   0.031417 |
|    6 | `change\|num_medications`              |   0.028935 |
|    7 | `num_lab_procedures`                   |   0.028816 |
|    8 | `age\|number_diagnoses`                |   0.027995 |
|    9 | `age`                                  |   0.027420 |
|   10 | `num_medications\|time_in_hospital`    |   0.026094 |

The most influential feature in the Decision Tree was:

```text
number_inpatient_log1p
```

with an importance score of approximately **0.2044**.

Feature importance is used here to describe the variables contributing to the model's predictions. It should not be interpreted as establishing clinical causation or independent statistical association.

---

## Project Workflow

```text
Original Diabetes Dataset
          ↓
Data Cleaning
          ↓
Exploratory Data Analysis
          ↓
Target Transformation
          ↓
Patient-Level Deduplication
          ↓
Diagnostic Feature Engineering
          ↓
Categorical Encoding
          ↓
Log Transformations
          ↓
Interaction Feature Engineering
          ↓
Final Dataset
67,580 Records × 71 Features
          ↓
Class Imbalance Handling
          ↓
SMOTE
          ↓
Train/Test Modelling
          ↓
 ┌───────────────────────┐
 │                       │
 ▼                       ▼
Decision Tree            KNN
 │                       │
 ▼                       ▼
Feature Importance       GridSearchCV
 │                       │
 └───────────┬───────────┘
             ↓
     Model Evaluation
             ↓
Accuracy / Precision /
Recall / F1 / ROC-AUC
```

---

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **SciPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **imbalanced-learn**
* **Google Colab**
* **GitHub**

---

## Key Takeaways

This project demonstrates an end-to-end machine learning workflow for a healthcare prediction problem, including:

* Exploratory data analysis
* Healthcare data preprocessing
* Diagnostic feature engineering
* Categorical encoding
* Log transformations
* Interaction feature engineering
* Class-imbalance handling using SMOTE
* Decision Tree classification
* KNN classification
* Hyperparameter tuning
* Model evaluation using multiple classification metrics
* Feature-importance analysis

The results also demonstrate an important aspect of applied machine learning: **high accuracy alone does not necessarily indicate strong predictive performance when the target classes are highly imbalanced**. For this reason, precision, recall, F1-score, and ROC-AUC were included in the evaluation.

---

## Limitations

Several limitations should be considered when interpreting the results:

1. The dataset is substantially imbalanced toward patients who were not readmitted within 30 days.
2. The relatively low ROC-AUC values indicate limited discriminative performance from the evaluated models.
3. Decision Tree feature importance indicates contribution to model predictions rather than clinical causation.
4. The dataset represents historical hospital encounters from 1999–2008 and may not fully represent current healthcare practices.
5. The models are intended as an academic machine-learning exercise and should not be interpreted as clinical decision-support systems.

---

## Repository Structure

```text
Diabetes-Readmission-Prediction/
│
├── fdsreadmission.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

### Main Notebook

`fdsreadmission.ipynb` contains the complete workflow, including:

* Data loading
* Data cleaning
* Exploratory analysis
* Feature engineering
* Data preprocessing
* SMOTE
* Decision Tree modelling
* KNN modelling
* Hyperparameter tuning
* Model evaluation
* Feature importance analysis

---

## Conclusion

This project explores the use of machine learning to predict **early hospital readmission among diabetic patients**. The workflow combines healthcare data preprocessing, feature engineering, class-imbalance handling, interpretable Decision Tree modelling, KNN classification, hyperparameter optimization, and multi-metric evaluation.

Although the final models demonstrate limited discriminative performance based on ROC-AUC, the project provides a complete practical example of applying machine-learning techniques to a highly imbalanced healthcare classification problem and highlights the importance of selecting appropriate evaluation metrics beyond accuracy.
