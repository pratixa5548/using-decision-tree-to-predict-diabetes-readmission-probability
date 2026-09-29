# Diabetes Hospital Readmission Prediction Using Machine Learning

## Overview

This project develops and evaluates machine learning models for predicting **early hospital readmission within 30 days of discharge among diabetic patients**.

The project uses a clinical dataset containing information about patient demographics, hospital admissions, diagnoses, laboratory procedures, medications, and healthcare utilization. An end-to-end machine learning workflow was developed covering **data cleaning, exploratory data analysis, feature engineering, categorical encoding, class-imbalance handling, model training, hyperparameter tuning, and model evaluation**.

Two classification approaches were evaluated:

* **Decision Tree Classifier**
* **K-Nearest Neighbors (KNN) Classifier**

The project also examines Decision Tree feature importance to identify the variables that contributed most strongly to the model's predictions.

---

## Objectives

The main objectives of the project are to:

* Analyze patterns associated with early hospital readmission among diabetic patients.
* Explore relationships between demographic, clinical, medication, and hospital-utilization variables and readmission.
* Clean and transform the clinical dataset into a machine-learning-ready format.
* Engineer meaningful diagnostic and interaction features.
* Convert the original three-class readmission outcome into a binary classification problem.
* Address the substantial class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**.
* Develop and compare Decision Tree and KNN classification models.
* Tune the KNN model using cross-validation.
* Evaluate model performance using accuracy, precision, recall, F1 score, and ROC-AUC.
* Analyze Decision Tree feature importance for model interpretability.

---

## Dataset

The project is based on a diabetes clinical dataset containing **101,766 patient encounters and 50 original attributes**.

The original dataset contains information relating to:

### Demographic Information

* Age
* Gender
* Race

### Hospital Admission Information

* Admission type
* Admission source
* Discharge disposition
* Time spent in hospital

### Clinical Information

* Primary, secondary, and tertiary diagnoses
* Number of diagnoses
* Number of laboratory procedures
* Number of procedures
* Glucose serum test results
* HbA1c test results

### Medication Information

The dataset contains information about several diabetes medications, including:

* Metformin
* Repaglinide
* Nateglinide
* Chlorpropamide
* Glimepiride
* Glipizide
* Glyburide
* Pioglitazone
* Rosiglitazone
* Acarbose
* Tolazamide
* Insulin
* Combination medications

### Healthcare Utilization

The dataset includes:

* Outpatient visits
* Emergency visits
* Inpatient visits

### Target Variable

The original `readmitted` variable contains three categories:

```text
NO
>30
<30
```

For this project, the target was converted into a binary classification problem:

```text
0 → No early readmission
1 → Readmitted within 30 days
```

The `<30` category was therefore treated as the positive class, while `NO` and `>30` were treated as the negative class.

---

## Final Dataset

After data cleaning, duplicate handling, feature engineering, and preprocessing, the final modelling dataset contained:

```text
Patient records: 67,580
Model features: 71
```

The final target distribution was:

| Readmission Class             | Number of Patients |
| ----------------------------- | -----------------: |
| 0 — No early readmission      |             61,451 |
| 1 — Readmitted within 30 days |              6,129 |

This corresponds to a strongly imbalanced classification problem, with early readmission representing approximately **9.1% of the final dataset**.

Because of this imbalance, accuracy alone is not sufficient to evaluate model performance. Precision, recall, F1 score, and ROC-AUC were therefore also considered.

---

## Exploratory Data Analysis

Exploratory analysis was performed to investigate relationships between patient characteristics and readmission.

The analysis examined variables including:

* Age
* Gender
* Race
* Time in hospital
* Number of medications
* Number of laboratory procedures
* Number of procedures
* Healthcare service utilization
* Glucose serum results
* HbA1c results
* Diabetes medication status
* Admission characteristics

Visualizations were created using **Matplotlib** and **Seaborn**, including:

* Readmission class distributions
* Count plots
* Kernel density plots
* Feature distributions
* Readmission comparisons across demographic groups
* Readmission comparisons across clinical variables

These visualizations were used to understand the structure of the dataset and identify potential relationships before model development.

---

## Data Preprocessing

Several preprocessing steps were performed to convert the raw clinical dataset into a suitable format for machine learning.

### 1. Removing Uninformative or Highly Missing Variables

Variables with extensive missing information or limited modelling usefulness were removed, including:

```text
weight
payer_code
medical_specialty
```

Additional medication variables with extremely limited variation were also removed.

---

### 2. Handling Missing and Unknown Values

Records containing unknown or invalid values in selected important variables were removed.

This included records with unavailable:

* Diagnosis codes
* Race
* Gender
* Certain discharge dispositions

Discharge disposition corresponding to death or hospice-related outcomes was also excluded from the modelling population.

---

### 3. Duplicate Patient Handling

The dataset contains multiple encounters for some patients.

Duplicate patient records were therefore handled using `patient_nbr`, retaining the first available encounter for each patient in the modelling dataset.

This reduced repeated patient observations and resulted in the final modelling population of:

```text
67,580 patients
```

---

### 4. Medication Feature Engineering

Medication variables were converted into numerical representations.

Medication status such as:

```text
No
Steady
Up
Down
```

was transformed into binary indicators representing whether the medication was active or changed.

Two additional medication-related features were engineered:

```text
numchange
nummed
```

`numchange` represents the number of medications whose status indicated a change.

`nummed` represents the number of medications recorded as active.

---

### 5. Healthcare Utilization Feature Engineering

A service utilization feature was initially created from:

```text
number_outpatient
number_emergency
number_inpatient
```

The resulting utilization information was explored during EDA.

The original utilization variables were subsequently transformed using logarithmic transformations where appropriate to reduce skewness.

---

### 6. Diagnostic Feature Engineering

The original diagnosis variables contain ICD-9 diagnosis codes.

Three primary diagnosis variables were processed:

```text
diag_1
diag_2
diag_3
```

Hierarchical diagnostic representations were created to group diagnosis codes into broader clinical categories.

The resulting diagnostic features included:

```text
level1_diag1
level1_diag2
level1_diag3

level2_diag1
level2_diag2
level2_diag3
```

These features grouped diagnosis codes into broader disease categories rather than treating every individual diagnosis code as a separate feature.

---

### 7. Categorical Encoding

Categorical variables were converted into machine-learning-compatible numerical representations.

One-hot encoding was applied to variables including:

* Gender
* Race
* Admission type
* Discharge disposition
* Admission source
* Maximum glucose serum result
* HbA1c result
* Diagnostic category

This produced the final numerical feature matrix used by the classification models.

---

### 8. Age Transformation

The original age ranges were converted into numerical categories and subsequently mapped to representative midpoint values.

For example, age ranges were represented using values such as:

```text
5
15
25
35
45
55
65
75
85
95
```

This allowed age to be incorporated as a numerical modelling feature.

---

### 9. Log Transformation

Several highly skewed numerical variables were examined using skewness and kurtosis.

Where appropriate, logarithmic transformations were applied using:

```python
log()
```

or

```python
log1p()
```

The resulting transformed features included variables such as:

```text
number_outpatient_log1p
number_emergency_log1p
number_inpatient_log1p
```

These transformations were used to reduce the influence of highly skewed distributions.

---

### 10. Interaction Features

Additional interaction terms were engineered to capture relationships between clinical and hospital-utilization variables.

Examples include:

```text
num_medications | time_in_hospital
num_medications | num_procedures
time_in_hospital | num_lab_procedures
num_medications | num_lab_procedures
num_medications | number_diagnoses
age | number_diagnoses
change | num_medications
number_diagnoses | time_in_hospital
num_medications | numchange
```

These features represent multiplicative interactions between the corresponding variables.

---

## Class Imbalance

The final target distribution was substantially imbalanced:

```text
Class 0: 61,451
Class 1:  6,129
```

Without addressing this imbalance, a model could achieve relatively high accuracy while performing poorly at identifying the minority early-readmission cases.

To address this problem, **SMOTE** was used on the training data.

SMOTE generates synthetic examples of the minority class rather than simply duplicating existing observations.

The modelling workflow therefore used the following structure:

```text
Original Dataset
       ↓
Train / Test Split
       ↓
SMOTE on Training Data
       ↓
Balanced Training Dataset
       ↓
Model Training
       ↓
Evaluation on Test Data
```

The test set was kept separate from SMOTE so that model evaluation could be performed on data that was not synthetically oversampled.

---

# Machine Learning Models

## 1. Decision Tree

A Decision Tree classifier was selected because it can model nonlinear relationships and provides an interpretable feature-importance measure.

The model configuration used:

```python
DecisionTreeClassifier(
    criterion='entropy',
    max_depth=28,
    min_samples_split=10
)
```

The Decision Tree was trained using the SMOTE-balanced training data.

---

## 2. K-Nearest Neighbors

A KNN classifier was also developed for comparison.

Because KNN is distance-based, the input variables were standardized using:

```python
StandardScaler
```

The scaler was fitted using the training data and then applied to the test data.

Hyperparameter tuning was performed using `GridSearchCV` with ROC-AUC as the optimization metric.

The final selected KNN configuration was:

```text
n_neighbors = 5
p = 2
weights = uniform
```

---

# Model Evaluation

The models were evaluated using multiple classification metrics:

### Accuracy

Measures the overall proportion of correctly classified observations.

### Precision

Measures the proportion of predicted positive cases that were actually early readmissions.

### Recall

Measures the proportion of actual early-readmission cases correctly identified by the model.

### F1 Score

Provides a harmonic mean of precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across different classification thresholds.

Because the target variable is highly imbalanced, particular attention was given to precision, recall, F1 score, and ROC-AUC rather than relying solely on accuracy.

---

# Results

The final models produced the following results on the test set:

| Model         |   Accuracy |  Precision |     Recall |   F1 Score |    ROC-AUC |
| ------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Decision Tree | **0.8212** | **0.1111** | **0.1387** | **0.1234** | **0.5205** |
| KNN           | **0.8273** | **0.1226** | **0.1468** | **0.1336** | **0.5481** |

### KNN Parameters

The best-performing KNN configuration identified through grid search was:

```text
{
    'n_neighbors': 5,
    'p': 2,
    'weights': 'uniform'
}
```

---

## Interpretation of Results

The models achieved overall accuracy above 82%, but accuracy should be interpreted cautiously because the dataset contains a substantial class imbalance.

The relatively low precision, recall, F1 score, and ROC-AUC indicate that the models had **limited ability to reliably distinguish early-readmission cases from non-early-readmission cases**.

The KNN model produced a ROC-AUC of approximately **0.548**, while the Decision Tree produced approximately **0.521**.

Therefore, the results demonstrate that the modelling pipeline successfully implemented the classification task, but the current feature set and modelling approaches do **not provide strong predictive discrimination for early readmission**.

This is an important finding rather than simply focusing on the accuracy score. In an imbalanced healthcare prediction problem, a model that achieves high accuracy but has weak minority-class detection may have limited practical predictive value.

---

# Decision Tree Feature Importance

The Decision Tree's built-in feature importance scores were examined to identify which variables contributed most strongly to its predictions.

The ten highest-ranked features were:

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

The largest feature importance was associated with:

```text
number_inpatient_log1p
```

followed by:

```text
change
```

and several engineered interaction features involving medication usage, laboratory procedures, age, diagnoses, and time in hospital.

These feature-importance values describe the variables used most heavily by the fitted Decision Tree. They should not be interpreted as evidence that an individual feature independently causes hospital readmission.

---

# Project Workflow

```text
                  Clinical Dataset
                        │
                        ▼
              Exploratory Data Analysis
                        │
                        ▼
                Data Cleaning
                        │
                        ▼
             Duplicate Patient Handling
                        │
                        ▼
            Diagnostic Feature Engineering
                        │
                        ▼
             Medication Feature Engineering
                        │
                        ▼
             Interaction Feature Engineering
                        │
                        ▼
             Categorical Variable Encoding
                        │
                        ▼
              Binary Target Construction
                        │
                        ▼
                 Train/Test Split
                        │
                        ▼
              SMOTE on Training Data
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
      Decision Tree               KNN
             │                     │
             │              Standard Scaling
             │                     │
             │              Hyperparameter
             │                 Tuning
             │                     │
             └──────────┬──────────┘
                        ▼
                Model Evaluation
                        │
                        ▼
          Accuracy / Precision / Recall
                 F1 / ROC-AUC
                        │
                        ▼
             Feature Importance Analysis
```

---

# Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **imbalanced-learn**
* **SciPy**
* **Google Colab**
* **GitHub**

---

# Key Skills Demonstrated

This project demonstrates experience with:

* Data cleaning
* Exploratory data analysis
* Healthcare data preprocessing
* Feature engineering
* Diagnostic code transformation
* Categorical encoding
* Log transformations
* Interaction-term engineering
* Class-imbalance handling
* SMOTE
* Decision Tree classification
* KNN classification
* Hyperparameter tuning
* Feature scaling
* Cross-validation
* Classification evaluation
* ROC-AUC analysis
* Model interpretability
* Data visualization
* Python-based machine learning workflows

---

# Limitations

Several limitations should be considered when interpreting the results.

### Class imbalance

Early readmission represents a relatively small proportion of the final dataset. Although SMOTE was used on the training data, minority-class prediction remains challenging.

### Model performance

The relatively low ROC-AUC, precision, recall, and F1 scores indicate that the current models have limited predictive discrimination.

### Feature limitations

The available variables may not capture all factors associated with hospital readmission, such as detailed socioeconomic information, clinical severity, treatment decisions, follow-up care, or other patient-level characteristics.

### Feature importance

Decision Tree feature importance indicates how heavily features were used by the fitted model. It does not establish causation or clinical importance.

### Clinical application

The models developed in this project are intended for **educational and analytical purposes** and should not be interpreted as validated clinical decision-support systems.

---

# Conclusion

This project demonstrates an end-to-end machine learning workflow for a highly imbalanced healthcare classification problem.

The analysis progressed from raw clinical data through exploratory analysis, preprocessing, diagnostic feature engineering, medication and interaction features, categorical encoding, class-imbalance handling, model development, hyperparameter tuning, and evaluation.

Both Decision Tree and KNN models were evaluated using multiple performance metrics. While the models achieved overall accuracy above 82%, the relatively low ROC-AUC, precision, recall, and F1 scores demonstrate the difficulty of accurately identifying early hospital readmissions in this dataset.

The project therefore highlights an important practical lesson in healthcare machine learning: **model evaluation should consider class imbalance and minority-class performance rather than relying on accuracy alone**.

---

## Repository Structure

```text
Diabetes-Readmission-Prediction/
│
├── Diabetes_Readmission_Prediction.ipynb
├── README.md
├── requirements.txt
└── data/
    └── diabetic_data.csv
```

> The dataset may need to be obtained separately depending on its licensing and distribution terms.

---

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Diabetes-Readmission-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

The project was developed in Google Colab and can also be run using Jupyter Notebook.

```bash
jupyter notebook
```

Open:

```text
Diabetes_Readmission_Prediction.ipynb
```

### 4. Provide the dataset

Upload or place the required dataset in the expected location before executing the notebook.

---

## Author

**Pratiksha Kamath**

B.Tech Information Technology

Machine Learning | Data Analytics | Python
