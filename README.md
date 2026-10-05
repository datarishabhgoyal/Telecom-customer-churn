# Telco Customer Churn Prediction using Logistic Regression

## 📌 Project Overview

Customer churn is one of the major challenges faced by telecommunication companies. When customers stop using a company's services, the business loses recurring revenue and may also have to spend additional money to acquire new customers.

The objective of this project is to analyze customer data, understand the factors associated with customer churn, and build a **Logistic Regression classification model** to predict whether a customer is likely to churn.

This project covers the complete basic machine learning workflow:

- Data Understanding
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Data Preparation
- Feature Selection
- Encoding Categorical Variables
- Feature Scaling
- Train-Test Split
- Logistic Regression
- Model Prediction
- Model Evaluation
- Interpretation of Classification Results

---

## 🎯 Project Objective

The main objective of this project is:

> **To predict whether a telecom customer will churn based on their demographic information, service usage, contract details, and billing information.**

The target variable is:

- `Churn = Yes` → Customer has churned
- `Churn = No` → Customer has not churned

This makes the problem a **binary classification problem**.

---

## 📊 Dataset

The dataset contains information about telecom customers and their relationship with the company.

### Dataset Size

- **Rows:** 7,043
- **Columns:** 21
- **Target Variable:** `Churn`

### Main Features

| Feature | Description |
|---|---|
| `customerID` | Unique identifier of the customer |
| `gender` | Customer gender |
| `SeniorCitizen` | Whether the customer is a senior citizen |
| `Partner` | Whether the customer has a partner |
| `Dependents` | Whether the customer has dependents |
| `tenure` | Number of months the customer has stayed with the company |
| `PhoneService` | Whether the customer has phone service |
| `MultipleLines` | Multiple phone line information |
| `InternetService` | Type of internet service |
| `OnlineSecurity` | Online security service |
| `OnlineBackup` | Online backup service |
| `DeviceProtection` | Device protection service |
| `TechSupport` | Technical support service |
| `StreamingTV` | Streaming TV service |
| `StreamingMovies` | Streaming movie service |
| `Contract` | Customer contract type |
| `PaperlessBilling` | Whether paperless billing is enabled |
| `PaymentMethod` | Customer payment method |
| `MonthlyCharges` | Monthly amount charged to the customer |
| `TotalCharges` | Total amount charged to the customer |
| `Churn` | Whether the customer left the company |

---

# 🔎 1. Data Understanding

The first step was to understand the structure and characteristics of the dataset.

The dataset contains both:

### Numerical Features

- `SeniorCitizen`
- `tenure`
- `MonthlyCharges`
- `TotalCharges` after conversion to numeric format

### Categorical Features

- `gender`
- `Partner`
- `Dependents`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `Churn`

The `customerID` column is an identifier and does not provide meaningful predictive information for the model.

---

# 🧹 2. Data Cleaning

Data cleaning was performed before building the machine learning model.

## Missing / Invalid Values

The `TotalCharges` column was stored as an object/string column instead of a numerical column.

During the investigation, blank values were identified in this column.

These records were handled before proceeding with the machine learning workflow.

After cleaning:

- The dataset contained **7,032 usable records**
- Duplicate rows were checked
- No duplicate rows were found

---

# 📈 3. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand customer behavior and identify patterns related to churn.

The target variable was also analyzed to understand the distribution between customers who stayed and customers who churned.

### Churn Distribution

Approximately:

- **73.46%** of customers did not churn
- **26.54%** of customers churned

This shows that the target variable is not perfectly balanced.

Therefore, evaluating the model using only accuracy would not be sufficient. Metrics such as:

- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

are also important.

---

# 🧠 4. Feature Selection

Before training the model, features that do not provide useful predictive information were considered for removal.

For example:

### `customerID`

`customerID` is a unique identifier and should not be used as a predictive feature.

Using identifiers as model features can cause the model to learn meaningless patterns.

The target variable:

### `Churn`

was separated from the input features because it is the variable that the model needs to predict.

---

# 🔄 5. Data Preparation

Machine learning algorithms require data to be represented in a numerical format.

Since the dataset contains many categorical variables, categorical features were converted into numerical representations before model training.

The general preparation process included:

1. Separating features and target
2. Handling categorical variables
3. Converting categorical values into numerical values
4. Preparing numerical features
5. Scaling features where required
6. Splitting the dataset into training and testing sets

---

# 🏷️ 6. Target Encoding

The target variable `Churn` contains two categories:

```text
No
Yes
```

For binary classification, these values were represented numerically so that the machine learning model could work with them.

Conceptually:

```text
No  → 0
Yes → 1
```

This allows the model to predict the probability of customer churn.

---

# ⚖️ 7. Feature Scaling

Feature scaling was considered because Logistic Regression can be affected by differences in the scale of numerical features.

For example:

- `tenure` represents months
- `MonthlyCharges` represents monetary values
- `TotalCharges` can have much larger numerical values

Scaling helps put numerical features on a comparable scale.

The project uses feature scaling as part of the preprocessing workflow before Logistic Regression.

---

# ✂️ 8. Train-Test Split

The prepared dataset was divided into two parts:

### Training Dataset

Used to train the Logistic Regression model.

### Testing Dataset

Used to evaluate how well the trained model performs on unseen data.

A common split used for this type of project is:

```text
80% → Training Data
20% → Testing Data
```

The important purpose of the train-test split is to evaluate whether the model can generalize to new customers rather than simply memorizing the training data.

---

# 🤖 9. Logistic Regression

Logistic Regression was selected as the primary machine learning algorithm.

Although its name contains "Regression", Logistic Regression is commonly used for **classification problems**.

In this project, the model predicts:

```text
0 → Customer will not churn
1 → Customer will churn
```

The model estimates the probability that a customer belongs to the churn class.

The basic idea is:

```text
Customer Features
       ↓
Logistic Regression
       ↓
Churn Probability
       ↓
Classification Threshold
       ↓
Churn / No Churn
```

For example, if the model predicts:

```text
Probability of Churn = 0.82
```

and the classification threshold is `0.50`, the customer would be classified as:

```text
Churn = Yes
```

---

# 🔮 10. Model Prediction

After training the Logistic Regression model, predictions were generated using the testing dataset.

Two types of predictions can be obtained:

### Class Prediction

```python
model.predict(X_test)
```

This returns the predicted class:

```text
0 or 1
```

### Probability Prediction

```python
model.predict_proba(X_test)
```

This returns the probability associated with each class.

Probability predictions are particularly useful in churn problems because businesses may want to identify customers with a high probability of leaving.

---

# 📊 11. Model Evaluation

The model should not be evaluated using accuracy alone.

Several evaluation techniques are useful for a churn classification problem.

## Confusion Matrix

The confusion matrix helps identify:

- True Positives
- True Negatives
- False Positives
- False Negatives

For churn prediction:

### True Positive

The model correctly predicts that a customer will churn.

### True Negative

The model correctly predicts that a customer will not churn.

### False Positive

The model predicts churn, but the customer does not actually churn.

### False Negative

The model predicts that the customer will stay, but the customer actually churns.

False negatives can be particularly important in a churn problem because the company may miss customers who are actually at risk of leaving.

---

# 📐 12. Classification Metrics

The following metrics are useful for evaluating the model.

### Accuracy

Measures the percentage of total predictions that are correct.

### Precision

Measures how many customers predicted as churners actually churned.

### Recall

Measures how many of the actual churners were correctly identified.

### F1-Score

The harmonic mean of precision and recall.

F1-score is useful when we want a balance between precision and recall.

### ROC-AUC

ROC-AUC measures the model's ability to distinguish between churn and non-churn customers across different classification thresholds.

---

# 📌 13. Why Recall Matters in Churn Prediction

In a customer churn problem, recall can be particularly important.

Suppose a customer is actually going to leave but the model predicts:

```text
No Churn
```

The company may never take any retention action.

Therefore, a good churn model should try to identify as many potential churners as possible while still maintaining reasonable precision.

The ideal balance depends on the business objective.

For example:

- If retention campaigns are expensive → Precision may become more important.
- If losing a customer is very expensive → Recall may become more important.

---

# 💡 14. Business Interpretation

The machine learning model can help a telecom company identify customers who are more likely to churn.

The predicted churn probability can potentially be used to:

- Identify high-risk customers
- Prioritize retention campaigns
- Offer personalized plans
- Provide discounts or incentives
- Improve customer support
- Investigate contract-related churn
- Reduce customer acquisition costs by improving retention

The model should therefore be viewed not only as a machine learning exercise but also as a tool that can support customer retention decisions.

---

# 🔬 15. Project Workflow

The complete workflow followed in this project can be summarized as:

```text
Raw Dataset
     ↓
Data Understanding
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Selection
     ↓
Data Preparation
     ↓
Categorical Encoding
     ↓
Feature Scaling
     ↓
Train-Test Split
     ↓
Logistic Regression
     ↓
Predictions
     ↓
Confusion Matrix
     ↓
Classification Metrics
     ↓
Model Interpretation
     ↓
Business Insights
```

---

# 🛠️ Technologies Used

The project was developed using Python and Google Colab.

### Programming Language

- Python

### Libraries

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### Environment

- Google Colab
- Jupyter Notebook

---

# 📂 Project Files

The repository contains the following main files:

```text
Telco-Customer-Churn/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── Telco Customer Churn.ipynb
└── README.md
```

### Dataset

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

Contains the original customer churn dataset used for analysis and modeling.

### Notebook

`Telco Customer Churn.ipynb`

Contains the complete Python analysis, data preprocessing, exploratory analysis, machine learning workflow, and model evaluation.

### README

This file provides an overview of the project, methodology, workflow, and business interpretation.

---

# 📚 Key Concepts Learned

This project helped reinforce several important Data Analytics and Machine Learning concepts:

- Classification problems
- Binary classification
- Data cleaning
- Missing value handling
- Duplicate detection
- Exploratory Data Analysis
- Categorical encoding
- Feature selection
- Data preprocessing
- Feature scaling
- Train-test split
- Logistic Regression
- Probability prediction
- Classification threshold
- Confusion Matrix
- Precision
- Recall
- F1-score
- ROC-AUC
- Business interpretation of machine learning results

---

# 🚀 Future Improvements

The project can be further improved by:

- Comparing Logistic Regression with Decision Tree
- Comparing Logistic Regression with Random Forest
- Performing hyperparameter tuning
- Handling class imbalance using appropriate techniques
- Performing cross-validation
- Performing feature importance analysis
- Optimizing the classification threshold
- Building a customer churn dashboard
- Deploying the model as a simple web application
- Creating a customer-level churn risk score

---

# 🎯 Conclusion

This project demonstrates a complete beginner-to-intermediate machine learning workflow for a real-world customer churn problem.

The main goal was not only to train a Logistic Regression model but also to understand the complete process involved in preparing real-world data for machine learning.

The project covers the journey from:

**Raw Customer Data → Data Cleaning → EDA → Feature Preparation → Logistic Regression → Model Evaluation → Business Interpretation**

This project also demonstrates how machine learning can be applied to a practical business problem where identifying customers at risk of churn can help a telecom company improve customer retention.

---

## 👨‍💻 Author

**Rishabh Goyal**

Data Analytics | Python | SQL | Machine Learning

---

## ⭐ If you found this project useful

Feel free to explore the notebook, review the analysis, and use the project as a reference for learning customer churn prediction and Logistic Regression.
