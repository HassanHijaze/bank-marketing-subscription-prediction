# Bank Marketing Subscription Prediction

An end-to-end machine learning project that predicts whether a bank client will subscribe to a term deposit using customer attributes and historical marketing campaign data.

## Overview

This project addresses a binary classification problem in the context of direct bank marketing. The objective is to identify customers who are more likely to subscribe to a term deposit so that marketing efforts can be targeted more efficiently.

The workflow covers the full machine learning pipeline, including exploratory data analysis, leakage detection, feature engineering, preprocessing, model training, model comparison, threshold tuning, calibration analysis, final test evaluation, and feature importance interpretation.

## Business Problem

Banks often contact a large number of customers during marketing campaigns, but only a small proportion subscribe to the offered product. A predictive model can help prioritize high-potential customers, reduce unnecessary outreach, and improve campaign efficiency.

A key constraint in this problem is that the model should rely only on information available before or at the time of contact. For that reason, the `duration` feature was excluded from modeling because it introduces target leakage.

## Dataset

The dataset used in this project contains client information, campaign details, and outcomes from previous marketing contacts. It includes variables such as:

- demographic information
- financial status
- contact method
- campaign timing
- previous campaign outcome

Target variable:
- `y` → whether the client subscribed to a term deposit

## Project Workflow

### 1. Data Understanding and Exploration
- reviewed dataset structure and distributions
- examined class imbalance in the target variable
- explored relationships between customer attributes and subscription outcome

### 2. Data Quality and Leakage Check
- identified `duration` as a leakage feature
- removed leakage-prone information from the modeling dataset

### 3. Feature Engineering
Created additional features to improve signal capture:
- `age_band`
- `is_contacted_before`
- `was_previously_successful`

### 4. Preprocessing
Used a `ColumnTransformer` and `Pipeline` to ensure consistent preprocessing:
- numerical features → median imputation + standard scaling
- categorical features → most-frequent imputation + one-hot encoding

### 5. Model Training and Comparison
The following models were trained and evaluated:
- Logistic Regression
- Random Forest
- XGBoost

### 6. Hyperparameter Tuning
- applied `RandomizedSearchCV` to XGBoost
- used 5-fold cross-validation
- optimized for F1-score

### 7. Threshold Tuning
Because the target classes are imbalanced, the default classification threshold of 0.50 was not assumed to be optimal. Multiple thresholds were evaluated on the validation set, and the best threshold for F1-score was selected.

### 8. Model Evaluation
Models were assessed using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision

Additional evaluation included:
- confusion matrices
- ROC curve
- precision-recall curve
- calibration curve

### 9. Final Model
XGBoost was selected as the final model based on validation performance and business relevance. After threshold tuning, the final model was retrained on the combined training and validation data and evaluated on the holdout test set.

## Final Results

Final XGBoost test performance:

- **Threshold:** 0.21
- **Accuracy:** 0.8779
- **Precision:** 0.4801
- **Recall:** 0.5236
- **F1-score:** 0.5009
- **ROC-AUC:** 0.8055
- **Average Precision:** 0.4665

These results show that the model provides a useful balance between precision and recall while maintaining solid ranking performance on unseen data.

## Key Insights

- previous campaign success was one of the strongest predictive signals
- contact method and month of contact contributed meaningfully to performance
- excluding leakage features made the project more realistic and operationally relevant
- threshold tuning produced more practical results than relying on the default cutoff

## Technologies Used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- XGBoost

