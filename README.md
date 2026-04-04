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

- age: Age of the customer  
- job: Type of job held by the customer  
- marital: Marital status of the customer  
- education: Education level of the customer  
- default: Whether the customer has credit in default  
- housing: Whether the customer has a housing loan  
- loan: Whether the customer has a personal loan  
- contact: Whether the customer was contacted 
- month: Month of the last contact with the customer  
- duration: Duration of the last contact, in seconds  
- campaign: Number of contacts performed during the current campaign for this customer  
- pdays: Number of days since the customer was last contacted in a previous campaign; `999` means the customer was not previously contacted  
- previous: Number of contacts performed before the current campaign  
- poutcome: Outcome of the previous marketing campaign  


Target variable:
- y → whether the client subscribed to a term deposit

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
- age_band
- is_contacted_before
- was_previously_successful

### 4. Preprocessing
Used a `ColumnTransformer` and `Pipeline` to ensure consistent preprocessing:
- numerical features → median imputation + standard scaling
- categorical features → most-frequent imputation + one-hot encoding

### 5. Model Training and Comparison
The following models were trained and evaluated:
- Logistic Regression


## Technologies Used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn


