# Dataset

## Credit Card Fraud Detection Dataset

Source:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

This is one of the datasets used in the parent paper for this project.
The dataset is publicly accessible, allowing the experiments from the
parent paper to be reproduced and extended during the implementation phase.

The dataset contains transactions made by European cardholders.

The dataset is highly imbalanced and contains 284,807 transactions,
including 492 fraudulent transactions.

## Main Variables

- Time
- V1-V28
- Amount
- Class

`Class` is the target variable:

- 0 = Normal transaction
- 1 = Fraudulent transaction

## Use in This Project

This dataset will be used to train and evaluate machine-learning
models for credit card fraud detection.

The project will investigate techniques including:

- XGBoost
- Bayesian Optimization
- SMOTE
- Random Undersampling
- Model performance comparison
