
# Baseline Implementation

## DS340W Credit Card Fraud Detection Project

This folder contains the baseline implementation used for Phase 1 of the DS340W project.

The purpose of this phase is to reproduce and understand existing machine-learning techniques for credit card fraud detection before developing an improved ensemble learning approach.

## Original Code Source

**IBM - XGBoost + SMOTE Fraud Detection**

https://github.com/IBM/xgboost-smote-fraud

The original implementation explores machine-learning techniques related to credit card fraud detection, including:

- XGBoost
- Handling imbalanced datasets
- SMOTE and sampling techniques
- Model evaluation
- Credit card fraud detection

This implementation serves as a foundation for understanding fraud detection using machine learning.

## Phase 1: Baseline Implementation

The first phase focuses on understanding and reproducing the existing implementation.

The objectives include:

1. Execute the original machine-learning code.
2. Understand the dataset and preprocessing procedures.
3. Examine class imbalance in fraudulent transactions.
4. Explore XGBoost and sampling techniques.
5. Evaluate model performance.
6. Identify possible improvements for the next phase.

The original code was executed and verified before later modifications were made.

## Phase 2: Proposed Improvements

Building upon the baseline research, the next phase investigates an improved ensemble-learning approach.

The primary objective is to improve credit card fraud detection by combining:

- XGBoost
- Bayesian Optimization
- SMOTE + ENN
- Random Forest
- K-Nearest Neighbors (KNN)
- Soft Voting Ensemble

### Original Ensemble

The original comparison ensemble consists of:

- AdaBoost
- Random Forest
- KNN

### Improved Ensemble

The proposed improved ensemble consists of:

- XGBoost
- Random Forest
- KNN

XGBoost replaces AdaBoost to investigate whether a stronger boosting algorithm improves fraud detection performance.

Bayesian optimization is applied to XGBoost to identify effective hyperparameter configurations.

Both ensembles use SMOTE + ENN to address severe class imbalance.

**Note:** The original comparison ensemble is separate from the IBM reference implementation preserved in this folder.

## Addressing Class Imbalance

The credit card fraud dataset contains 284,807 transactions, including only 492 fraudulent transactions.

Because fraudulent transactions represent approximately 0.173% of the dataset, traditional machine-learning models may become biased toward normal transactions.

The improved project uses SMOTE + ENN:

- **SMOTE:** Generates synthetic minority-class observations.
- **ENN:** Removes selected observations based on nearest-neighbor disagreement.
- **SMOTE + ENN:** Combines oversampling and data cleaning to improve minority-class representation.

## Preventing Data Leakage

To ensure reliable evaluation, SMOTE + ENN will only be applied to training data.

The dataset will be divided into training and testing sets using stratified sampling.

During Bayesian optimization, preprocessing and resampling will occur separately within each training fold of cross-validation.

The testing dataset will remain unchanged.

This approach prevents information from the validation or testing datasets from influencing training through resampling.

## Model Evaluation

The original and improved ensembles will be compared using the same dataset and evaluation metrics.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC (Average Precision)
- Confusion Matrix
- False Positives
- False Negatives

Special attention will be given to recall, F1-score, PR-AUC, and false negatives because fraudulent transactions represent a very small portion of the dataset.

## Research Objectives

The improved research aims to:

1. Replace AdaBoost with XGBoost in the comparison ensemble.
2. Apply Bayesian optimization to tune XGBoost hyperparameters.
3. Maintain SMOTE + ENN to address class imbalance.
4. Compare both ensembles using identical dataset partitions.
5. Prevent data leakage through training-only resampling.
6. Determine whether the improved approach increases fraud detection performance.

## Project Files

- `1st_code.ipynb` - Original Phase 1 baseline implementation.

The improved machine-learning implementation is maintained separately in the `data/` folder:

- `01_dataset_split.py` - Google Colab execution script.
- `01_dataset_split_local.py` - Local Python execution script.
- `fraud_experiment.py` - Model training, Bayesian optimization, SMOTE + ENN, and ensemble comparison.

## Project Status

Phase 1 established the original implementation used as a reference for the research project.

Phase 2 extends the research through a comparison between the AdaBoost-based ensemble and the Bayesian-optimized XGBoost-based ensemble.

The improved approach will be evaluated using experimental results rather than assuming that it outperforms the original ensemble.
