# Credit Card Fraud Detection Using Machine Learning

## DS340W Project

This project investigates machine-learning techniques for detecting fraudulent credit card transactions. Because fraudulent transactions are extremely rare, the project focuses on improving fraud detection while controlling false positives and preventing data leakage.

The original approach compares machine-learning models including **AdaBoost, Random Forest, and K-Nearest Neighbors (KNN)** and combines them in a voting ensemble. The proposed improvement **replaces AdaBoost with XGBoost**, applies **Bayesian hyperparameter optimization**, and maintains **SMOTE + Edited Nearest Neighbors (ENN)** to address severe class imbalance. Both ensembles will be evaluated on the same dataset using identical splits and metrics.

## Parent Papers

The project draws on three research articles:

1. Tayebi, M., & El Kafhali, S. (2025). *A Novel Approach Based on XGBoost Classifier and Bayesian Optimization for Credit Card Fraud Detection.* https://doi.org/10.1016/j.csa.2025.100093
2. ScienceDirect research article: https://www.sciencedirect.com/science/article/pii/S2667344424000124
3. ScienceDirect research article: https://www.sciencedirect.com/science/article/pii/S2666827025000581

The first paper motivates the XGBoost and Bayesian optimization extension. The other two articles are included as additional parent-paper references; their full bibliographic details should be confirmed against the articles before final submission.

## Dataset

**Credit Card Fraud Detection Dataset:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains **284,807 transactions**, including **492 fraudulent transactions** (approximately **0.1727%**). The `Class` column is the target (`0` = legitimate, `1` = fraud). Features include `Time`, `Amount`, and anonymized components `V1`–`V28`.

Keep the downloaded `creditcard.csv` file locally in `data/` (or in the path expected by your notebook). Do not commit the large raw dataset unless permitted and necessary; document how to obtain it instead.

## Project Goals

1. Explore and preprocess the credit card transaction dataset.
2. Examine the severe class imbalance between fraud and normal transactions.
3. Establish baseline machine-learning models and an original voting ensemble using AdaBoost, Random Forest, and KNN.
4. Replace AdaBoost with XGBoost in the improved ensemble, retaining Random Forest and KNN.
5. Maintain SMOTE + ENN to improve minority-class representation and clean ambiguous examples.
6. Optimize XGBoost hyperparameters using Bayesian optimization on training data only.
7. Compare both ensembles using the same train/test split, cross-validation protocol, and evaluation metrics.
8. Prevent data leakage by applying scaling, SMOTE, and ENN only within training folds, never to validation or test data.
9. Compare experimental findings with the parent papers and discuss trade-offs, including false positives, false negatives, and computational cost.

## Proposed Methodology

### 1. Data splitting and leakage prevention

Use a fixed random seed and a **stratified train/test split** so the rare fraud class is represented in both partitions. Reserve the test set for final evaluation only. During cross-validation or Bayesian optimization, fit preprocessing and SMOTE + ENN **only on each training fold**. Never resample validation or test observations. If scaling is used for KNN, fit the scaler on training data only and apply the fitted transformation to held-out data.

### 2. Original ensemble (baseline)

- AdaBoost
- Random Forest
- K-Nearest Neighbors (KNN)
- Voting ensemble combining the three models

### 3. Improved ensemble (proposed)

- **XGBoost** (replaces AdaBoost)
- Random Forest
- K-Nearest Neighbors (KNN)
- Voting ensemble combining the three models

Use the same voting strategy for both ensembles (for example, soft voting when calibrated or otherwise comparable probability outputs are available). Keep shared model settings and evaluation procedures consistent to isolate the effect of the replacement and tuning.

### 4. Class imbalance: SMOTE + ENN

**SMOTE** creates synthetic minority-class training examples; **ENN** removes selected training examples whose labels disagree with nearby neighbors. Apply SMOTE + ENN inside an `imblearn` training pipeline, after the split and separately inside every cross-validation fold. Because the dataset is highly imbalanced, report performance on the **original, unresampled test distribution**.

### 5. Bayesian optimization

Use a Bayesian optimization library such as **Optuna** to tune XGBoost hyperparameters (for example, `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`, and regularization). Optimize an imbalance-sensitive cross-validation objective such as average precision (PR-AUC), using only training folds. Fix random seeds where supported and record the search space, trial count, and best parameters.

### 6. Evaluation and comparison

Evaluate both ensembles on the **same untouched test set** using:

- Accuracy
- Precision
- Recall (fraud class)
- F1-score (fraud class)
- ROC-AUC
- Average precision / PR-AUC
- Confusion matrix, especially false negatives and false positives
- Training and prediction time, where practical

Because non-fraud transactions dominate the dataset, **accuracy alone is insufficient**. Prioritize fraud recall, precision, F1, PR-AUC, and the operational trade-off between missed fraud and false alarms. Choose any decision threshold using training/validation data, not the test set.

## Project Structure

Current repository folders and files include:

```text
Credit-Card-Fraud-Detection/
├── baseline_code/                # Original implementation and baseline experiments
├── data/
│   ├── 01_dataset_split.ipynb
│   ├── 01_dataset_split_local.ipynb
│   └── README.md
├── references/                   # Research references
├── .gitignore
└── README.md
```

As the improved implementation is added, its notebooks/scripts, generated results, and supporting documentation will be linked here. The repository tree above describes the **current** structure rather than promising files that do not yet exist.

## Running the Project

1. Clone this repository.
2. Download `creditcard.csv` from the Kaggle dataset link above.
3. Open the existing dataset-splitting notebook in `data/` and update its dataset path as needed.
4. Run the baseline experiments and, once added, the improved XGBoost/Bayesian-optimization experiments using the same saved data partitions.
5. Record the evaluation metrics and confusion matrices for both ensembles.

Detailed package installation commands and commands for the improved pipeline will be added alongside the corresponding code files, so this README does not reference scripts that are not yet in the repository.

## Results

**Pending full experiments.** This README does not claim that XGBoost or Bayesian optimization has improved performance. Final results should include a side-by-side table of the original and improved ensembles on the identical held-out test set, with training-only sampling and tuning.

| Metric | Original ensemble | Improved ensemble |
|---|---|---|
| Accuracy | Pending | Pending |
| Precision (fraud) | Pending | Pending |
| Recall (fraud) | Pending | Pending |
| F1-score (fraud) | Pending | Pending |
| ROC-AUC | Pending | Pending |
| PR-AUC / average precision | Pending | Pending |
| False positives | Pending | Pending |
| False negatives | Pending | Pending |

## Status

**Project setup and research phase; improved implementation and experimental evaluation in progress.**
