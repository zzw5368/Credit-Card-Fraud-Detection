# Parent Papers

This document lists the three research papers used as references for the DS340W Credit Card Fraud Detection project. The project investigates whether replacing AdaBoost with XGBoost, optimizing XGBoost hyperparameters using Bayesian optimization, and retaining SMOTE + ENN improves fraud detection relative to the original ensemble.

## Parent Paper 1 — XGBoost and Bayesian Optimization

**Citation:** Tayebi, M., & El Kahfali, S. (2025). *A novel approach based on XGBoost classifier and Bayesian optimization for credit card fraud detection.* Cyber Security and Applications, 3, 100093.

**DOI:** https://doi.org/10.1016/j.csa.2025.100093

**Official publication:** https://www.sciencedirect.com/science/article/pii/S2772918425000104

### Techniques Relevant to This Project
- XGBoost classification
- Bayesian hyperparameter optimization
- Credit card fraud detection
- Classification model evaluation

### Purpose
This paper provides the main methodological inspiration for using an XGBoost classifier and Bayesian optimization. In our project, XGBoost will replace AdaBoost in the improved voting ensemble, and Bayesian optimization will be used to select XGBoost hyperparameters.

## Parent Paper 2 — Additional Research Reference

**Official publication:** https://www.sciencedirect.com/science/article/pii/S2667344424000124

**Citation details:** To be completed after verifying the paper's title, authors, year, and DOI from the publication itself.

### Purpose
This is one of the three parent papers selected for the project. Its specific methods and findings will be documented after reviewing the full article, rather than assuming they support a technique that has not yet been verified.

## Parent Paper 3 — Additional Research Reference

**Official publication:** https://www.sciencedirect.com/science/article/pii/S2666827025000581

**Citation details:** To be completed after verifying the paper's title, authors, year, and DOI from the publication itself.

### Purpose
This is the third selected parent paper. Its methodological contribution and relationship to the project will be described after confirming the article's contents.

## Proposed Research Extension

The project will compare two voting ensembles using the same credit card fraud dataset and evaluation procedure:

| Component | Original ensemble | Improved ensemble |
| --- | --- | --- |
| Boosting model | AdaBoost | XGBoost |
| Other classifiers | Random Forest + KNN | Random Forest + KNN |
| Imbalance handling | SMOTE + ENN | SMOTE + ENN |
| Hyperparameter search | Baseline configuration | Bayesian optimization for XGBoost |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix | Same metrics |

To prevent data leakage, resampling must occur **only on training data** (and within training folds during model selection). The held-out test set must retain its original class distribution. Final performance claims will be added only after running the experiments.
