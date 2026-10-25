
# Credit Card Fraud Detection Results

This folder contains the experimental results of the DS340W Credit Card Fraud Detection project.

## Model Comparison

Two ensemble models are evaluated:

- Baseline: AdaBoost + Random Forest + KNN
- Improved: Bayesian-optimized XGBoost + Random Forest + KNN

Both ensembles use SMOTE + ENN to address class imbalance.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- False Positives and False Negatives

## Experiment Outputs

After running the models locally, this folder may contain:

- `model_comparison.csv`
- `model_comparison.json`
- `optuna_trials.csv`
- `xgboost_best_parameters.json`
- `AdaBoost_RF_KNN_predictions.csv`
- `XGBoost_RF_KNN_predictions.csv`
- `charts/` — Performance graphs and confusion matrices

Results will be uploaded after the experiments have been completed and verified.
