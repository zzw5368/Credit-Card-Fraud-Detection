# Dataset

## Credit Card Fraud Detection Dataset

**Source:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

This is one of the datasets used in the parent paper for this project. The dataset is publicly accessible, allowing us to reproduce and extend the experiments from the parent papers.

The dataset contains transactions made by European cardholders.

The dataset is highly imbalanced and contains:
- Total transactions: 284,807
- Normal transactions: 284,315
- Fraudulent transactions: 492
- Fraud rate: Approximately 0.173%

## Main Variables

- **Time:** Seconds elapsed between each transaction and the first transaction.
- **V1–V28:** Numerical features obtained through Principal Component Analysis (PCA).
- **Amount:** Transaction amount.
- **Class:** Target variable identifying fraudulent transactions.

**Class labels:**
- 0 = Normal transaction
- 1 = Fraudulent transaction

## Use in This Project

This dataset will be used to train and evaluate machine-learning models for credit card fraud detection.

The primary objective is to improve fraud detection performance by combining XGBoost, Bayesian Optimization, and SMOTE + ENN.

The project will investigate the following techniques:

- AdaBoost
- XGBoost
- Random Forest
- K-Nearest Neighbors (KNN)
- Bayesian Optimization
- SMOTE + ENN
- Soft Voting Ensemble
- Model Performance Comparison

## Data Preprocessing

Before training the models, the dataset will undergo preprocessing.

1. Load the credit card transaction dataset.
2. Check for missing values and duplicate records.
3. Separate features from the target variable.
4. Divide the dataset into training and testing sets using stratified sampling.
5. Apply appropriate feature scaling for models that require it.
6. Address class imbalance using SMOTE + ENN.
7. Train and evaluate the machine-learning models.

### Preventing Data Leakage

To ensure reliable model evaluation, SMOTE + ENN will only be applied to the training data.

The testing dataset will remain unchanged to represent real-world credit card transactions.

During Bayesian optimization, resampling and preprocessing will be performed within each training fold of cross-validation.

This prevents synthetic or resampled observations from leaking into validation or testing data.

## Handling Class Imbalance

Because fraudulent transactions represent only approximately 0.173% of the dataset, class imbalance is a major challenge.

The project will use **SMOTE + ENN** to address this issue.

### SMOTE (Synthetic Minority Oversampling Technique)

SMOTE generates synthetic fraudulent transactions based on existing minority-class observations.

This helps the machine-learning models learn patterns associated with fraudulent transactions.

### ENN (Edited Nearest Neighbors)

ENN removes selected observations whose class labels disagree with their nearest neighbors.

This helps clean overlapping or noisy regions of the training dataset.

### SMOTE + ENN

Combining SMOTE and ENN allows the project to increase minority-class representation while cleaning the resampled dataset.

The goal is to improve fraud detection without excessively increasing false positives.

## Machine Learning Models

### Original Ensemble

The original ensemble combines:

- AdaBoost
- Random Forest
- K-Nearest Neighbors (KNN)

These models are combined using a voting ensemble.

### Improved Ensemble

The proposed improved ensemble combines:

- XGBoost
- Random Forest
- K-Nearest Neighbors (KNN)

XGBoost replaces AdaBoost to explore a potentially stronger boosting alternative.

Bayesian optimization will be applied to XGBoost to identify better hyperparameter configurations.

Both ensembles will use SMOTE + ENN and the same dataset partitions.

## Bayesian Optimization

Bayesian optimization will be used to tune XGBoost hyperparameters.

Hyperparameters considered include:

- Number of estimators
- Maximum tree depth
- Learning rate
- Subsample ratio
- Column sampling ratio
- Minimum child weight
- Gamma

The optimization process will use cross-validation on the training data.

The objective is to improve model performance while reducing the need for manual hyperparameter tuning.

## Model Evaluation

The original and improved ensembles will be evaluated using the same dataset and evaluation metrics.

The evaluation metrics include:

- **Accuracy:** Overall percentage of correctly classified transactions.
- **Precision:** Percentage of predicted fraudulent transactions that are actually fraudulent.
- **Recall:** Percentage of actual fraudulent transactions correctly detected.
- **F1-Score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well the model separates fraud and normal transactions across different thresholds.
- **PR-AUC:** Evaluates precision-recall performance, especially important for imbalanced datasets.
- **Confusion Matrix:** Shows true positives, true negatives, false positives, and false negatives.

Special attention will be given to recall, F1-score, PR-AUC, and false negatives because missing fraudulent transactions can be costly.

## Model Comparison

| Feature | Original Ensemble | Improved Ensemble |
|---------|-------------------|-------------------|
| Boosting Model | AdaBoost | XGBoost |
| Random Forest | Yes | Yes |
| KNN | Yes | Yes |
| Voting Ensemble | Yes | Yes |
| Sampling Method | SMOTE + ENN | SMOTE + ENN |
| Bayesian Optimization | No | Yes, for XGBoost |
| Dataset | Credit Card Fraud Dataset | Same Dataset |
| Evaluation Metrics | Same Metrics | Same Metrics |
| Data Leakage Prevention | Training-Only Sampling | Training-Only Sampling |

## Project Files

- `01_dataset_split.ipynb` - Google Colab notebook for dataset preprocessing and model experiments.
- `01_dataset_split_local.ipynb` - Local Jupyter Notebook version.
- `fraud_experiment.py` - Shared Python implementation for training, optimization, and ensemble comparison.
- `README.md` - Dataset description and methodology documentation.

## Expected Outcomes

The improved approach aims to:

1. Improve fraudulent transaction detection.
2. Reduce the number of false negatives.
3. Improve recall, F1-score, and PR-AUC.
4. Evaluate whether XGBoost outperforms AdaBoost.
5. Determine whether Bayesian optimization improves ensemble performance.
6. Maintain reliable evaluation by preventing data leakage.

The improved model is expected to provide a stronger alternative to the original ensemble, but its actual performance will be determined through experimental results.

## Project Status

The improved methodology has been defined.

Full experimental results and final performance comparisons will be documented after the models have been trained and evaluated on the dataset.
