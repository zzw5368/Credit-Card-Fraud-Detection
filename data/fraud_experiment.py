
"""
Credit Card Fraud Detection
DS340W Research Project

Research Objective:
Compare an original AdaBoost ensemble against an improved
Bayesian-optimized XGBoost ensemble.

Baseline:
    AdaBoost + Random Forest + KNN

Improved:
    XGBoost + Random Forest + KNN

Both models:
    - Use SMOTE + ENN
    - Use the same train/test split
    - Prevent data leakage
    - Use identical evaluation metrics

Author: DS340W Credit Card Fraud Detection Project
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd

from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline

from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    VotingClassifier
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# ==========================================================
# 1. GLOBAL SETTINGS
# ==========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
CLASSIFICATION_THRESHOLD = 0.50


# ==========================================================
# 2. LOAD DATASET
# ==========================================================

def load_data(data_path):

    print("\nLoading Credit Card Fraud Dataset...")

    df = pd.read_csv(data_path)

    if "Class" not in df.columns:
        raise ValueError(
            "Dataset must contain a 'Class' target column."
        )

    if df["Class"].isna().any():
        raise ValueError("Target column contains missing values.")

    if set(df["Class"].unique()) != {0, 1}:
        raise ValueError(
            "Target must contain binary labels 0 and 1."
        )

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    if X.isna().any().any():
        raise ValueError("Features contain missing values.")

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in X.dtypes
    ):
        raise ValueError("All features must be numeric.")

    print("Total transactions:", len(df))
    print("Normal transactions:", int((y == 0).sum()))
    print("Fraudulent transactions:", int((y == 1).sum()))
    print("Fraud percentage:", round(y.mean() * 100, 4), "%")

    return X, y


# ==========================================================
# 3. SMOTE + ENN
# ==========================================================

def create_sampler():

    return SMOTEENN(
        random_state=RANDOM_STATE
    )


# ==========================================================
# 4. BASELINE ENSEMBLE
# AdaBoost + Random Forest + KNN
# ==========================================================

def create_baseline_ensemble():

    adaboost = AdaBoostClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE
    )

    random_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    knn = KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    )

    ensemble = VotingClassifier(
        estimators=[
            ("adaboost", adaboost),
            ("random_forest", random_forest),
            ("knn", knn)
        ],
        voting="soft",
        n_jobs=1
    )

    return ensemble


# ==========================================================
# 5. IMPROVED ENSEMBLE
# XGBoost + Random Forest + KNN
# ==========================================================

def create_improved_ensemble(best_params):

    xgboost = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=2,
        **best_params
    )

    random_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    knn = KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    )

    ensemble = VotingClassifier(
        estimators=[
            ("xgboost", xgboost),
            ("random_forest", random_forest),
            ("knn", knn)
        ],
        voting="soft",
        n_jobs=1
    )

    return ensemble


# ==========================================================
# 6. CREATE LEAKAGE-SAFE PIPELINE
# ==========================================================

def create_pipeline(model):

    """
    StandardScaler is fitted on training data only.

    SMOTE + ENN is applied only during fitting.

    During cross-validation, each training fold receives
    its own scaling and resampling.

    Validation and test data are never resampled.
    """

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("smote_enn", create_sampler()),
            ("classifier", model)
        ]
    )

    return pipeline


# ==========================================================
# 7. BAYESIAN OPTIMIZATION
# ==========================================================

def optimize_xgboost(
    X_train,
    y_train,
    trials=20,
    folds=3
):

    print("\nStarting Bayesian Optimization...")

    cross_validation = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    def objective(trial):

        params = {

            "n_estimators": trial.suggest_int(
                "n_estimators",
                100,
                400,
                step=50
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                8
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.30,
                log=True
            ),

            "subsample": trial.suggest_float(
                "subsample",
                0.60,
                1.00
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.60,
                1.00
            ),

            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                1,
                10
            ),

            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                0.10,
                10.00,
                log=True
            )
        }

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=2,
            **params
        )

        pipeline = create_pipeline(model)

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cross_validation,
            scoring="average_precision",
            n_jobs=1,
            error_score="raise"
        )

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        )
    )

    study.optimize(
        objective,
        n_trials=trials
    )

    print("\nBayesian Optimization Completed")
    print("Best cross-validation PR-AUC:",
          study.best_value)
    print("Best parameters:", study.best_params)

    return (
        study.best_params,
        study.best_value,
        study.trials_dataframe()
    )


# ==========================================================
# 8. MODEL EVALUATION
# ==========================================================

def evaluate_model(model, X_test, y_test):

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    metrics = {

        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),

        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_test,
                probabilities
            )
        ),

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }

    return metrics, predictions, probabilities


# ==========================================================
# 9. RUN EXPERIMENT
# ==========================================================

def run(
    data_path="creditcard.csv",
    output_dir="results",
    trials=20,
    folds=3
):

    if trials < 1:
        raise ValueError("trials must be at least 1.")

    if folds < 2:
        raise ValueError("folds must be at least 2.")

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------
    # Step 1: Load dataset
    # ------------------------------------------------------

    X, y = load_data(data_path)

    # ------------------------------------------------------
    # Step 2: Stratified train/test split
    # ------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("\nTraining transactions:", len(X_train))
    print("Testing transactions:", len(X_test))
    print("Training fraud cases:", int(y_train.sum()))
    print("Testing fraud cases:", int(y_test.sum()))

    # ------------------------------------------------------
    # Step 3: Optimize XGBoost
    # ------------------------------------------------------

    best_params, best_cv_score, optuna_trials = (
        optimize_xgboost(
            X_train,
            y_train,
            trials=trials,
            folds=folds
        )
    )

    # Save optimization results

    optuna_trials.to_csv(
        output_path / "optuna_trials.csv",
        index=False
    )

    with open(
        output_path / "best_xgb_params.json",
        "w"
    ) as file:

        json.dump(
            {
                "best_params": best_params,
                "cv_pr_auc": best_cv_score
            },
            file,
            indent=4
        )

    # ------------------------------------------------------
    # Step 4: Define ensembles
    # ------------------------------------------------------

    models = {

        "Baseline_AdaBoost_RF_KNN":
            create_baseline_ensemble(),

        "Improved_XGBoost_RF_KNN":
            create_improved_ensemble(best_params)
    }

    results = {}

    predictions_df = pd.DataFrame(
        {
            "actual_class": y_test.to_numpy()
        },
        index=X_test.index
    )

    # ------------------------------------------------------
    # Step 5: Train both ensembles
    # ------------------------------------------------------

    for model_name, estimator in models.items():

        print("\n" + "=" * 55)
        print("Training:", model_name)
        print("=" * 55)

        pipeline = create_pipeline(estimator)

        # SMOTE + ENN is applied only to X_train.
        pipeline.fit(X_train, y_train)

        # X_test remains untouched.
        metrics, predictions, probabilities = evaluate_model(
            pipeline,
            X_test,
            y_test
        )

        results[model_name] = metrics

        predictions_df[
            model_name + "_prediction"
        ] = predictions

        predictions_df[
            model_name + "_probability"
        ] = probabilities

        # Save trained model

        joblib.dump(
            pipeline,
            output_path / f"{model_name}.joblib"
        )

        print("\nModel Performance:")

        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value}")

    # ------------------------------------------------------
    # Step 6: Compare results
    # ------------------------------------------------------

    comparison = pd.DataFrame.from_dict(
        results,
        orient="index"
    )

    comparison.index.name = "model"

    print("\n" + "=" * 55)
    print("FINAL MODEL COMPARISON")
    print("=" * 55)

    print(comparison.to_string())

    # ------------------------------------------------------
    # Step 7: Save results
    # ------------------------------------------------------

    comparison.to_csv(
        output_path / "model_comparison.csv"
    )

    predictions_df.index.name = "original_row_index"

    predictions_df.to_csv(
        output_path / "test_predictions.csv"
    )

    with open(
        output_path / "metrics.json",
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    experiment_config = {

        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "trials": trials,
        "folds": folds,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "optimization_metric": "average_precision",
        "sampling_method": "SMOTE + ENN",
        "leakage_prevention": "Training-only resampling",
        "training_rows": len(X_train),
        "testing_rows": len(X_test)
    }

    with open(
        output_path / "experiment_config.json",
        "w"
    ) as file:

        json.dump(
            experiment_config,
            file,
            indent=4
        )

    print("\nResults saved to:", output_path.resolve())

    return comparison


# ==========================================================
# 10. MAIN
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection Experiment"
    )

    parser.add_argument(
        "--data",
        default="creditcard.csv",
        help="Path to dataset"
    )

    parser.add_argument(
        "--output",
        default="results",
        help="Output directory"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of Bayesian optimization trials"
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Number of cross-validation folds"
    )

    args = parser.parse_args()

    run(
        data_path=args.data,
        output_dir=args.output,
        trials=args.trials,
        folds=args.folds
    )
