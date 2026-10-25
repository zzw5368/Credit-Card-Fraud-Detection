"""DS340W Credit Card Fraud Detection — IBM-inspired Phase 1 extension.

Baseline: AdaBoost + Random Forest + KNN (soft voting)
Improved: Bayesian-tuned XGBoost + Random Forest + KNN (soft voting)
Both: StandardScaler -> SMOTEENN -> VotingClassifier, fitted on training only.

Retains the uploaded IBM-adapted notebook's 70/20/10 stratified split and
20%-of-train / 20%-of-test low-compute development mode. The separate 10%
final validation set is saved but never read for model selection/evaluation.

Install:
    pip install numpy pandas scikit-learn imbalanced-learn xgboost optuna joblib
Run locally:
    python 1st_code.py --data creditcard.csv --output results --trials 20
Run with full train/test (more demanding):
    python 1st_code.py --data creditcard.csv --output full_results --full-data
In Colab, mount Drive separately, then pass --data /content/drive/MyDrive/DS340W/creditcard.csv.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
THRESHOLD = 0.50


def load_data(path):
    df = pd.read_csv(path)
    needed = set(FEATURES + ["Class"])
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df[FEATURES + ["Class"]].copy()
    if df.isnull().any().any():
        raise ValueError("Missing values found. Investigate before training.")
    if set(df["Class"].unique()) != {0, 1}:
        raise ValueError("Class must contain both 0 (normal) and 1 (fraud).")
    print(f"Rows: {len(df):,}; fraud: {df['Class'].sum():,} "
          f"({df['Class'].mean():.3%})")
    return df


def make_splits(df, output, full_data):
    # First separate the final 10% validation set; it remains unseen.
    development, validation = train_test_split(
        df, test_size=0.10, stratify=df["Class"], random_state=SEED
    )
    # 2/9 of the remaining 90% is exactly 20% of the original data.
    training, test = train_test_split(
        development, test_size=2 / 9, stratify=development["Class"],
        random_state=SEED,
    )
    # Preserve the original notebook's saved full split files.
    training.to_csv(output / "training_data.csv", index=False)
    test.to_csv(output / "test_data.csv", index=False)
    validation.to_csv(output / "validation_data.csv", index=False)
    print("Full split (train/test/held-out validation):",
          len(training), len(test), len(validation))

    if full_data:
        used_train, used_test = training, test
    else:
        # Reproduce Phase 1's low-compute subsets, from train and test only.
        used_train, _ = train_test_split(
            training, train_size=0.20, stratify=training["Class"],
            random_state=SEED,
        )
        used_test, _ = train_test_split(
            test, train_size=0.20, stratify=test["Class"],
            random_state=SEED,
        )
    print("Evaluation split (train/test):", len(used_train), len(used_test))
    print("Fraud counts (train/test):", int(used_train["Class"].sum()),
          int(used_test["Class"].sum()))
    return (used_train[FEATURES], used_train["Class"],
            used_test[FEATURES], used_test["Class"])


def pipeline_for(classifier):
    # imblearn Pipeline ensures samplers operate only during .fit().
    # CV re-fits scaler and SMOTEENN in each training fold; never on test.
    return Pipeline([
        ("scale", StandardScaler()),
        ("sample", SMOTEENN(
            random_state=SEED,
            smote=SMOTE(random_state=SEED, k_neighbors=3),
        )),
        ("model", classifier),
    ])


def rf():
    # The original notebook used a 50-tree, max-depth-10 RF model.
    return RandomForestClassifier(
        n_estimators=50, max_depth=10, random_state=SEED, n_jobs=2
    )


def knn():
    return KNeighborsClassifier(n_neighbors=5, n_jobs=2)


def baseline_ensemble():
    return VotingClassifier(
        estimators=[
            ("ada", AdaBoostClassifier(n_estimators=100, random_state=SEED)),
            ("rf", rf()),
            ("knn", knn()),
        ],
        voting="soft",
        n_jobs=1,
    )


def make_xgb(params):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=2,
        **params,
    )


def improved_ensemble(best_params):
    return VotingClassifier(
        estimators=[
            ("xgb", make_xgb(best_params)),
            ("rf", rf()),
            ("knn", knn()),
        ],
        voting="soft",
        n_jobs=1,
    )


def tune_xgboost(X_train, y_train, trials, folds, output):
    """Optuna TPE Bayesian/sequential optimization of standalone XGBoost AP.

    XGBoost is optimized separately, then inserted into improved soft voting.
    The test and final validation data play no role in the search.
    """
    if int((y_train == 1).sum()) < folds * 4:
        raise ValueError("Not enough fraud cases for SMOTE and CV folds.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 75, 250, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.25, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.65, 1.0
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.1, 10.0, log=True
            ),
        }
        scores = cross_val_score(
            pipeline_for(make_xgb(params)), X_train, y_train,
            cv=cv, scoring="average_precision", n_jobs=1,
            error_score="raise",
        )
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=trials)
    study.trials_dataframe().to_csv(output / "optuna_trials.csv", index=False)
    (output / "xgboost_best_parameters.json").write_text(
        json.dumps({"best_parameters": study.best_params,
                    "cross_validation_average_precision": study.best_value}, indent=2)
    )
    print("Best XGBoost CV average precision:", round(study.best_value, 5))
    print("XGBoost hyperparameters:", study.best_params)
    return study.best_params


def evaluate(name, pipeline, X_test, y_test, output):
    # Average precision and ROC-AUC MUST use probabilities, not hard labels.
    probs = pipeline.predict_proba(X_test)[:, 1]
    pred = (probs >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    scores = {
        "Model": name,
        "Accuracy": float(accuracy_score(y_test, pred)),
        "Precision": float(precision_score(y_test, pred, zero_division=0)),
        "Recall": float(recall_score(y_test, pred, zero_division=0)),
        "F1": float(f1_score(y_test, pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_test, probs)),
        "PR_AUC_AP": float(average_precision_score(y_test, probs)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    print(f"\n{name} confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, pred, labels=[0, 1]))
    print(classification_report(y_test, pred, zero_division=0))
    pd.DataFrame({
        "dataset_index": X_test.index,
        "actual": y_test.to_numpy(),
        "predicted": pred,
        "fraud_probability": probs,
    }).to_csv(output / f"{name}_predictions.csv", index=False)
    joblib.dump(pipeline, output / f"{name}.joblib")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="creditcard.csv")
    parser.add_argument("--output", default="results")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--full-data", action="store_true",
                        help="Use the full training and test sets rather than the reduced development subsets")
    args = parser.parse_args()
    if args.trials < 1 or args.folds < 2:
        parser.error("--trials must be >=1 and --folds must be >=2")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data)
    X_train, y_train, X_test, y_test = make_splits(
        df, output, args.full_data
    )
    best = tune_xgboost(
        X_train, y_train, args.trials, args.folds, output
    )
    experiments = {
        "AdaBoost_RF_KNN": baseline_ensemble(),
        "XGBoost_RF_KNN": improved_ensemble(best),
    }
    rows = []
    for name, classifier in experiments.items():
        print(f"\nTraining {name} with SMOTE + ENN...")
        model = pipeline_for(classifier)
        model.fit(X_train, y_train)
        rows.append(evaluate(name, model, X_test, y_test, output))

    comparison = pd.DataFrame(rows).set_index("Model")
    comparison.to_csv(output / "model_comparison.csv")
    (output / "model_comparison.json").write_text(
        comparison.to_json(orient="index", indent=2)
    )
    (output / "run_config.json").write_text(json.dumps({
        "random_seed": SEED,
        "train_fraction_of_original": 0.70,
        "test_fraction_of_original": 0.20,
        "held_out_validation_fraction": 0.10,
        "low_compute_subsets": not args.full_data,
        "development_train_rows": len(X_train),
        "development_test_rows": len(X_test),
        "bayesian_trials": args.trials,
        "cross_validation_folds": args.folds,
        "threshold": THRESHOLD,
        "bayesian_objective": "XGBoost cross-validated average precision",
        "sampling": "SMOTEENN applied to training folds and training data only",
        "final_validation_used": False,
    }, indent=2))
    print("\n=== Model comparison (development test only) ===")
    print(comparison.round(5).to_string())
    print("\nValidation data was saved and never used for model selection.")
    print("Results written to:", output.resolve())


if __name__ == "__main__":
    main()
