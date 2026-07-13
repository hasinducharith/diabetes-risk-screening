import os
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FEATURES: List[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"
INVALID_ZERO_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"


def ensure_dirs() -> None:
    os.makedirs("report_assets/figures", exist_ok=True)


def load_dataset(path: str = "dataset.csv") -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(DATASET_URL, header=None)
        df.columns = FEATURES + [TARGET]
        df.to_csv(path, index=False)

    if TARGET not in df.columns:
        # Handles no-header CSV case if user replaced dataset manually.
        df.columns = FEATURES + [TARGET]
    return df


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df.drop_duplicates(inplace=True)
    for col in INVALID_ZERO_COLUMNS:
        df[col] = df[col].replace(0, np.nan)

    x = df[FEATURES]
    y = df[TARGET].astype(int)
    return x, y


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(transformers=[("num", numeric_pipe, FEATURES)])


def build_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1200,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    GradientBoostingClassifier(
                        random_state=RANDOM_STATE,
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=3,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def choose_best(metrics_df: pd.DataFrame) -> str:
    ranked = metrics_df.sort_values(by=["recall", "f1", "roc_auc"], ascending=False)
    return ranked.iloc[0]["model"]


def save_artifacts(
    metrics_df: pd.DataFrame,
    best_name: str,
    best_model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model_probs: Dict[str, np.ndarray],
) -> None:
    metrics_df.to_csv("model_metrics.csv", index=False)

    with open("model.pkl", "wb") as f:
        pickle.dump({"model": best_model, "features": FEATURES, "model_name": best_name}, f)

    # Final model confusion matrix
    ConfusionMatrixDisplay.from_estimator(best_model, x_test, y_test)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig("report_assets/figures/confusion_matrix.png", dpi=220)
    plt.close()

    # ROC curves for all models
    plt.figure(figsize=(8, 6))
    for name, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("report_assets/figures/roc_comparison.png", dpi=220)
    plt.close()

    with open("debug_notes.md", "w", encoding="utf-8") as f:
        f.write("# Debugging and Iteration Notes\n\n")
        f.write("1. Replaced medically invalid zeros with missing values and imputed medians.\n")
        f.write("2. Added class balancing for Logistic Regression and Random Forest to improve recall.\n")
        f.write("3. Tuned tree depth and leaf size to reduce overfitting in Random Forest.\n")


def main() -> None:
    ensure_dirs()
    df = load_dataset()
    x, y = preprocess(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()
    models = build_models(preprocessor)

    rows = []
    fitted_models: Dict[str, Pipeline] = {}
    model_probs: Dict[str, np.ndarray] = {}

    for name, pipe in models.items():
        pipe.fit(x_train, y_train)
        fitted_models[name] = pipe

        metrics = evaluate_model(pipe, x_test, y_test)
        metrics["model"] = name
        rows.append(metrics)

        model_probs[name] = pipe.predict_proba(x_test)[:, 1]

    metrics_df = pd.DataFrame(rows)[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]]
    best_name = choose_best(metrics_df)
    best_model = fitted_models[best_name]

    save_artifacts(metrics_df, best_name, best_model, x_test, y_test, model_probs)

    print("Training complete.")
    print("Best model:", best_name)
    print("\nModel metrics:")
    print(metrics_df.sort_values(by="recall", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
