from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree,
)


# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).parent

TITANIC_FILE = BASE_DIR / "titanic.csv"

CHARTS_DIR = BASE_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)


SPLIT_SUMMARY_FILE = (
    BASE_DIR / "task7_split_summary.csv"
)

PREPROCESSING_SUMMARY_FILE = (
    BASE_DIR / "task8_preprocessing_summary.csv"
)

MODEL_COMPARISON_FILE = (
    BASE_DIR / "task10_model_comparison.csv"
)

IMBALANCE_COMPARISON_FILE = (
    BASE_DIR / "task11_imbalance_comparison.csv"
)

IMBALANCE_CONCLUSION_FILE = (
    BASE_DIR / "task11_imbalance_conclusion.txt"
)

TASK12_RESULTS_FILE = (
    BASE_DIR / "task12_random_forest_tuning.csv"
)

REGRESSION_RESULTS_FILE = (
    BASE_DIR / "task13_regression_metrics.csv"
)

REGRESSION_CONCLUSION_FILE = (
    BASE_DIR / "task13_regression_conclusion.txt"
)

FINAL_COMPARISON_FILE = (
    BASE_DIR / "task14_final_model_comparison.csv"
)

FINAL_RECOMMENDATION_FILE = (
    BASE_DIR / "task14_final_recommendation.txt"
)

BEST_PIPELINE_FILE = (
    BASE_DIR / "best_titanic_pipeline.joblib"
)

TASK15_PREDICTION_FILE = (
    BASE_DIR / "task15_reload_prediction.csv"
)


DECISION_TREE_FILE = (
    CHARTS_DIR / "decision_tree.png"
)

LOGISTIC_CM_FILE = (
    CHARTS_DIR / "logistic_confusion_matrix.png"
)

TREE_CM_FILE = (
    CHARTS_DIR / "decision_tree_confusion_matrix.png"
)

FOREST_CM_FILE = (
    CHARTS_DIR / "random_forest_confusion_matrix.png"
)

ROC_CURVE_FILE = (
    CHARTS_DIR / "classifier_roc_curves.png"
)

IMBALANCE_CHART_FILE = (
    CHARTS_DIR / "imbalance_comparison.png"
)

REGRESSION_RESIDUAL_FILE = (
    CHARTS_DIR / "regression_residual_plot.png"
)


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20


# =========================================================
# CLASSIFICATION FEATURES
# =========================================================

FEATURE_COLUMNS = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
]

TARGET_COLUMN = "survived"


NUMERIC_FEATURES = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]


CATEGORICAL_FEATURES = [
    "sex",
    "embarked",
]


# =========================================================
# REGRESSION FEATURES
# =========================================================

REGRESSION_FEATURES = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked",
]

REGRESSION_TARGET = "fare"


REGRESSION_NUMERIC_FEATURES = [
    "pclass",
    "age",
    "sibsp",
    "parch",
]


REGRESSION_CATEGORICAL_FEATURES = [
    "sex",
    "embarked",
]


# =========================================================
# CLASSIFICATION PREPROCESSOR
# =========================================================

def create_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


# =========================================================
# REGRESSION PREPROCESSOR
# =========================================================

def create_regression_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                REGRESSION_NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                REGRESSION_CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


# =========================================================
# CLASSIFIER EVALUATION
# =========================================================

def evaluate_model(
    model_name,
    model,
    X_test,
    y_test,
):

    predictions = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    result = {
        "model": model_name,

        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities,
    )

    return (
        result,
        matrix,
        fpr,
        tpr,
    )


# =========================================================
# IMBALANCE MODEL EVALUATION
# =========================================================

def evaluate_imbalance_model(
    strategy,
    model,
    X_test,
    y_test,
):

    predictions = model.predict(
        X_test
    )

    return {
        "strategy": strategy,

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
    }


# =========================================================
# CONFUSION MATRIX
# =========================================================

def save_confusion_matrix(
    matrix,
    model_name,
    output_file,
):

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Not Survived",
            "Survived",
        ],
    )

    display.plot(
        values_format="d"
    )

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        output_file
    )

    plt.close()


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n========================================")
    print("MODULE 2 - ANALYTICS PIPELINE")
    print("PART B - TASKS 7 TO 15")
    print("========================================")

    # =====================================================
    # LOAD TITANIC CSV
    # =====================================================

    if not TITANIC_FILE.exists():

        raise FileNotFoundError(
            "titanic.csv not found. "
            "Run analytics/01_eda.py first."
        )

    df = pd.read_csv(
        TITANIC_FILE
    )

    print(
        "\nExisting titanic.csv loaded successfully."
    )

    # =====================================================
    # TASK 7 - STRATIFIED TRAIN/TEST SPLIT
    # =====================================================

    print("\n========================================")
    print("TASK 7 - STRATIFIED TRAIN/TEST SPLIT")
    print("========================================")

    X = df[
        FEATURE_COLUMNS
    ].copy()

    y = df[
        TARGET_COLUMN
    ].copy()

    class_counts = (
        y
        .value_counts()
        .sort_index()
    )

    class_percentages = (
        y
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    not_survived_percent = (
        class_percentages.get(
            0,
            0
        )
    )

    survived_percent = (
        class_percentages.get(
            1,
            0
        )
    )

    print(
        f"\nNot survived (0): "
        f"{class_counts.get(0, 0)} "
        f"({not_survived_percent:.2f}%)"
    )

    print(
        f"Survived (1): "
        f"{class_counts.get(1, 0)} "
        f"({survived_percent:.2f}%)"
    )

    print(
        "\nA stratified split is used because "
        "the classes are not perfectly balanced."
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    train_percentages = (
        y_train
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    test_percentages = (
        y_test
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    split_summary = pd.DataFrame(
        {
            "dataset": [
                "Full dataset",
                "Training set",
                "Test set",
            ],
            "rows": [
                len(y),
                len(y_train),
                len(y_test),
            ],
            "not_survived_percent": [
                not_survived_percent,
                train_percentages.get(0, 0),
                test_percentages.get(0, 0),
            ],
            "survived_percent": [
                survived_percent,
                train_percentages.get(1, 0),
                test_percentages.get(1, 0),
            ],
        }
    )

    split_summary.to_csv(
        SPLIT_SUMMARY_FILE,
        index=False,
    )

    print("\n========================================")
    print("TASK 7 COMPLETED SUCCESSFULLY")
    print("========================================")

    # =====================================================
    # TASK 8 - TRAIN-ONLY PREPROCESSING
    # =====================================================

    print("\n========================================")
    print("TASK 8 - TRAIN-ONLY PREPROCESSING")
    print("========================================")

    demo_preprocessor = (
        create_preprocessor()
    )

    X_train_processed = (
        demo_preprocessor
        .fit_transform(
            X_train
        )
    )

    X_test_processed = (
        demo_preprocessor
        .transform(
            X_test
        )
    )

    feature_names = (
        demo_preprocessor
        .get_feature_names_out()
    )

    X_train_processed_df = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
    )

    X_test_processed_df = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
    )

    if (
        X_train_processed_df
        .isnull()
        .sum()
        .sum()
        != 0
    ):

        raise ValueError(
            "Task 8 failed."
        )

    if (
        X_test_processed_df
        .isnull()
        .sum()
        .sum()
        != 0
    ):

        raise ValueError(
            "Task 8 failed."
        )

    preprocessing_summary = pd.DataFrame(
        {
            "feature": (
                NUMERIC_FEATURES
                +
                CATEGORICAL_FEATURES
            ),

            "type": (
                ["numeric"]
                * len(NUMERIC_FEATURES)
                +
                ["categorical"]
                * len(CATEGORICAL_FEATURES)
            ),

            "missing_strategy": (
                ["median"]
                * len(NUMERIC_FEATURES)
                +
                ["most_frequent"]
                * len(CATEGORICAL_FEATURES)
            ),

            "processing": (
                ["StandardScaler"]
                * len(NUMERIC_FEATURES)
                +
                ["OneHotEncoder"]
                * len(CATEGORICAL_FEATURES)
            ),
        }
    )

    preprocessing_summary.to_csv(
        PREPROCESSING_SUMMARY_FILE,
        index=False,
    )

    print("\n========================================")
    print("TASK 8 COMPLETED SUCCESSFULLY")
    print("========================================")

    # =====================================================
    # TASK 9 - THREE CLASSIFIERS
    # =====================================================

    print("\n========================================")
    print("TASK 9 - THREE CLASSIFIERS")
    print("========================================")

    logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    decision_tree_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=4,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    logistic_pipeline.fit(
        X_train,
        y_train,
    )

    decision_tree_pipeline.fit(
        X_train,
        y_train,
    )

    random_forest_pipeline.fit(
        X_train,
        y_train,
    )

    tree_feature_names = (
        decision_tree_pipeline
        .named_steps[
            "preprocessor"
        ]
        .get_feature_names_out()
    )

    fitted_tree = (
        decision_tree_pipeline
        .named_steps[
            "classifier"
        ]
    )

    plt.figure(
        figsize=(24, 12)
    )

    plot_tree(
        fitted_tree,
        feature_names=tree_feature_names,
        class_names=[
            "Not Survived",
            "Survived",
        ],
        filled=True,
        rounded=True,
        fontsize=8,
    )

    plt.title(
        "Titanic Decision Tree Classifier"
    )

    plt.tight_layout()

    plt.savefig(
        DECISION_TREE_FILE,
        bbox_inches="tight",
    )

    plt.close()

    print("\n========================================")
    print("TASK 9 COMPLETED SUCCESSFULLY")
    print("========================================")

    # =====================================================
    # TASK 10 - CLASSIFIER EVALUATION
    # =====================================================

    print("\n========================================")
    print("TASK 10 - CLASSIFIER EVALUATION")
    print("========================================")

    (
        logistic_result,
        logistic_matrix,
        logistic_fpr,
        logistic_tpr,
    ) = evaluate_model(
        "Logistic Regression",
        logistic_pipeline,
        X_test,
        y_test,
    )

    (
        tree_result,
        tree_matrix,
        tree_fpr,
        tree_tpr,
    ) = evaluate_model(
        "Decision Tree",
        decision_tree_pipeline,
        X_test,
        y_test,
    )

    (
        forest_result,
        forest_matrix,
        forest_fpr,
        forest_tpr,
    ) = evaluate_model(
        "Random Forest",
        random_forest_pipeline,
        X_test,
        y_test,
    )

    save_confusion_matrix(
        logistic_matrix,
        "Logistic Regression",
        LOGISTIC_CM_FILE,
    )

    save_confusion_matrix(
        tree_matrix,
        "Decision Tree",
        TREE_CM_FILE,
    )

    save_confusion_matrix(
        forest_matrix,
        "Random Forest",
        FOREST_CM_FILE,
    )

    model_comparison = pd.DataFrame(
        [
            logistic_result,
            tree_result,
            forest_result,
        ]
    )

    model_comparison.to_csv(
        MODEL_COMPARISON_FILE,
        index=False,
    )

    print(
        model_comparison
        .round(4)
        .to_string(
            index=False
        )
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        logistic_fpr,
        logistic_tpr,
        label=(
            "Logistic Regression "
            f"(AUC={logistic_result['auc']:.3f})"
        ),
    )

    plt.plot(
        tree_fpr,
        tree_tpr,
        label=(
            "Decision Tree "
            f"(AUC={tree_result['auc']:.3f})"
        ),
    )

    plt.plot(
        forest_fpr,
        forest_tpr,
        label=(
            "Random Forest "
            f"(AUC={forest_result['auc']:.3f})"
        ),
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curves - Titanic Classifiers"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ROC_CURVE_FILE
    )

    plt.close()

    print("\n========================================")
    print("TASK 10 COMPLETED SUCCESSFULLY")
    print("========================================")

    # =====================================================
    # TASK 11 - IMBALANCE HANDLING
    # =====================================================

    print("\n========================================")
    print("TASK 11 - IMBALANCE HANDLING")
    print("========================================")

    baseline_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    balanced_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    smote_pipeline = ImbPipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    baseline_pipeline.fit(
        X_train,
        y_train,
    )

    balanced_pipeline.fit(
        X_train,
        y_train,
    )

    smote_pipeline.fit(
        X_train,
        y_train,
    )

    baseline_result = (
        evaluate_imbalance_model(
            "Baseline",
            baseline_pipeline,
            X_test,
            y_test,
        )
    )

    balanced_result = (
        evaluate_imbalance_model(
            "Class Weight Balanced",
            balanced_pipeline,
            X_test,
            y_test,
        )
    )

    smote_result = (
        evaluate_imbalance_model(
            "SMOTE",
            smote_pipeline,
            X_test,
            y_test,
        )
    )

    imbalance_comparison = pd.DataFrame(
        [
            baseline_result,
            balanced_result,
            smote_result,
        ]
    )

    imbalance_comparison.to_csv(
        IMBALANCE_COMPARISON_FILE,
        index=False,
    )

    best_imbalance = (
        imbalance_comparison
        .sort_values(
            by="f1",
            ascending=False,
        )
        .iloc[0]
    )

    imbalance_conclusion = (
        f"The Titanic target contains "
        f"{not_survived_percent:.2f}% non-survivors and "
        f"{survived_percent:.2f}% survivors. "
        f"{best_imbalance['strategy']} produced the highest "
        f"F1 score of {best_imbalance['f1']:.4f}, "
        f"with precision {best_imbalance['precision']:.4f} "
        f"and recall {best_imbalance['recall']:.4f}. "
        "SMOTE was applied only to training data."
    )

    with open(
        IMBALANCE_CONCLUSION_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            imbalance_conclusion
        )

    print("\n========================================")
    print("TASK 11 COMPLETED SUCCESSFULLY")
    print("========================================")

    # =====================================================
    # TASK 12 - GRID SEARCH + OOB
    # =====================================================

    print("\n========================================")
    print("TASK 12 - RANDOM FOREST GRIDSEARCHCV")
    print("========================================")

    tuned_rf_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    oob_score=True,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [
            100,
            200,
        ],

        "classifier__max_depth": [
            None,
            5,
            10,
        ],

        "classifier__max_features": [
            "sqrt",
            "log2",
        ],
    }

    grid_search = GridSearchCV(
        estimator=tuned_rf_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    best_params = (
        grid_search.best_params_
    )

    best_rf_pipeline = (
        grid_search.best_estimator_
    )

    best_rf_classifier = (
        best_rf_pipeline
        .named_steps[
            "classifier"
        ]
    )

    best_oob_score = (
        best_rf_classifier.oob_score_
    )

    best_rf_predictions = (
        best_rf_pipeline.predict(
            X_test
        )
    )

    best_rf_probabilities = (
        best_rf_pipeline
        .predict_proba(
            X_test
        )[:, 1]
    )

    tuned_accuracy = accuracy_score(
        y_test,
        best_rf_predictions,
    )

    tuned_precision = precision_score(
        y_test,
        best_rf_predictions,
        zero_division=0,
    )

    tuned_recall = recall_score(
        y_test,
        best_rf_predictions,
        zero_division=0,
    )

    tuned_f1 = f1_score(
        y_test,
        best_rf_predictions,
        zero_division=0,
    )

    tuned_auc = roc_auc_score(
        y_test,
        best_rf_probabilities,
    )

    print(
        f"\nBest parameters: "
        f"{best_params}"
    )

    print(
        f"Best CV F1: "
        f"{grid_search.best_score_:.4f}"
    )

    print(
        f"OOB score: "
        f"{best_oob_score:.4f}"
    )

    task12_results = pd.DataFrame(
        [
            {
                "n_estimators": (
                    best_params[
                        "classifier__n_estimators"
                    ]
                ),

                "max_depth": (
                    best_params[
                        "classifier__max_depth"
                    ]
                ),

                "max_features": (
                    best_params[
                        "classifier__max_features"
                    ]
                ),

                "best_cv_f1": (
                    grid_search.best_score_
                ),

                "oob_score": (
                    best_oob_score
                ),

                "test_accuracy": (
                    tuned_accuracy
                ),

                "test_precision": (
                    tuned_precision
                ),

                "test_recall": (
                    tuned_recall
                ),

                "test_f1": (
                    tuned_f1
                ),

                "test_auc": (
                    tuned_auc
                ),
            }
        ]
    )

    task12_results.to_csv(
        TASK12_RESULTS_FILE,
        index=False,
    )

    print("\n========================================")
    print("TASK 12 COMPLETED SUCCESSFULLY")
    print("========================================")

    # =====================================================
    # TASK 13 - REGRESSION
    # =====================================================

    print("\n========================================")
    print("TASK 13 - FARE REGRESSION")
    print("========================================")

    X_regression = df[
        REGRESSION_FEATURES
    ].copy()

    y_regression = df[
        REGRESSION_TARGET
    ].copy()

    (
        X_reg_train,
        X_reg_test,
        y_reg_train,
        y_reg_test,
    ) = train_test_split(
        X_regression,
        y_regression,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    regression_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_regression_preprocessor(),
            ),
            (
                "regressor",
                LinearRegression(),
            ),
        ]
    )

    regression_pipeline.fit(
        X_reg_train,
        y_reg_train,
    )

    regression_predictions = (
        regression_pipeline.predict(
            X_reg_test
        )
    )

    regression_mae = mean_absolute_error(
        y_reg_test,
        regression_predictions,
    )

    regression_rmse = np.sqrt(
        mean_squared_error(
            y_reg_test,
            regression_predictions,
        )
    )

    regression_r2 = r2_score(
        y_reg_test,
        regression_predictions,
    )

    number_of_predictors = len(
        regression_pipeline
        .named_steps[
            "preprocessor"
        ]
        .get_feature_names_out()
    )

    n_test = len(
        y_reg_test
    )

    adjusted_r2 = (
        1
        -
        (
            (1 - regression_r2)
            *
            (n_test - 1)
            /
            (
                n_test
                -
                number_of_predictors
                -
                1
            )
        )
    )

    residuals = (
        y_reg_test.to_numpy()
        -
        regression_predictions
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.scatter(
        regression_predictions,
        residuals,
        alpha=0.65,
    )

    plt.axhline(
        y=0,
        linestyle="--",
    )

    plt.xlabel(
        "Predicted Fare"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        "Fare Regression Residual Plot"
    )

    plt.tight_layout()

    plt.savefig(
        REGRESSION_RESIDUAL_FILE
    )

    plt.close()

    absolute_residuals = np.abs(
        residuals
    )

    residual_correlation = np.corrcoef(
        regression_predictions,
        absolute_residuals,
    )[0, 1]

    prediction_median = np.median(
        regression_predictions
    )

    lower_residuals = residuals[
        regression_predictions
        <=
        prediction_median
    ]

    upper_residuals = residuals[
        regression_predictions
        >
        prediction_median
    ]

    lower_std = np.std(
        lower_residuals
    )

    upper_std = np.std(
        upper_residuals
    )

    if lower_std == 0:

        spread_ratio = np.inf

    else:

        spread_ratio = (
            upper_std
            /
            lower_std
        )

    heteroscedasticity_detected = (
        abs(residual_correlation) >= 0.20
        or
        spread_ratio >= 1.50
        or
        spread_ratio <= (1 / 1.50)
    )

    if heteroscedasticity_detected:

        heteroscedasticity_statement = (
            "The residual plot shows evidence of "
            "heteroscedasticity because residual spread "
            "changes across predicted fare values."
        )

    else:

        heteroscedasticity_statement = (
            "The residual plot does not show strong evidence "
            "of heteroscedasticity because residual spread "
            "is reasonably stable across predicted fares."
        )

    regression_conclusion = (
        f"The fare regression produced MAE "
        f"{regression_mae:.4f}, RMSE "
        f"{regression_rmse:.4f}, R² "
        f"{regression_r2:.4f}, and Adjusted R² "
        f"{adjusted_r2:.4f}. "
        f"{heteroscedasticity_statement}"
    )

    regression_metrics = pd.DataFrame(
        [
            {
                "model": (
                    "Multivariate Linear Regression"
                ),
                "mae": regression_mae,
                "rmse": regression_rmse,
                "r2": regression_r2,
                "adjusted_r2": adjusted_r2,
                "heteroscedasticity_detected": (
                    heteroscedasticity_detected
                ),
            }
        ]
    )

    regression_metrics.to_csv(
        REGRESSION_RESULTS_FILE,
        index=False,
    )

    with open(
        REGRESSION_CONCLUSION_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            regression_conclusion
        )

    print("\n========================================")
    print("TASK 13 COMPLETED SUCCESSFULLY")
    print("========================================")

    # =====================================================
    # TASK 14 - FINAL COMPARISON
    # =====================================================

    print("\n========================================")
    print("TASK 14 - FINAL MODEL COMPARISON")
    print("========================================")

    final_comparison = pd.DataFrame(
        [
            {
                "model_type": "Classification",
                "model": "Logistic Regression",
                "classification_accuracy": (
                    logistic_result["accuracy"]
                ),
                "classification_precision": (
                    logistic_result["precision"]
                ),
                "classification_recall": (
                    logistic_result["recall"]
                ),
                "classification_f1": (
                    logistic_result["f1"]
                ),
                "classification_auc": (
                    logistic_result["auc"]
                ),
                "regression_mae": np.nan,
                "regression_rmse": np.nan,
                "regression_r2": np.nan,
                "regression_adjusted_r2": np.nan,
            },

            {
                "model_type": "Classification",
                "model": "Decision Tree",
                "classification_accuracy": (
                    tree_result["accuracy"]
                ),
                "classification_precision": (
                    tree_result["precision"]
                ),
                "classification_recall": (
                    tree_result["recall"]
                ),
                "classification_f1": (
                    tree_result["f1"]
                ),
                "classification_auc": (
                    tree_result["auc"]
                ),
                "regression_mae": np.nan,
                "regression_rmse": np.nan,
                "regression_r2": np.nan,
                "regression_adjusted_r2": np.nan,
            },

            {
                "model_type": "Classification",
                "model": "Random Forest",
                "classification_accuracy": (
                    forest_result["accuracy"]
                ),
                "classification_precision": (
                    forest_result["precision"]
                ),
                "classification_recall": (
                    forest_result["recall"]
                ),
                "classification_f1": (
                    forest_result["f1"]
                ),
                "classification_auc": (
                    forest_result["auc"]
                ),
                "regression_mae": np.nan,
                "regression_rmse": np.nan,
                "regression_r2": np.nan,
                "regression_adjusted_r2": np.nan,
            },

            {
                "model_type": "Regression",
                "model": (
                    "Multivariate Linear Regression"
                ),
                "classification_accuracy": np.nan,
                "classification_precision": np.nan,
                "classification_recall": np.nan,
                "classification_f1": np.nan,
                "classification_auc": np.nan,
                "regression_mae": regression_mae,
                "regression_rmse": regression_rmse,
                "regression_r2": regression_r2,
                "regression_adjusted_r2": adjusted_r2,
            },
        ]
    )

    final_comparison.to_csv(
        FINAL_COMPARISON_FILE,
        index=False,
    )

    print(
        final_comparison
        .round(4)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # BEST BASELINE CLASSIFIER
    # -----------------------------------------------------

    best_classifier_row = (
        model_comparison
        .sort_values(
            by=[
                "auc",
                "f1",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    best_classifier_name = (
        best_classifier_row[
            "model"
        ]
    )

    recommendation = (
        f"I recommend deploying the "
        f"{best_classifier_name} classifier among the three "
        f"baseline classifiers evaluated. "
        f"It achieved accuracy "
        f"{best_classifier_row['accuracy']:.4f}, "
        f"precision {best_classifier_row['precision']:.4f}, "
        f"recall {best_classifier_row['recall']:.4f}, "
        f"and F1 score {best_classifier_row['f1']:.4f}. "
        f"Its ROC-AUC of "
        f"{best_classifier_row['auc']:.4f} provides an "
        f"overall measure of discrimination between "
        f"survivors and non-survivors. "
        f"Classification scores are considered separately "
        f"from the fare-regression metrics because the two "
        f"model types solve different prediction problems."
    )

    with open(
        FINAL_RECOMMENDATION_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            recommendation
        )

    print("\n========================================")
    print("TASK 14 COMPLETED SUCCESSFULLY")
    print("========================================")

    # =====================================================
    # TASK 15 - SAVE COMPLETE BEST PIPELINE
    # =====================================================

    print("\n========================================")
    print("TASK 15 - SAVE COMPLETE FITTED PIPELINE")
    print("========================================")

    # -----------------------------------------------------
    # CHOOSE THE BEST COMPLETE CLASSIFIER PIPELINE
    # -----------------------------------------------------
    #
    # We compare:
    # Logistic Regression
    # Decision Tree
    # Random Forest
    # Tuned Random Forest
    #
    # The tuned RF is a complete sklearn Pipeline:
    #
    # raw data
    #   -> imputation
    #   -> one-hot encoding
    #   -> scaling
    #   -> Random Forest
    #
    # Therefore it can be saved directly and reused
    # end-to-end on raw input.

    tuned_rf_result = {
        "model": "Tuned Random Forest",
        "accuracy": tuned_accuracy,
        "precision": tuned_precision,
        "recall": tuned_recall,
        "f1": tuned_f1,
        "auc": tuned_auc,
    }

    deployment_candidates = pd.DataFrame(
        [
            logistic_result,
            tree_result,
            forest_result,
            tuned_rf_result,
        ]
    )

    print("\n========================================")
    print("DEPLOYMENT CANDIDATES")
    print("========================================")

    print(
        deployment_candidates
        .round(4)
        .to_string(
            index=False
        )
    )

    # Select the strongest candidate using:
    # 1. ROC-AUC
    # 2. F1 as tie-breaker
    best_deployment_row = (
        deployment_candidates
        .sort_values(
            by=[
                "auc",
                "f1",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    best_deployment_name = (
        best_deployment_row[
            "model"
        ]
    )

    # Map name to already-fitted COMPLETE pipeline
    candidate_pipelines = {
        "Logistic Regression": (
            logistic_pipeline
        ),

        "Decision Tree": (
            decision_tree_pipeline
        ),

        "Random Forest": (
            random_forest_pipeline
        ),

        "Tuned Random Forest": (
            best_rf_pipeline
        ),
    }

    best_full_pipeline = (
        candidate_pipelines[
            best_deployment_name
        ]
    )

    print(
        f"\nBest pipeline selected: "
        f"{best_deployment_name}"
    )

    print(
        f"Accuracy : "
        f"{best_deployment_row['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{best_deployment_row['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{best_deployment_row['recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{best_deployment_row['f1']:.4f}"
    )

    print(
        f"AUC      : "
        f"{best_deployment_row['auc']:.4f}"
    )

    # =====================================================
    # TASK 15A - VERIFY COMPLETE PIPELINE
    # =====================================================

    if not isinstance(
        best_full_pipeline,
        Pipeline,
    ):

        raise ValueError(
            "Task 15 failed: selected object "
            "is not a complete sklearn Pipeline."
        )

    if (
        "preprocessor"
        not in
        best_full_pipeline.named_steps
    ):

        raise ValueError(
            "Task 15 failed: preprocessing "
            "step is missing."
        )

    if (
        "classifier"
        not in
        best_full_pipeline.named_steps
    ):

        raise ValueError(
            "Task 15 failed: classifier "
            "step is missing."
        )

    print(
        "\nComplete pipeline structure verified."
    )

    print(
        "Preprocessor included: PASS"
    )

    print(
        "Classifier included: PASS"
    )

    # =====================================================
    # TASK 15B - SAVE PIPELINE
    # =====================================================

    joblib.dump(
        best_full_pipeline,
        BEST_PIPELINE_FILE,
    )

    print(
        f"\nComplete fitted pipeline saved to:\n"
        f"{BEST_PIPELINE_FILE}"
    )

    if not BEST_PIPELINE_FILE.exists():

        raise ValueError(
            "Task 15 failed: joblib file "
            "was not created."
        )

    # =====================================================
    # TASK 15C - DELETE IN-MEMORY REFERENCE
    # =====================================================
    #
    # This helps demonstrate that the next prediction
    # really comes from the reloaded artifact.

    del best_full_pipeline

    # =====================================================
    # TASK 15D - RELOAD PIPELINE
    # =====================================================

    reloaded_pipeline = joblib.load(
        BEST_PIPELINE_FILE
    )

    print(
        "\nPipeline reloaded successfully."
    )

    # =====================================================
    # TASK 15E - CREATE RAW INPUT
    # =====================================================
    #
    # This input is deliberately NOT:
    # - imputed
    # - encoded
    # - scaled
    #
    # It uses the same raw feature structure expected
    # by the complete pipeline.

    raw_new_passengers = pd.DataFrame(
        [
            {
                "pclass": 1,
                "sex": "female",
                "age": 29.0,
                "sibsp": 0,
                "parch": 0,
                "fare": 80.0,
                "embarked": "S",
            },

            {
                "pclass": 3,
                "sex": "male",

                # Deliberately missing value to prove
                # the saved imputer is included.
                "age": np.nan,

                "sibsp": 0,
                "parch": 0,
                "fare": 8.0,
                "embarked": "S",
            },
        ]
    )

    print("\n========================================")
    print("RAW INPUT USED AFTER RELOAD")
    print("========================================")

    print(
        raw_new_passengers
        .to_string(
            index=False
        )
    )

    # =====================================================
    # TASK 15F - PREDICT USING RELOADED PIPELINE
    # =====================================================

    reloaded_predictions = (
        reloaded_pipeline.predict(
            raw_new_passengers
        )
    )

    reloaded_probabilities = (
        reloaded_pipeline
        .predict_proba(
            raw_new_passengers
        )[:, 1]
    )

    prediction_results = (
        raw_new_passengers.copy()
    )

    prediction_results[
        "predicted_survived"
    ] = reloaded_predictions

    prediction_results[
        "survival_probability"
    ] = reloaded_probabilities

    print("\n========================================")
    print("RELOADED PIPELINE PREDICTIONS")
    print("========================================")

    print(
        prediction_results
        .round(
            {
                "survival_probability": 4
            }
        )
        .to_string(
            index=False
        )
    )

    # Save proof of reload prediction
    prediction_results.to_csv(
        TASK15_PREDICTION_FILE,
        index=False,
    )

    # =====================================================
    # TASK 15G - VALIDATION
    # =====================================================

    print("\n========================================")
    print("TASK 15 VALIDATION")
    print("========================================")

    if not isinstance(
        reloaded_pipeline,
        Pipeline,
    ):

        raise ValueError(
            "Task 15 failed: reloaded "
            "artifact is not a Pipeline."
        )

    if (
        "preprocessor"
        not in
        reloaded_pipeline.named_steps
    ):

        raise ValueError(
            "Task 15 failed: reloaded pipeline "
            "does not contain preprocessing."
        )

    if (
        "classifier"
        not in
        reloaded_pipeline.named_steps
    ):

        raise ValueError(
            "Task 15 failed: reloaded pipeline "
            "does not contain a classifier."
        )

    if (
        len(reloaded_predictions)
        !=
        len(raw_new_passengers)
    ):

        raise ValueError(
            "Task 15 failed: prediction "
            "count is incorrect."
        )

    if not np.all(
        np.isfinite(
            reloaded_probabilities
        )
    ):

        raise ValueError(
            "Task 15 failed: invalid "
            "probabilities returned."
        )

    if (
        (
            reloaded_probabilities
            <
            0
        ).any()
        or
        (
            reloaded_probabilities
            >
            1
        ).any()
    ):

        raise ValueError(
            "Task 15 failed: probability "
            "outside 0-1 range."
        )

    if not TASK15_PREDICTION_FILE.exists():

        raise ValueError(
            "Task 15 failed: reload prediction "
            "CSV was not created."
        )

    print(
        "Complete fitted pipeline selected: PASS"
    )

    print(
        "Preprocessing saved with classifier: PASS"
    )

    print(
        "joblib.dump completed: PASS"
    )

    print(
        "joblib.load completed: PASS"
    )

    print(
        "Raw unprocessed DataFrame accepted: PASS"
    )

    print(
        "Missing raw age handled by saved imputer: PASS"
    )

    print(
        "Categorical raw values encoded internally: PASS"
    )

    print(
        "Numeric raw values scaled internally: PASS"
    )

    print(
        "Reloaded pipeline predictions successful: PASS"
    )

    print(
        "Prediction probabilities successful: PASS"
    )

    # =====================================================
    # TASK 15 SUCCESS
    # =====================================================

    print("\n========================================")
    print("TASK 15 COMPLETED SUCCESSFULLY")
    print("========================================")

    # =====================================================
    # COMPLETE MODULE 2 STATUS
    # =====================================================

    print("\n========================================")
    print("MODULE 2 TASKS 1 TO 15 COMPLETE")
    print("========================================")

    print(
        "Tasks 1-6: COMPLETE in 01_eda.py"
    )

    print(
        "Tasks 7-15: COMPLETE in 02_modeling.py"
    )

    print(
        "\nBest saved model:"
    )

    print(
        best_deployment_name
    )

    print(
        f"\nSaved complete pipeline:\n"
        f"{BEST_PIPELINE_FILE}"
    )

    print(
        "\nMODULE 2 ANALYTICS PIPELINE "
        "COMPLETED SUCCESSFULLY"
    )


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()