from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler


# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).parent

TITANIC_FILE = BASE_DIR / "titanic.csv"

CHARTS_DIR = BASE_DIR / "charts"

INTERPRETATION_FILE = (
    BASE_DIR / "task5_interpretations.txt"
)

STANDARDIZATION_FILE = (
    BASE_DIR / "task6_standardization_summary.csv"
)

CHARTS_DIR.mkdir(
    exist_ok=True
)


# =========================================================
# HELPER FUNCTION - IQR OUTLIER COUNT
# =========================================================

def calculate_iqr_outliers(series):
    """
    Calculate IQR boundaries and count outliers.

    Outliers are values outside:
    [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
    """

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outliers = series[
        (series < lower_bound)
        |
        (series > upper_bound)
    ]

    return (
        q1,
        q3,
        iqr,
        lower_bound,
        upper_bound,
        len(outliers)
    )


# =========================================================
# HELPER FUNCTION - SAVE TASK 5 INTERPRETATION
# =========================================================

def save_interpretation(
    file_handle,
    chart_number,
    chart_title,
    interpretation
):
    """
    Print and save a Task 5 chart interpretation.
    """

    section = (
        f"\nChart {chart_number}: {chart_title}\n"
        f"{'-' * 60}\n"
        f"{interpretation}\n"
    )

    print(section)

    file_handle.write(section)


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    print("\n========================================")
    print("MODULE 2 - ANALYTICS PIPELINE")
    print("PART A - EDA")
    print("TASKS 1 TO 6")
    print("========================================")

    # =====================================================
    # TASK 1 - LOAD TITANIC DATASET
    # =====================================================

    print("\n========================================")
    print("TASK 1 - LOAD AND PROFILE DATA")
    print("========================================")

    print("\nLoading Titanic dataset...")

    # IMPORTANT:
    # This is the ONLY sns.load_dataset("titanic")
    # call in the analytics module.
    df = sns.load_dataset(
        "titanic"
    )

    print(
        "\nTitanic dataset loaded successfully."
    )

    # -----------------------------------------------------
    # SAVE RAW OFFLINE FALLBACK IMMEDIATELY
    # -----------------------------------------------------

    df.to_csv(
        TITANIC_FILE,
        index=False
    )

    print(
        f"\nOffline fallback saved to:\n"
        f"{TITANIC_FILE}"
    )

    # -----------------------------------------------------
    # SHAPE
    # -----------------------------------------------------

    print("\n========================================")
    print("DATASET SHAPE")
    print("========================================")

    print(df.shape)

    print(
        f"\nNumber of rows: {df.shape[0]}"
    )

    print(
        f"Number of columns: {df.shape[1]}"
    )

    # -----------------------------------------------------
    # INFO
    # -----------------------------------------------------

    print("\n========================================")
    print("DF.INFO()")
    print("========================================")

    df.info()

    # -----------------------------------------------------
    # DESCRIBE
    # -----------------------------------------------------

    print("\n========================================")
    print("DF.DESCRIBE()")
    print("========================================")

    print(
        df.describe().to_string()
    )

    # -----------------------------------------------------
    # MISSING VALUE PERCENTAGES
    # -----------------------------------------------------

    print("\n========================================")
    print("MISSING VALUE PERCENTAGES")
    print("========================================")

    missing_percentage = (
        df.isnull().mean() * 100
    )

    affected_columns = (
        missing_percentage[
            missing_percentage > 0
        ]
        .sort_values(
            ascending=False
        )
    )

    print(
        affected_columns.to_string()
    )

    # -----------------------------------------------------
    # SURVIVAL CLASS BALANCE
    # -----------------------------------------------------

    print("\n========================================")
    print("SURVIVAL CLASS BALANCE")
    print("========================================")

    class_counts = (
        df["survived"]
        .value_counts()
        .sort_index()
    )

    class_percentages = (
        df["survived"]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    print("\nClass counts:")

    print(class_counts)

    print("\nClass percentages:")

    print(
        class_percentages.round(2)
    )

    # -----------------------------------------------------
    # TASK 1 VALIDATION
    # -----------------------------------------------------

    if df.empty:
        raise ValueError(
            "Task 1 failed: Titanic dataset is empty."
        )

    if not TITANIC_FILE.exists():
        raise ValueError(
            "Task 1 failed: titanic.csv was not created."
        )

    if "survived" not in df.columns:
        raise ValueError(
            "Task 1 failed: survived column is missing."
        )

    print("\n========================================")
    print("TASK 1 COMPLETED SUCCESSFULLY")
    print("========================================")

    # =====================================================
    # TASK 2 - MISSING VALUE HANDLING
    # =====================================================

    print("\n========================================")
    print("TASK 2 - MISSING VALUE HANDLING")
    print("========================================")

    rows_before_cleaning = len(df)

    print(
        f"\nRows before cleaning: "
        f"{rows_before_cleaning}"
    )

    # -----------------------------------------------------
    # DECK
    # -----------------------------------------------------

    deck_missing_pct = (
        df["deck"]
        .isnull()
        .mean()
        * 100
    )

    print(
        f"\ndeck missing: "
        f"{deck_missing_pct:.6f}%"
    )

    print(
        "Decision: DROP COLUMN."
    )

    print(
        "Reason: deck has extremely high missingness, "
        "so reliable imputation would not be defensible."
    )

    df = df.drop(
        columns=[
            "deck"
        ]
    )

    # -----------------------------------------------------
    # AGE
    # -----------------------------------------------------

    age_missing_pct = (
        df["age"]
        .isnull()
        .mean()
        * 100
    )

    print(
        f"\nage missing: "
        f"{age_missing_pct:.6f}%"
    )

    print(
        "Decision: IMPUTE."
    )

    print(
        "Reason: age missingness falls between "
        "5% and 30%."
    )

    age_median = (
        df["age"]
        .median()
    )

    print(
        f"Median age used for imputation: "
        f"{age_median:.2f}"
    )

    df["age"] = (
        df["age"]
        .fillna(
            age_median
        )
    )

    # -----------------------------------------------------
    # EMBARKED
    # -----------------------------------------------------

    embarked_missing_pct = (
        df["embarked"]
        .isnull()
        .mean()
        * 100
    )

    print(
        f"\nembarked missing: "
        f"{embarked_missing_pct:.6f}%"
    )

    print(
        "Decision: DROP AFFECTED ROWS."
    )

    print(
        "Reason: embarked missingness is below 5%."
    )

    # -----------------------------------------------------
    # EMBARK TOWN
    # -----------------------------------------------------

    embark_town_missing_pct = (
        df["embark_town"]
        .isnull()
        .mean()
        * 100
    )

    print(
        f"\nembark_town missing: "
        f"{embark_town_missing_pct:.6f}%"
    )

    print(
        "Decision: DROP AFFECTED ROWS."
    )

    print(
        "Reason: embark_town missingness is below 5%."
    )

    # Drop rows missing either field
    df = (
        df
        .dropna(
            subset=[
                "embarked",
                "embark_town"
            ]
        )
        .copy()
    )

    df = df.reset_index(
        drop=True
    )

    # -----------------------------------------------------
    # CLEANING SUMMARY
    # -----------------------------------------------------

    rows_after_cleaning = len(df)

    rows_removed = (
        rows_before_cleaning
        - rows_after_cleaning
    )

    print("\n========================================")
    print("CLEANING SUMMARY")
    print("========================================")

    print(
        f"Rows before cleaning: "
        f"{rows_before_cleaning}"
    )

    print(
        f"Rows after cleaning: "
        f"{rows_after_cleaning}"
    )

    print(
        f"Rows removed: "
        f"{rows_removed}"
    )

    remaining_missing = (
        df.isnull().sum()
    )

    remaining_missing = (
        remaining_missing[
            remaining_missing > 0
        ]
    )

    print(
        "\nRemaining missing values:"
    )

    if remaining_missing.empty:

        print(
            "No missing values remain."
        )

    else:

        print(
            remaining_missing.to_string()
        )

    # -----------------------------------------------------
    # TASK 2 VALIDATION
    # -----------------------------------------------------

    if "deck" in df.columns:

        raise ValueError(
            "Task 2 failed: deck was not dropped."
        )

    if df["age"].isnull().any():

        raise ValueError(
            "Task 2 failed: age still contains missing values."
        )

    if df["embarked"].isnull().any():

        raise ValueError(
            "Task 2 failed: embarked still contains missing values."
        )

    if df["embark_town"].isnull().any():

        raise ValueError(
            "Task 2 failed: embark_town still contains missing values."
        )

    print("\n========================================")
    print("TASK 2 COMPLETED SUCCESSFULLY")
    print("========================================")

    # =====================================================
    # TASK 3 - UNIVARIATE ANALYSIS
    # =====================================================

    print("\n========================================")
    print("TASK 3 - UNIVARIATE ANALYSIS")
    print("========================================")

    # -----------------------------------------------------
    # AGE HISTOGRAM
    # -----------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    sns.histplot(
        data=df,
        x="age",
        bins=30,
        kde=True
    )

    plt.title(
        "Titanic Passenger Age Distribution"
    )

    plt.xlabel(
        "Age"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    age_hist_file = (
        CHARTS_DIR
        / "age_histogram.png"
    )

    plt.savefig(
        age_hist_file
    )

    plt.close()

    # -----------------------------------------------------
    # AGE BOX PLOT
    # -----------------------------------------------------

    plt.figure(
        figsize=(8, 4)
    )

    sns.boxplot(
        data=df,
        x="age"
    )

    plt.title(
        "Titanic Passenger Age Box Plot"
    )

    plt.xlabel(
        "Age"
    )

    plt.tight_layout()

    age_box_file = (
        CHARTS_DIR
        / "age_boxplot.png"
    )

    plt.savefig(
        age_box_file
    )

    plt.close()

    # -----------------------------------------------------
    # FARE HISTOGRAM
    # -----------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    sns.histplot(
        data=df,
        x="fare",
        bins=30,
        kde=True
    )

    plt.title(
        "Titanic Fare Distribution"
    )

    plt.xlabel(
        "Fare"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    fare_hist_file = (
        CHARTS_DIR
        / "fare_histogram.png"
    )

    plt.savefig(
        fare_hist_file
    )

    plt.close()

    # -----------------------------------------------------
    # FARE BOX PLOT
    # -----------------------------------------------------

    plt.figure(
        figsize=(8, 4)
    )

    sns.boxplot(
        data=df,
        x="fare"
    )

    plt.title(
        "Titanic Fare Box Plot"
    )

    plt.xlabel(
        "Fare"
    )

    plt.tight_layout()

    fare_box_file = (
        CHARTS_DIR
        / "fare_boxplot.png"
    )

    plt.savefig(
        fare_box_file
    )

    plt.close()

    # -----------------------------------------------------
    # AGE IQR OUTLIERS
    # -----------------------------------------------------

    (
        age_q1,
        age_q3,
        age_iqr,
        age_lower,
        age_upper,
        age_outlier_count
    ) = calculate_iqr_outliers(
        df["age"]
    )

    print("\n========================================")
    print("AGE IQR OUTLIER ANALYSIS")
    print("========================================")

    print(
        f"Q1: {age_q1:.2f}"
    )

    print(
        f"Q3: {age_q3:.2f}"
    )

    print(
        f"IQR: {age_iqr:.2f}"
    )

    print(
        f"Lower bound: "
        f"{age_lower:.2f}"
    )

    print(
        f"Upper bound: "
        f"{age_upper:.2f}"
    )

    print(
        f"Age outlier count: "
        f"{age_outlier_count}"
    )

    # -----------------------------------------------------
    # FARE IQR OUTLIERS
    # -----------------------------------------------------

    (
        fare_q1,
        fare_q3,
        fare_iqr,
        fare_lower,
        fare_upper,
        fare_outlier_count
    ) = calculate_iqr_outliers(
        df["fare"]
    )

    print("\n========================================")
    print("FARE IQR OUTLIER ANALYSIS")
    print("========================================")

    print(
        f"Q1: {fare_q1:.2f}"
    )

    print(
        f"Q3: {fare_q3:.2f}"
    )

    print(
        f"IQR: {fare_iqr:.2f}"
    )

    print(
        f"Lower bound: "
        f"{fare_lower:.2f}"
    )

    print(
        f"Upper bound: "
        f"{fare_upper:.2f}"
    )

    print(
        f"Fare outlier count: "
        f"{fare_outlier_count}"
    )

    # -----------------------------------------------------
    # FARE MEAN / MEDIAN / MODE
    # -----------------------------------------------------

    fare_mean = (
        df["fare"].mean()
    )

    fare_median = (
        df["fare"].median()
    )

    fare_mode = (
        df["fare"]
        .mode()
        .iloc[0]
    )

    print("\n========================================")
    print("FARE CENTRAL TENDENCY")
    print("========================================")

    print(
        f"Fare mean: "
        f"{fare_mean:.4f}"
    )

    print(
        f"Fare median: "
        f"{fare_median:.4f}"
    )

    print(
        f"Fare mode: "
        f"{fare_mode:.4f}"
    )

    # -----------------------------------------------------
    # FARE SKEWNESS
    # -----------------------------------------------------

    if (
        fare_mean
        >
        fare_median
        >=
        fare_mode
    ):

        fare_distribution = (
            "right-skewed"
        )

    elif (
        fare_mean
        <
        fare_median
        <=
        fare_mode
    ):

        fare_distribution = (
            "left-skewed"
        )

    else:

        fare_distribution = (
            "approximately symmetric "
            "or not clearly ordered"
        )

    print("\n========================================")
    print("FARE DISTRIBUTION INTERPRETATION")
    print("========================================")

    print(
        f"The fare distribution is "
        f"{fare_distribution}."
    )

    print(
        f"Mean = {fare_mean:.4f}, "
        f"Median = {fare_median:.4f}, "
        f"Mode = {fare_mode:.4f}."
    )

    # -----------------------------------------------------
    # TASK 3 VALIDATION
    # -----------------------------------------------------

    required_task3_charts = [
        age_hist_file,
        age_box_file,
        fare_hist_file,
        fare_box_file
    ]

    for chart_file in required_task3_charts:

        if not chart_file.exists():

            raise ValueError(
                f"Task 3 failed: "
                f"{chart_file.name} was not created."
            )

    print("\n========================================")
    print("TASK 3 COMPLETED SUCCESSFULLY")
    print("========================================")

    # =====================================================
    # TASK 4 - BIVARIATE ANALYSIS
    # =====================================================

    print("\n========================================")
    print("TASK 4 - BIVARIATE ANALYSIS")
    print("========================================")

    # -----------------------------------------------------
    # SURVIVAL BY SEX
    # -----------------------------------------------------

    female_mask = (
        df["sex"] == "female"
    )

    male_mask = (
        df["sex"] == "male"
    )

    female_survival_rate = (
        df.loc[
            female_mask,
            "survived"
        ]
        .mean()
        * 100
    )

    male_survival_rate = (
        df.loc[
            male_mask,
            "survived"
        ]
        .mean()
        * 100
    )

    print("\n========================================")
    print("SURVIVAL RATE BY SEX")
    print("========================================")

    print(
        f"Female survival rate: "
        f"{female_survival_rate:.2f}%"
    )

    print(
        f"Male survival rate: "
        f"{male_survival_rate:.2f}%"
    )

    # -----------------------------------------------------
    # SURVIVAL BY PCLASS
    # -----------------------------------------------------

    pclass_survival_rates = {}

    print("\n========================================")
    print("SURVIVAL RATE BY PASSENGER CLASS")
    print("========================================")

    for passenger_class in [
        1,
        2,
        3
    ]:

        class_mask = (
            df["pclass"]
            ==
            passenger_class
        )

        survival_rate = (
            df.loc[
                class_mask,
                "survived"
            ]
            .mean()
            * 100
        )

        pclass_survival_rates[
            passenger_class
        ] = survival_rate

        print(
            f"Class {passenger_class} "
            f"survival rate: "
            f"{survival_rate:.2f}%"
        )

    # -----------------------------------------------------
    # SURVIVAL BY SEX + PCLASS
    # -----------------------------------------------------

    print("\n========================================")
    print("SURVIVAL RATE BY SEX AND PCLASS")
    print("========================================")

    sex_pclass_results = []

    for sex_value in [
        "female",
        "male"
    ]:

        for passenger_class in [
            1,
            2,
            3
        ]:

            combined_mask = (
                (df["sex"] == sex_value)
                &
                (
                    df["pclass"]
                    ==
                    passenger_class
                )
            )

            passenger_count = (
                combined_mask.sum()
            )

            survival_rate = (
                df.loc[
                    combined_mask,
                    "survived"
                ]
                .mean()
                * 100
            )

            sex_pclass_results.append(
                {
                    "sex": sex_value,
                    "pclass": passenger_class,
                    "passenger_count": passenger_count,
                    "survival_rate_percent": survival_rate
                }
            )

            print(
                f"{sex_value.capitalize()} "
                f"Class {passenger_class}: "
                f"{survival_rate:.2f}% survived "
                f"(n={passenger_count})"
            )

    sex_pclass_df = pd.DataFrame(
        sex_pclass_results
    )

    # -----------------------------------------------------
    # EXACT 6-COLUMN CORRELATION MATRIX
    # -----------------------------------------------------

    correlation_columns = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare"
    ]

    correlation_matrix = (
        df[
            correlation_columns
        ]
        .corr()
    )

    print("\n========================================")
    print("6x6 CORRELATION MATRIX")
    print("========================================")

    print(
        correlation_matrix
        .round(4)
        .to_string()
    )

    # -----------------------------------------------------
    # HEATMAP
    # -----------------------------------------------------

    plt.figure(
        figsize=(9, 7)
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        center=0,
        square=True
    )

    plt.title(
        "Titanic Correlation Matrix"
    )

    plt.tight_layout()

    correlation_heatmap_file = (
        CHARTS_DIR
        / "correlation_heatmap.png"
    )

    plt.savefig(
        correlation_heatmap_file
    )

    plt.close()

    # -----------------------------------------------------
    # TWO STRONGEST CORRELATIONS
    # -----------------------------------------------------

    correlation_pairs = []

    for column_a, column_b in combinations(
        correlation_columns,
        2
    ):

        correlation_value = (
            correlation_matrix.loc[
                column_a,
                column_b
            ]
        )

        correlation_pairs.append(
            {
                "feature_1": column_a,
                "feature_2": column_b,
                "correlation": correlation_value,
                "absolute_correlation": abs(
                    correlation_value
                )
            }
        )

    correlation_pairs_df = (
        pd.DataFrame(
            correlation_pairs
        )
        .sort_values(
            by="absolute_correlation",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    strongest_two = (
        correlation_pairs_df
        .head(2)
    )

    print("\n========================================")
    print("TWO STRONGEST CORRELATIONS")
    print("========================================")

    for position, row in strongest_two.iterrows():

        direction = (
            "positive"
            if row["correlation"] > 0
            else "negative"
        )

        print(
            f"\n{position + 1}. "
            f"{row['feature_1']} vs "
            f"{row['feature_2']}"
        )

        print(
            f"Correlation: "
            f"{row['correlation']:.4f}"
        )

        print(
            f"Interpretation: "
            f"{direction} relationship."
        )

    # -----------------------------------------------------
    # TASK 4 VALIDATION
    # -----------------------------------------------------

    if correlation_matrix.shape != (
        6,
        6
    ):

        raise ValueError(
            "Task 4 failed: correlation matrix "
            "must be exactly 6x6."
        )

    if (
        "adult_male"
        in correlation_matrix.columns
    ):

        raise ValueError(
            "Task 4 failed: adult_male "
            "must be excluded."
        )

    if (
        "alone"
        in correlation_matrix.columns
    ):

        raise ValueError(
            "Task 4 failed: alone "
            "must be excluded."
        )

    print("\n========================================")
    print("TASK 4 COMPLETED SUCCESSFULLY")
    print("========================================")

    # =====================================================
    # TASK 5 - MULTIVARIATE DATA STORY
    # =====================================================

    print("\n========================================")
    print("TASK 5 - MULTIVARIATE DATA STORY")
    print("========================================")

    with open(
        INTERPRETATION_FILE,
        "w",
        encoding="utf-8"
    ) as interpretation_file:

        # -------------------------------------------------
        # CHART 1 - SURVIVAL BY SEX
        # -------------------------------------------------

        plt.figure(
            figsize=(7, 5)
        )

        sns.barplot(
            data=df,
            x="sex",
            y="survived",
            errorbar=None
        )

        plt.title(
            "Survival Rate by Sex"
        )

        plt.xlabel(
            "Sex"
        )

        plt.ylabel(
            "Survival Rate"
        )

        plt.ylim(
            0,
            1
        )

        plt.tight_layout()

        survival_sex_file = (
            CHARTS_DIR
            / "survival_by_sex.png"
        )

        plt.savefig(
            survival_sex_file
        )

        plt.close()

        sex_interpretation = (
            f"Female passengers had a survival rate of "
            f"{female_survival_rate:.2f}%, compared with "
            f"{male_survival_rate:.2f}% for male passengers. "
            "The large difference indicates that sex was "
            "strongly associated with survival. "
            "Female passengers were substantially more "
            "likely to survive."
        )

        save_interpretation(
            interpretation_file,
            1,
            "Survival Rate by Sex",
            sex_interpretation
        )

        # -------------------------------------------------
        # CHART 2 - SURVIVAL BY PCLASS
        # -------------------------------------------------

        plt.figure(
            figsize=(7, 5)
        )

        sns.barplot(
            data=df,
            x="pclass",
            y="survived",
            errorbar=None
        )

        plt.title(
            "Survival Rate by Passenger Class"
        )

        plt.xlabel(
            "Passenger Class"
        )

        plt.ylabel(
            "Survival Rate"
        )

        plt.ylim(
            0,
            1
        )

        plt.tight_layout()

        survival_pclass_file = (
            CHARTS_DIR
            / "survival_by_pclass.png"
        )

        plt.savefig(
            survival_pclass_file
        )

        plt.close()

        pclass_interpretation = (
            f"First-class passengers had a survival rate of "
            f"{pclass_survival_rates[1]:.2f}%, compared with "
            f"{pclass_survival_rates[2]:.2f}% in second class "
            f"and {pclass_survival_rates[3]:.2f}% in third class. "
            "Survival declined as passenger class moved from "
            "first to third. "
            "This suggests socioeconomic position was "
            "associated with survival opportunities."
        )

        save_interpretation(
            interpretation_file,
            2,
            "Survival Rate by Passenger Class",
            pclass_interpretation
        )

        # -------------------------------------------------
        # CHART 3 - SEX + PCLASS
        # -------------------------------------------------

        plt.figure(
            figsize=(8, 5)
        )

        sns.barplot(
            data=df,
            x="pclass",
            y="survived",
            hue="sex",
            errorbar=None
        )

        plt.title(
            "Survival Rate by Sex and Passenger Class"
        )

        plt.xlabel(
            "Passenger Class"
        )

        plt.ylabel(
            "Survival Rate"
        )

        plt.ylim(
            0,
            1
        )

        plt.tight_layout()

        sex_pclass_chart_file = (
            CHARTS_DIR
            / "survival_by_sex_pclass.png"
        )

        plt.savefig(
            sex_pclass_chart_file
        )

        plt.close()

        female_class1 = (
            sex_pclass_df[
                (sex_pclass_df["sex"] == "female")
                &
                (sex_pclass_df["pclass"] == 1)
            ][
                "survival_rate_percent"
            ]
            .iloc[0]
        )

        female_class3 = (
            sex_pclass_df[
                (sex_pclass_df["sex"] == "female")
                &
                (sex_pclass_df["pclass"] == 3)
            ][
                "survival_rate_percent"
            ]
            .iloc[0]
        )

        male_class1 = (
            sex_pclass_df[
                (sex_pclass_df["sex"] == "male")
                &
                (sex_pclass_df["pclass"] == 1)
            ][
                "survival_rate_percent"
            ]
            .iloc[0]
        )

        male_class3 = (
            sex_pclass_df[
                (sex_pclass_df["sex"] == "male")
                &
                (sex_pclass_df["pclass"] == 3)
            ][
                "survival_rate_percent"
            ]
            .iloc[0]
        )

        sex_pclass_interpretation = (
            f"Survival depended on both sex and passenger class. "
            f"First-class females had a survival rate of "
            f"{female_class1:.2f}%, while third-class females had "
            f"{female_class3:.2f}%; first-class males had "
            f"{male_class1:.2f}% compared with "
            f"{male_class3:.2f}% for third-class males. "
            "The pattern suggests that being female provided a "
            "major survival advantage, while higher passenger "
            "class added another advantage."
        )

        save_interpretation(
            interpretation_file,
            3,
            "Survival Rate by Sex and Passenger Class",
            sex_pclass_interpretation
        )

        # -------------------------------------------------
        # CHART 4 - AGE + FARE + SURVIVAL
        # -------------------------------------------------

        plt.figure(
            figsize=(9, 6)
        )

        sns.scatterplot(
            data=df,
            x="age",
            y="fare",
            hue="survived",
            alpha=0.65
        )

        plt.title(
            "Age and Fare by Survival Status"
        )

        plt.xlabel(
            "Age"
        )

        plt.ylabel(
            "Fare"
        )

        plt.tight_layout()

        age_fare_survival_file = (
            CHARTS_DIR
            / "age_fare_survival_scatter.png"
        )

        plt.savefig(
            age_fare_survival_file
        )

        plt.close()

        survived_fare_median = (
            df.loc[
                df["survived"] == 1,
                "fare"
            ]
            .median()
        )

        not_survived_fare_median = (
            df.loc[
                df["survived"] == 0,
                "fare"
            ]
            .median()
        )

        scatter_interpretation = (
            "The age-versus-fare scatter plot shows that "
            "survival was not determined by age alone, because "
            "both survivors and non-survivors appear across a "
            "wide range of ages. "
            f"Survivors had a median fare of "
            f"{survived_fare_median:.2f}, compared with "
            f"{not_survived_fare_median:.2f} for non-survivors. "
            "This supports the passenger-class analysis by "
            "showing that passengers paying higher fares were "
            "more strongly represented among survivors."
        )

        save_interpretation(
            interpretation_file,
            4,
            "Age and Fare by Survival Status",
            scatter_interpretation
        )

    # -----------------------------------------------------
    # TASK 5 VALIDATION
    # -----------------------------------------------------

    required_task5_charts = [
        survival_sex_file,
        survival_pclass_file,
        sex_pclass_chart_file,
        age_fare_survival_file
    ]

    for chart_file in required_task5_charts:

        if not chart_file.exists():

            raise ValueError(
                f"Task 5 failed: "
                f"{chart_file.name} was not created."
            )

    if not INTERPRETATION_FILE.exists():

        raise ValueError(
            "Task 5 failed: interpretations were not saved."
        )

    print("\n========================================")
    print("TASK 5 COMPLETED SUCCESSFULLY")
    print("========================================")

    # =====================================================
    # TASK 6 - EXPLORATORY STANDARDIZATION
    # =====================================================

    print("\n========================================")
    print("TASK 6 - EXPLORATORY STANDARDIZATION")
    print("========================================")

    print(
        "\nThis is an EDA-only standardization check."
    )

    print(
        "These standardized values will NOT be used "
        "directly in the modeling pipeline."
    )

    # -----------------------------------------------------
    # BEFORE STANDARDIZATION
    # -----------------------------------------------------

    print("\n========================================")
    print("BEFORE STANDARDIZATION")
    print("========================================")

    age_mean_before = (
        df["age"].mean()
    )

    age_std_before = (
        df["age"].std(
            ddof=0
        )
    )

    fare_mean_before = (
        df["fare"].mean()
    )

    fare_std_before = (
        df["fare"].std(
            ddof=0
        )
    )

    print(
        f"Age mean before: "
        f"{age_mean_before:.6f}"
    )

    print(
        f"Age std before: "
        f"{age_std_before:.6f}"
    )

    print(
        f"Fare mean before: "
        f"{fare_mean_before:.6f}"
    )

    print(
        f"Fare std before: "
        f"{fare_std_before:.6f}"
    )

    # -----------------------------------------------------
    # STANDARDIZE AGE AND FARE
    # -----------------------------------------------------

    scaler = StandardScaler()

    standardized_values = (
        scaler.fit_transform(
            df[
                [
                    "age",
                    "fare"
                ]
            ]
        )
    )

    # IMPORTANT:
    # Create NEW EDA columns.
    # Do not overwrite the original age/fare columns.
    df["age_zscore"] = (
        standardized_values[
            :,
            0
        ]
    )

    df["fare_zscore"] = (
        standardized_values[
            :,
            1
        ]
    )

    # -----------------------------------------------------
    # AFTER STANDARDIZATION
    # -----------------------------------------------------

    age_mean_after = (
        df["age_zscore"].mean()
    )

    age_std_after = (
        df["age_zscore"].std(
            ddof=0
        )
    )

    fare_mean_after = (
        df["fare_zscore"].mean()
    )

    fare_std_after = (
        df["fare_zscore"].std(
            ddof=0
        )
    )

    print("\n========================================")
    print("AFTER STANDARDIZATION")
    print("========================================")

    print(
        f"Age z-score mean: "
        f"{age_mean_after:.8f}"
    )

    print(
        f"Age z-score std: "
        f"{age_std_after:.8f}"
    )

    print(
        f"Fare z-score mean: "
        f"{fare_mean_after:.8f}"
    )

    print(
        f"Fare z-score std: "
        f"{fare_std_after:.8f}"
    )

    # -----------------------------------------------------
    # CREATE BEFORE/AFTER SUMMARY TABLE
    # -----------------------------------------------------

    standardization_summary = pd.DataFrame(
        {
            "feature": [
                "age",
                "fare"
            ],

            "mean_before": [
                age_mean_before,
                fare_mean_before
            ],

            "std_before": [
                age_std_before,
                fare_std_before
            ],

            "mean_after": [
                age_mean_after,
                fare_mean_after
            ],

            "std_after": [
                age_std_after,
                fare_std_after
            ]
        }
    )

    print("\n========================================")
    print("STANDARDIZATION BEFORE/AFTER SUMMARY")
    print("========================================")

    print(
        standardization_summary
        .round(6)
        .to_string(
            index=False
        )
    )

    # Save summary
    standardization_summary.to_csv(
        STANDARDIZATION_FILE,
        index=False
    )

    print(
        f"\nStandardization summary saved to:\n"
        f"{STANDARDIZATION_FILE}"
    )

    # -----------------------------------------------------
    # OPTIONAL VISUAL CHECK - AGE
    # -----------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    sns.histplot(
        data=df,
        x="age_zscore",
        bins=30,
        kde=True
    )

    plt.title(
        "Standardized Age Distribution"
    )

    plt.xlabel(
        "Age Z-Score"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    age_standardized_file = (
        CHARTS_DIR
        / "age_standardized.png"
    )

    plt.savefig(
        age_standardized_file
    )

    plt.close()

    # -----------------------------------------------------
    # OPTIONAL VISUAL CHECK - FARE
    # -----------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    sns.histplot(
        data=df,
        x="fare_zscore",
        bins=30,
        kde=True
    )

    plt.title(
        "Standardized Fare Distribution"
    )

    plt.xlabel(
        "Fare Z-Score"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    fare_standardized_file = (
        CHARTS_DIR
        / "fare_standardized.png"
    )

    plt.savefig(
        fare_standardized_file
    )

    plt.close()

    # =====================================================
    # TASK 6 INTERPRETATION
    # =====================================================

    print("\n========================================")
    print("TASK 6 INTERPRETATION")
    print("========================================")

    print(
        "After standardization, both age and fare "
        "have means approximately equal to 0 and "
        "standard deviations approximately equal to 1."
    )

    print(
        "This confirms that the z-score transformation "
        "was applied correctly."
    )

    print(
        "This transformation is only an exploratory "
        "EDA-stage check and will not feed directly into "
        "the modeling pipeline."
    )

    print(
        "The modeling pipeline will fit its own scaler "
        "using training data only to prevent data leakage."
    )

    # =====================================================
    # TASK 6 VALIDATION
    # =====================================================

    print("\n========================================")
    print("TASK 6 VALIDATION")
    print("========================================")

    tolerance = 1e-6

    if abs(
        age_mean_after
    ) > tolerance:

        raise ValueError(
            "Task 6 failed: standardized age "
            "mean is not approximately zero."
        )

    if abs(
        fare_mean_after
    ) > tolerance:

        raise ValueError(
            "Task 6 failed: standardized fare "
            "mean is not approximately zero."
        )

    if abs(
        age_std_after - 1
    ) > tolerance:

        raise ValueError(
            "Task 6 failed: standardized age "
            "standard deviation is not approximately 1."
        )

    if abs(
        fare_std_after - 1
    ) > tolerance:

        raise ValueError(
            "Task 6 failed: standardized fare "
            "standard deviation is not approximately 1."
        )

    if not STANDARDIZATION_FILE.exists():

        raise ValueError(
            "Task 6 failed: standardization summary "
            "file was not created."
        )

    if not age_standardized_file.exists():

        raise ValueError(
            "Task 6 failed: standardized age "
            "chart was not created."
        )

    if not fare_standardized_file.exists():

        raise ValueError(
            "Task 6 failed: standardized fare "
            "chart was not created."
        )

    print(
        "Age standardized: PASS"
    )

    print(
        "Fare standardized: PASS"
    )

    print(
        "Age mean approximately 0: PASS"
    )

    print(
        "Age standard deviation approximately 1: PASS"
    )

    print(
        "Fare mean approximately 0: PASS"
    )

    print(
        "Fare standard deviation approximately 1: PASS"
    )

    print(
        "Before/after comparison created: PASS"
    )

    print(
        "EDA-only standardization documented: PASS"
    )

    print("\n========================================")
    print("TASK 6 COMPLETED SUCCESSFULLY")
    print("========================================")

    # =====================================================
    # FINAL PART A SUMMARY
    # =====================================================

    print("\n========================================")
    print("PART A - TASKS 1 TO 6 COMPLETE")
    print("========================================")

    print(
        "Task 1 - Profiling: PASS"
    )

    print(
        "Task 2 - Missing-value handling: PASS"
    )

    print(
        "Task 3 - Univariate analysis: PASS"
    )

    print(
        "Task 4 - Bivariate analysis: PASS"
    )

    print(
        "Task 5 - Multivariate data story: PASS"
    )

    print(
        "Task 6 - Standardization check: PASS"
    )

    print(
        f"\nCleaned EDA rows: "
        f"{len(df)}"
    )

    print(
        f"\nCharts location:\n"
        f"{CHARTS_DIR}"
    )

    print(
        f"\nRaw offline fallback:\n"
        f"{TITANIC_FILE}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "titanic.csv remains the original raw "
        "Seaborn-loaded dataset."
    )

    print(
        "Part B modeling will read this same CSV "
        "without calling sns.load_dataset again."
    )


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()