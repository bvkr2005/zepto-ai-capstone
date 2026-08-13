# Module 2 — Analytics Pipeline

## Overview

This module implements a complete data analytics and machine-learning pipeline using the Titanic dataset.

The dataset is loaded once using Seaborn's built-in Titanic dataset loader. An offline copy is immediately saved as `titanic.csv`. All later modeling work uses this saved dataset instead of loading the Titanic dataset again.

The module covers data profiling, data cleaning, exploratory data analysis, visualization, preprocessing, classification, class-imbalance handling, hyperparameter tuning, regression, model comparison, and model pipeline serialization.


## Project Files

The main files for this module are:

- `01_eda.py` — Tasks 1 through 6
- `02_modeling.py` — Tasks 7 through 15
- `titanic.csv` — offline copy of the Titanic dataset
- `best_titanic_pipeline.joblib` — saved complete machine-learning pipeline
- `charts/` — generated charts and visualizations
- CSV and TXT files — generated results and written conclusions


# Part A — Profiling, Cleaning, and Data Story


## Task 1 — Load and Profile the Dataset

The Titanic dataset is loaded once using:

```python
sns.load_dataset("titanic")

This allows the project to run even if internet access is unavailable later.

The following profiling operations are performed:

df.info()
df.describe()
df.shape
Missing-value percentage calculation
Missing Values Found

The original Titanic dataset contained the following missing-value percentages:

Column	Missing Percentage
deck	77.216611%
age	19.865320%
embarked	0.224467%
embark_town	0.224467%
Task 2 — Missing-Value Handling

The assignment requires the following threshold rule:

Less than 5% missing → drop affected rows
Between 5% and 30% missing → impute
Very high missing percentage → drop the column or represent missing as a separate category

The strategies used were:

age

age has approximately 19.87% missing values.

Because this falls between 5% and 30%, missing ages are handled using imputation where required.

embarked

embarked has approximately 0.22% missing values.

Because this is below 5%, the small number of affected rows can be dropped during the EDA cleaning stage.

embark_town

embark_town also has approximately 0.22% missing values.

Because this is below 5%, the small number of affected rows can be dropped during the EDA cleaning stage.

deck

deck has approximately 77.22% missing values.

This level of missingness is too high for reliable imputation. Therefore, the deck column is dropped from the cleaned analytical dataset.

Task 3 — Univariate Analysis

Univariate analysis is performed on:

age
fare

For both variables, histogram and box-plot visualizations are generated.

Outliers are identified using the IQR rule:

Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR

Values outside these boundaries are considered outliers.

For fare, the mean, median, and mode are also calculated.

The relationship between the mean, median, and mode is used to determine whether the fare distribution is right-skewed, left-skewed, or approximately symmetric.

The exact calculated values and outlier counts are printed by 01_eda.py.

Task 4 — Bivariate Analysis

Survival rates are analyzed by:

Sex
Passenger class
Sex and passenger class together

Boolean masking using & and/or | is used for these calculations.

A correlation matrix is created using exactly these six numeric columns:

survived
pclass
age
sibsp
parch
fare

The derived boolean columns adult_male and alone are intentionally excluded.

The correlation matrix is displayed using a Seaborn heatmap.

The two strongest off-diagonal correlations are determined by ranking feature pairs according to the absolute value of their correlation coefficients.

Task 5 — Multivariate Data Story

At least four different visualizations are generated to investigate which passenger characteristics are associated with survival.

The visualizations examine relationships involving characteristics such as:

Sex
Passenger class
Age
Fare
Survival

Each chart is accompanied by a written interpretation explaining the pattern shown by the visualization.

Together, the charts provide a data story about the characteristics associated with higher or lower survival rates.

Task 6 — Standardization Check

As an exploratory EDA check, age and fare are standardized using the z-score concept:

z = (x - mean) / standard deviation

The before-and-after statistics are compared.

After standardization, both transformed variables should have approximately:

Mean = 0
Standard deviation = 1

This Task 6 standardization is only an exploratory check.

It is not used for model training. The modeling pipeline performs its own scaling using training data only.

Part B — Predictive Modeling
Task 7 — Stratified Train/Test Split

The classification target is:

survived

The following features are used:

pclass
sex
age
sibsp
parch
fare
embarked

A stratified train/test split is performed.

Stratification is important because the survived and non-survived classes are not perfectly balanced.

Using stratification keeps approximately the same class distribution in both the training and test sets.

Task 8 — Preprocessing

Preprocessing is performed using a ColumnTransformer and scikit-learn pipelines.

Numeric Features

The numeric features are:

pclass
age
sibsp
parch
fare

Numeric missing values are handled with median imputation.

Numeric variables are scaled using:

StandardScaler

Categorical Features

The categorical features are:

sex
embarked

Categorical missing values are handled using most-frequent imputation.

Categorical variables are encoded using:

OneHotEncoder

Data Leakage Prevention

All preprocessing steps are fitted only on the training data.

The fitted preprocessing transformations are then applied to the test data in transform-only mode.

This prevents information from the test dataset from leaking into model training.

Task 9 — Classification Models

Three classification algorithms are trained using the same train/test split:

Logistic Regression
Decision Tree
Random Forest

The Decision Tree is also visualized using plot_tree() with feature names and class labels.

Task 10 — Classification Evaluation

All three classifiers are evaluated using:

Confusion Matrix
Accuracy
Precision
Recall
F1 Score
ROC Curve
AUC

The results are stored in a model-comparison table.

ROC curves are generated to compare the ability of the classifiers to distinguish between survivors and non-survivors.

Task 11 — Class Imbalance Handling

The survived/not-survived class distribution is examined.

Three approaches are compared:

Baseline — no special imbalance handling
class_weight="balanced"
SMOTE

The approaches are compared using:

Precision
Recall
F1 Score

SMOTE is applied only to the training data.

The test data is never oversampled because doing so would cause data leakage.

The strategy producing the strongest F1 result is reported by the modeling script.

Task 12 — Hyperparameter Tuning

Random Forest hyperparameters are tuned using:

GridSearchCV

The following parameters are searched:

n_estimators
max_depth
max_features

The Random Forest classifier is created with:

RandomForestClassifier(
    oob_score=True,
    random_state=42
)

Setting oob_score=True allows the fitted model's out-of-bag score to be reported.

The program reports:

Best parameter combination
Best cross-validation F1 score
OOB score
Test-set metrics
Task 13 — Regression

A multivariate Linear Regression model is created to predict:

fare

The regression model uses other available passenger characteristics as predictors.

The model is evaluated using:

MAE
RMSE
R²
Adjusted R²

A residual plot is also generated.

The residual pattern is examined to determine whether there is evidence of heteroscedasticity.

Task 14 — Final Model Comparison

The three classification models are compared using:

Accuracy
Precision
Recall
F1
AUC

The regression model is evaluated separately using:

MAE
RMSE
R²
Adjusted R²

Classification and regression metrics are intentionally presented as separate metric groups because they measure different types of prediction problems and are not directly comparable.

The final classifier recommendation is based on the actual classification results generated by the modeling script.

Task 15 — Save and Reload the Complete Pipeline

The best-performing complete classification pipeline is saved using Joblib.

The output file is:

best_titanic_pipeline.joblib

The saved object contains both:

Preprocessing steps
Final classifier

This is important because saving only the classifier would require new data to be manually preprocessed before prediction.

The saved pipeline is reloaded using:

joblib.load(...)

After reloading, raw passenger data is passed directly into the pipeline.

The test raw data includes numeric and categorical features and even a missing age value.

Successful prediction confirms that the saved artifact performs:

Missing-value imputation
Categorical encoding
Numeric scaling
Classification

as one complete end-to-end pipeline.

Generated Output Files

The module generates files including:

titanic.csv
task7_split_summary.csv
task8_preprocessing_summary.csv
task10_model_comparison.csv
task11_imbalance_comparison.csv
task11_imbalance_conclusion.txt
task12_random_forest_tuning.csv
task13_regression_metrics.csv
task13_regression_conclusion.txt
task14_final_model_comparison.csv
task14_final_recommendation.txt
task15_reload_prediction.csv
best_titanic_pipeline.joblib

Generated charts are stored in the charts directory.

How to Run

The scripts should be run from the project root directory.

First run the EDA pipeline:

python analytics/01_eda.py

After Tasks 1–6 complete successfully, run the modeling pipeline:

python analytics/02_modeling.py

The modeling pipeline performs Tasks 7–15 and creates the model, evaluation files, charts, and saved pipeline.

Design Decisions

The main design decisions used in this project are:

The Titanic dataset is loaded only once and saved as titanic.csv for offline reuse.
Missing values are handled according to the assignment's percentage thresholds during EDA.
Train/test splitting is performed before modeling preprocessing.
Modeling preprocessing is fitted only on training data to prevent leakage.
Numeric and categorical variables use separate preprocessing pipelines.
SMOTE is applied only to training data.
Random Forest hyperparameters are tuned using cross-validation.
Classification and regression results are evaluated separately.
The complete preprocessing and classifier pipeline is saved rather than saving only the final estimator.
Conclusion

This module demonstrates an end-to-end analytics and machine-learning workflow using the Titanic dataset.

The workflow begins with profiling and cleaning, continues through exploratory and multivariate analysis, and then builds classification and regression models. The classification models are evaluated using multiple performance metrics, class imbalance is investigated, and Random Forest hyperparameters are tuned using GridSearchCV.

Finally, the best complete classification pipeline is saved and successfully reloaded for prediction on raw passenger data.

# Actual Model Results

The following results were generated by running `02_modeling.py` on the committed Titanic dataset.


## Task 10 — Classification Model Results

The three classification models were evaluated on the same test set.

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8437 |
| Decision Tree | 0.7933 | 0.8636 | 0.5507 | 0.6726 | 0.8292 |
| Random Forest | 0.8156 | 0.8000 | 0.6957 | 0.7442 | 0.8300 |

Random Forest achieved the highest accuracy and F1 score among the three baseline classifiers. Logistic Regression achieved the highest ROC-AUC score of 0.8437. The Decision Tree achieved the highest precision, but its lower recall resulted in the lowest F1 score of the three models.


## Task 11 — Class Imbalance Results

The target class distribution was:

- Non-survivors: 61.62%
- Survivors: 38.38%

Three imbalance-handling strategies were compared using Logistic Regression.

| Strategy | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Baseline | 0.7931 | 0.6667 | 0.7244 |
| Class Weight Balanced | 0.7297 | 0.7826 | 0.7552 |
| SMOTE | 0.7397 | 0.7826 | 0.7606 |

SMOTE produced the highest F1 score of 0.7606, with precision of 0.7397 and recall of 0.7826. Therefore, SMOTE provided the best precision-recall balance among the three tested imbalance strategies. SMOTE was applied only to the training data so that information from the test set did not leak into model training.


## Task 12 — Random Forest Hyperparameter Tuning

Random Forest was tuned using `GridSearchCV`.

The best hyperparameter combination was:

- `n_estimators = 100`
- `max_depth = 5`
- `max_features = "sqrt"`

The best cross-validation F1 score was:

`0.7459`

The out-of-bag score was:

`0.8272`

The tuned Random Forest produced the following test results:

| Metric | Result |
|---|---:|
| Accuracy | 0.8156 |
| Precision | 0.8750 |
| Recall | 0.6087 |
| F1 Score | 0.7179 |
| AUC | 0.8431 |

The tuned Random Forest achieved strong precision of 0.8750 and an AUC of 0.8431. However, its recall of 0.6087 was lower than the baseline Random Forest recall.


## Task 13 — Fare Regression Results

The multivariate Linear Regression model produced the following results:

| Metric | Result |
|---|---:|
| MAE | 20.8094 |
| RMSE | 30.4731 |
| R² | 0.3999 |
| Adjusted R² | 0.3679 |

The R² value of 0.3999 indicates that the regression model explains approximately 40% of the variation in passenger fare in the test data.

The residual plot shows evidence of heteroscedasticity because the residual spread changes across predicted fare values. Therefore, the constant-variance assumption of ordinary linear regression is not fully satisfied for this model.


## Task 14 — Final Classification Comparison and Recommendation

The three baseline classifiers produced different strengths.

Random Forest achieved the highest baseline accuracy of 0.8156 and the highest baseline F1 score of 0.7442. Logistic Regression achieved the highest ROC-AUC of 0.8437, compared with 0.8292 for Decision Tree and 0.8300 for Random Forest.

For the deployment recommendation used in this project, Logistic Regression was selected because the model-selection logic prioritizes ROC-AUC and then F1 score. Logistic Regression achieved accuracy of 0.8045, precision of 0.7931, recall of 0.6667, F1 score of 0.7244, and the highest baseline ROC-AUC of 0.8437.

Classification metrics are considered separately from the fare-regression metrics because classification and regression solve different prediction problems and their metric values are not directly comparable.


## Task 15 — Saved Complete Pipeline

The best deployment candidate is saved as a complete fitted machine-learning pipeline using Joblib.

The saved file is:

`best_titanic_pipeline.joblib`

The saved object contains both preprocessing and classification steps. It can therefore accept raw passenger data and internally perform missing-value imputation, categorical encoding, numeric scaling, and classification.

The pipeline was successfully reloaded using `joblib.load()` and tested on raw passenger records. The successful predictions confirmed that the saved artifact works end-to-end without requiring manual preprocessing.

# Actual EDA Results

## Task 1 — Dataset Profile

The original Titanic dataset contained:

- 891 rows
- 15 columns

The survival target was moderately imbalanced:

- Non-survivors: 549 (61.62%)
- Survivors: 342 (38.38%)

The following missing-value percentages were measured:

| Column | Missing Percentage |
|---|---:|
| deck | 77.216611% |
| age | 19.865320% |
| embarked | 0.224467% |
| embark_town | 0.224467% |

The raw dataset was saved as `titanic.csv` immediately after loading so that the same dataset could be reused by the modeling stage without another `sns.load_dataset()` call.


## Task 2 — Cleaning Results

The percentage-based missing-value rule was applied.

`deck` had 77.216611% missing values. Because the missing percentage was extremely high, the column was dropped rather than imputed.

`age` had 19.865320% missing values. Because this falls between 5% and 30%, missing ages were imputed using the median age of 28.00.

`embarked` had 0.224467% missing values. Because this is below 5%, the affected rows were dropped.

`embark_town` also had 0.224467% missing values. Because this is below 5%, the affected rows were dropped.

Cleaning results:

- Rows before cleaning: 891
- Rows after cleaning: 889
- Rows removed: 2
- Remaining missing values: 0


## Task 3 — Univariate Analysis Results

### Age Outliers

The IQR analysis for `age` produced:

- Q1 = 22.00
- Q3 = 35.00
- IQR = 13.00
- Lower bound = 2.50
- Upper bound = 54.50
- Number of age outliers = 65

Therefore, 65 age observations fall outside the IQR outlier boundaries.


### Fare Outliers

The IQR analysis for `fare` produced:

- Q1 = 7.90
- Q3 = 31.00
- IQR = 23.10
- Lower bound = -26.76
- Upper bound = 65.66
- Number of fare outliers = 114

Therefore, 114 fare observations fall outside the IQR outlier boundaries.


### Fare Distribution

The central-tendency statistics for fare were:

- Mean = 32.0967
- Median = 14.4542
- Mode = 8.0500

The ordering is:

`Mean > Median > Mode`

Therefore, the fare distribution is right-skewed. The relatively high mean is influenced by passengers who paid substantially higher fares.


## Task 4 — Bivariate Analysis Results

### Survival Rate by Sex

- Female: 74.04%
- Male: 18.89%

Female passengers had a substantially higher survival rate than male passengers.


### Survival Rate by Passenger Class

- First class: 62.62%
- Second class: 47.28%
- Third class: 24.24%

Survival decreased as passenger class moved from first to third.


### Survival Rate by Sex and Passenger Class

| Sex | Passenger Class | Survival Rate |
|---|---:|---:|
| Female | 1 | 96.74% |
| Female | 2 | 92.11% |
| Female | 3 | 50.00% |
| Male | 1 | 36.89% |
| Male | 2 | 15.74% |
| Male | 3 | 13.54% |

These results show that both sex and passenger class were strongly associated with survival. Female passengers had much higher survival rates within every passenger class, while first-class passengers generally had better outcomes than lower-class passengers.


### Correlation Analysis

The correlation matrix was restricted to exactly:

- `survived`
- `pclass`
- `age`
- `sibsp`
- `parch`
- `fare`

The derived boolean variables `adult_male` and `alone` were excluded as required.

The two strongest absolute off-diagonal correlations were:

1. `pclass` vs `fare`: -0.5482
2. `sibsp` vs `parch`: +0.4145

The negative correlation between `pclass` and `fare` indicates that numerically lower passenger classes, particularly first class, were associated with higher fares.

The positive correlation between `sibsp` and `parch` indicates that passengers traveling with siblings or spouses also had some tendency to travel with parents or children.


## Task 5 — Multivariate Data Story

### Chart 1 — Survival Rate by Sex

Female passengers had a survival rate of 74.04%, compared with 18.89% for male passengers. The large difference indicates that sex was strongly associated with survival. Female passengers were substantially more likely to survive.


### Chart 2 — Survival Rate by Passenger Class

First-class passengers had a survival rate of 62.62%, compared with 47.28% in second class and 24.24% in third class. Survival declined as passenger class moved from first to third. This suggests socioeconomic position was associated with survival opportunities.


### Chart 3 — Survival Rate by Sex and Passenger Class

Survival depended on both sex and passenger class. First-class females had a survival rate of 96.74%, while third-class females had 50.00%.

First-class males had a survival rate of 36.89%, compared with only 13.54% for third-class males. The pattern suggests that being female provided a major survival advantage, while higher passenger class added another advantage.


### Chart 4 — Age and Fare by Survival Status

The age-versus-fare scatter plot shows that survival was not determined by age alone because both survivors and non-survivors appear across a wide range of ages.

Survivors had a median fare of 26.00, compared with 10.50 for non-survivors. This supports the passenger-class analysis by showing that passengers paying higher fares were more strongly represented among survivors.


## Task 6 — Exploratory Standardization Results

Before standardization:

| Feature | Mean | Standard Deviation |
|---|---:|---:|
| age | 29.315152 | 12.977627 |
| fare | 32.096681 | 49.669545 |

After z-score standardization:

| Feature | Mean | Standard Deviation |
|---|---:|---:|
| age | approximately 0 | approximately 1 |
| fare | approximately 0 | approximately 1 |

After transformation, both `age` and `fare` had means approximately equal to 0 and standard deviations approximately equal to 1.

This confirms that the z-score standardization was applied correctly.

This standardization was performed only as an EDA-stage sanity check. It was not passed into the modeling stage. The modeling pipeline fits its own `StandardScaler` using training data only, which prevents test-set data leakage.