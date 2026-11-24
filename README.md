
# Abalone Age Prediction with K-Nearest Neighbors

A supervised machine learning project that predicts **abalone shell ring counts** from physical measurements using **K-Nearest Neighbors (KNN) Regression**. Ring counts can be used to estimate age (approximately `rings + 1.5` years), but **all model evaluation metrics reported here are in rings**.

## Project Objective

Build, evaluate, and improve a reproducible regression pipeline, using cross-validation and an untouched final test set to assess performance on unseen observations.

## Dataset

- **Source:** [UCI Machine Learning Repository — Abalone](https://archive.ics.uci.edu/dataset/1/abalone)
- **Observations:** 4,177
- **Original input features:** 8 (`sex`, `length`, `diameter`, `height`, `whole_weight`, `shucked_weight`, `viscera_weight`, `shell_weight`)
- **Target:** `rings`
- **Split:** 3,341 training observations (80%) and 836 test observations (20%); `random_state=42`

The final model uses seven input features, excluding `viscera_weight` after training-only cross-validation experiments.

## Approach

1. Audited data types, missing values, duplicates, distributions, and potentially invalid measurements.
2. Performed exploratory analysis of numerical and categorical features and their relationships with ring counts.
3. Built a Scikit-learn `ColumnTransformer` and `Pipeline` to keep preprocessing within cross-validation folds.
4. Compared KNN with a mean-prediction `DummyRegressor` baseline using five-fold cross-validation.
5. Tuned KNN with `GridSearchCV` and compared scaling strategies and selected feature subsets using training data only.
6. Selected the model before evaluating it once on the held-out test set.

## Final Model

| Component | Configuration |
| --- | --- |
| Algorithm | `KNeighborsRegressor` |
| Number of neighbors | `21` |
| Neighbor weights | `distance` |
| Distance metric | `manhattan` |
| Categorical preprocessing | `OneHotEncoder` for `sex`, retaining all categories |
| Numerical preprocessing | `StandardScaler` for `length`, `diameter`, `whole_weight`, `shucked_weight`, `shell_weight` |
| Height preprocessing | `RobustScaler` for `height` |
| Excluded input | `viscera_weight` |

## Results

| Evaluation | Metric | Value |
| --- | --- | ---: |
| Baseline, five-fold CV | Mean validation RMSE | 3.2033 rings |
| Initial KNN, five-fold CV | Mean validation RMSE | 2.2959 rings |
| Selected KNN configuration, five-fold CV | Best mean validation RMSE | 2.1874 rings |
| **Final held-out test** | **R²** | **0.5353** |
| **Final held-out test** | **MAE** | **1.5398 rings** |
| **Final held-out test** | **MSE** | **5.0302 squared rings** |
| **Final held-out test** | **RMSE** | **2.2428 rings** |

The test RMSE is about 0.0554 rings higher than the selected cross-validation RMSE. The final model improved substantially on the simple mean-prediction baseline, while still having meaningful prediction error.

## Data Quality Observations and Limitations

Training-data inspection revealed two zero-height records, unusually high height values including `1.13`, and an observation whose recorded shell weight exceeded its whole weight.

These observations were flagged rather than silently corrected or removed, because verified replacement values were not available. Their possible effects on KNN distances remain a limitation.

The feature and hyperparameter experiments reused the development cross-validation folds; therefore the best CV result may be somewhat optimistic. The held-out test result is the final reported evaluation.

This model predicts **ring counts**, not a directly observed age label, and should not be treated as an exact biological age measurement.

## Reproduce the Project

1. Clone this repository and create a Python virtual environment.
2. Install the required packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `jupyter`.
3. Open `notebooks/01_knn_regression_abalone.ipynb` and run the notebook in order. The raw Abalone dataset should be located in `data/raw/` as expected by the notebook.
4. The completed notebook can save the fitted preprocessing-and-KNN pipeline to `models/abalone_knn_model.joblib`.

**Security note:** Scikit-learn/joblib files use pickle-based serialization. Load model files only from trusted sources. The fitted KNN artifact also contains training-derived observations.

## Potential Future Work

- Investigate and validate anomalous source measurements, then evaluate a documented cleaning strategy.
- Compare with additional regression algorithms and more robust validation designs.
- Create a simple prediction interface and add automated tests.

## Repository Structure

```text
abalone-age-prediction-knn/
├── data/
│   └── raw/
│       ├── abalone.data
│       └── abalone.names
├── notebooks/
│   └── 01_knn_regression_abalone.ipynb
├── models/
│   └── abalone_knn_model.joblib  # after saving the fitted model
├── .gitignore
└── README.md
```
