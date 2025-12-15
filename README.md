# Predicting Depression in College Students Using Machine Learning

## Overview
This project investigates the use of machine learning to predict depression among U.S. college students using large-scale survey data from the Healthy Minds Network. The primary aims are to identify which machine learning models most accurately classify depression and to determine which survey features most strongly contribute to prediction. We evaluate Random Forests, Logistic Regression, and Multi-Layer Perceptron (MLP) models on both one-year and three-year datasets exceeding 200,000 responses. Across all settings, neural network models, particularly the MLP with hyperparameter tuning, achieve the strongest performance, indicating that depression risk reflects nonlinear interactions among student characteristics. When mental-health-related predictors are included, prior diagnostic history dominates feature importance across models, producing very high predictive accuracy. However, a complementary analysis excluding all mental health variables shows that depression can still be moderately predicted using non-clinical factors such as social connection, academic engagement, lifestyle behaviors, and disabilities, though with reduced performance. Together, these findings demonstrate both the power and limitations of machine learning for mental health prediction, highlighting the critical role of clinical history while also revealing the potential for broader, non-clinical early screening approaches in college populations. 

## Replication Instructions
This section describes how to reproduce all preprocessing, modeling, and evaluation steps used in this project. The analysis was conducted in Python using standard machine learning libraries. 

### 1. Environment Setup
Ensure you are using Python 3.9+ and install the following packages:
* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib`

These packages can be installed via the following line in the terminal:
`pip install numpy pandas scikit-learn matplotlib`

_<ins> Note</ins>:_ All experiments were run with fixed random seeds (`random_state=42`) to ensure reproducibility.

### 2. Data Acquisition
Request access to the Healthy Minds Study (HMS) survey data from the Healthy Minds Network. Download the relevant CSV files (e.g., `HMS_2024-2025.csv` for the most recent year of survey responses) and place them in the project directory. To request HMN data, visit: https://healthymindsnetwork.org/research/data-for-researchers/ 

The target variable is `dx_dep`, a binary indicator of depression diagnosis.

### 3. Preprocessing
Run `preprocess.py` to generate the model-ready feature matrix.

This script performs the following steps:
1. Loads the HMS CSV file(s).
2. Optionally removes all mental-health-related variables using the predefined `FILTER_VARIABLES` list, while retaining `dx_dep` as the prediction target.
3. Converts object-type columns to numeric where possible.
4. Drops high-cardinality categorical variables (more than 15 unique values).
5. One-hot encodes remaining small categorical variables.
6. Combines numeric and encoded categorical features into a single feature matrix.
7. Replaces all missing values with zeros.
8. Saves the resulting arrays to disk:
   * `X.npy` - feature matrix
   * `y.npy` - target labels
   * `feature_names.npy` - feature names

To replicate the filtered (non-mental health) analysis, ensure the `preprocess_filtered()` function is applied before model training. To replicate the full-feature analysis, comment out or bypass the filtering step.

### 4. Train-Test Split
All models use the same data split:
* 80% training
* 20% testing
* Stratified by `dx_dep`

This split is performed internally within each model script using:
`train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)`

### 5. Model Training and Evaluation
#### Random Forest
Run `random_forests_model.py` to train a Random Forest classifier with:
* 300 trees (`n_estimators=300`)
* Maximum depth of 20 (`max_depth=20`)
* Balanced class weights (`class_weight="balanced"`)

The script reports accuracy, precision, recall, and F1-score and saves a PDF (`rand_forest_preds.pdf`) showing the top 20 most important features based on Gini importance.

#### Logistic Regression
Run `logistic_regression_model.py` to train a regularized Logistic Regression model. Steps include:
* Standardizing features using `StandardScaler`
* Training with L2 regularization (`penalty="l2"`) and balanced class weights (`class_weight="balanced"`)

The script outputs the same performance metrics as Random Forest and saves a coefficient-based feature importance plot (`log_reg_preds.pdf`) for the top 20 predictors.

#### Baseline Multi-Layer Perceptron (No Hyperparameter Tuning)
Run `MLP_NN_model.py` to train a neural network with:
* Two hidden layers of sizes 128 and 64 (`hidden_layer_sizes=(128,64)`)
* ReLU activation function (`activation="relu"`)
* Adam optimizer (`solver="adam"`)
* Batch size of 256 (`batch_size=256`)
* Adaptive learning rate (`learning_rate="adaptive"`)
* 50 training iterations (`max_iter=50`)

Like in Logistic Regression, features are standardized prior to training. The same performance metrics are reported, and the training loss curve is saved as `MLP_Loss.pdf`.

#### Tuned Multi-Layer Perceptron (With Hyperparameter Training)
Run `MLP_NN_HT_model.py` to perform hyperparameter tuning using `GridSearchCV`. The grid search explores:
* Network depth and width (`'hidden_layer_sizes': [ (64,), (64, 32), (128, 64, 32)]`)
* Activation functions (`'activation': ['relu', 'tanh']`)
* Regularization strength (`'alpha': [0.0001, 0.001, 0.01]`)
* Learning rates (`'learning_rate_init': [0.001, 0.01]`)
* Training iterations (`'max_iter': [200, 400]`)

Five-fold cross-validation is used with F1-score as the optimization metric. The best-performing model is evaluated on the test set, and the training loss curve is saved as `MLP_HT_Loss.pdf`.

### 6. Reproducing Main Results
To reproduce the primary findings:
1. Run `preprocess.py` to generate `X.npy` and `y.npy`.
2. Execute each model script independently.
3. Record reported performance metrics and generated plots.
4. Repeat preprocessing with and without mental health predictors to replicate the comparative analyses.

Minor numerical variation may occur for neural network models due to the optimization dynamics, but overall performance patterns and feature importance rankings should remain consistent.

## Future Directions
While the Multi-Layer Perceptron achieved the strongest predictive performance, its limited interpretability motivates future work focused on model transparency. A natural next step is to apply SHAP (SHapley Additive exPlanations) to quantify the contribution of individual features to each prediction, enabling more interpretable explanations and facilitating direct comparison with feature importance measures from Random Forests and coefficients from Logistic Regression. In addition, future research could examine temporal dynamics in depression risk by leveraging longitudinal data from the same students across multiple years. Such an analysis would allow investigation of how risk factors evolve over time, whether the relative importance of predictors shifts and whether depression becomes easier or more difficult to predict as students progress through college. Together, these extensions would strengthen both the interpretability and the real-world applicability of machine learning-based depression risk models. 

## Contributions
This project was completed individually. I was responsible for all stages of the work, including data acquisition and preprocessing, model design and implementation, experimental evaluation, and interpretation of results. Specifically, I implemented the preprocessing pipeline, constructed and trained all machine learning models, and conducted comparative analyses across datasets and feature sets. I also performed feature importance analyses, generated all visualizations, and synthesized the findings into the final artifact and poster. The project required approximately 35 total hours, distributed across data cleaning and preprocessing, model development and tuning, model runs in both VSCode and JupyterLab, result analysis, and final documentation. 
