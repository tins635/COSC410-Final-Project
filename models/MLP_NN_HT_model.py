import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings("ignore")

# LOADING THE DATA
df = pd.read_csv("data/HMS_2024-2025.csv", low_memory=False)

df["dx_dep"] = df["dx_dep"].fillna(0).astype(int)
y = df["dx_dep"].to_numpy()
Xdf = df.drop(columns=["dx_dep"])

# PREPROCESSING THE DATA

# convert mixed-type columns (i.e., converting "object" to numeric when possible)
for col in Xdf.columns:
    if Xdf[col].dtype == "object":
        try:
            Xdf[col] = pd.to_numeric(Xdf[col])
        except:
            pass

numeric_cols = Xdf.select_dtypes(include=["number"]).columns.tolist()
object_cols = Xdf.select_dtypes(include=["object"]).columns.tolist()

# only encode small categorical columns and drop high-cardinality object columns
small_cat_cols = [c for c in object_cols if Xdf[c].nunique() <= 15]
large_cat_cols = [c for c in object_cols if c not in small_cat_cols]
Xdf = Xdf.drop(columns=large_cat_cols)

# one-hot encode small categorical columns
if len(small_cat_cols) > 0:
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    X_cat = ohe.fit_transform(Xdf[small_cat_cols])
    cat_feature_names = ohe.get_feature_names_out(small_cat_cols)
else:
    X_cat = np.empty((len(Xdf), 0))
    cat_feature_names = []

X_num = Xdf[numeric_cols].to_numpy()

# final feature matrix
X = np.hstack([X_num, X_cat])

# replace all NaN with 0
X = np.nan_to_num(X, nan=0.0)

# TRAINING THE MODEL
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# HYPERPARAMETER GRID
param_grid = {
    'hidden_layer_sizes': [
        (64,),            # 1 layer
        (64, 32),         # 2 layers
        (128, 64, 32),    # 3 layers
    ],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],  # L2 regularization
    'learning_rate_init': [0.001, 0.01],
    'max_iter': [200, 400]
}

# GRID SEARCH 
base_mlp = MLPClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=base_mlp,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train, y_train)
print("\nBest Hyperparameters Found:")
print(grid_search.best_params_)

# EVALUATE THE BEST MODEL
best_mlp = grid_search.best_estimator_
pred = best_mlp.predict(X_test)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, zero_division=0)
recall = recall_score(y_test, pred, zero_division=0)
f1 = f1_score(y_test, pred, zero_division=0)

print("Tuned MLP Neural Network Performance Evaluation")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)