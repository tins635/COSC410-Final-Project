import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
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

# final feature matrix - combining numeric and encoded categorical features
X = np.hstack([X_num, X_cat])
feature_names = numeric_cols + list(cat_feature_names)

# replace all NaN with 0
X = np.nan_to_num(X, nan=0.0)

# TRAINING THE MODEL
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# SCALING FOR LOGISTIC REGRESSION
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(
    penalty="l2",
    C=1.0,
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs"
)
log_reg.fit(X_train_scaled, y_train)

# EVALUATE THE MODEL
pred = log_reg.predict(X_test_scaled)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, zero_division=0)
recall = recall_score(y_test, pred, zero_division=0)
f1 = f1_score(y_test, pred, zero_division=0)

print("Logistic Regression Performance Evaluation")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# FEATURE IMPORTANCE
coefs = log_reg.coef_[0]
idx = np.argsort(coefs)[::-1][:20]  # top 20 features
names = [feature_names[i] for i in idx]
vals = coefs[idx]

# PLOT OF TOP 20 FEATURES
plt.figure(figsize=(10, 8))
plt.barh(names[::-1], vals[::-1])
plt.xlabel("Coefficient Value")
plt.title("Top 20 Predictors of Depression (dx_dep)")
plt.tight_layout()
plt.show()