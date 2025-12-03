import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

X = np.load("X.npy")
y = np.load("y.npy")
feature_names = np.load("feature_names.npy", allow_pickle=True)

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
plt.savefig("log_reg_preds.pdf")