import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

X = np.load("X.npy")
y = np.load("y.npy")
feature_names = np.load("feature_names.npy", allow_pickle=True)

# TRAINING THE MODEL
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

# EVALUATE THE MODEL
pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, zero_division=0)
recall = recall_score(y_test, pred, zero_division=0)
f1 = f1_score(y_test, pred, zero_division=0)

print("Random Forest Performance Evaluation")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# FEATURE IMPORTANCE
importances = rf.feature_importances_
idx = np.argsort(importances)[::-1][:20]  # top 20 features
names = [feature_names[i] for i in idx]
vals = importances[idx]

# PLOT OF TOP 20 FEATURES
plt.figure(figsize=(10, 8))
plt.barh(names[::-1], vals[::-1])
plt.xlabel("Feature Importance")
plt.title("Top 20 Predictors of Depression (dx_dep)")
plt.tight_layout()
plt.savefig("rand_forest_preds.pdf")