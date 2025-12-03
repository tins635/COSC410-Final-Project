import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

X = np.load("X.npy")
y = np.load("y.npy")
feature_names = np.load("feature_names.npy", allow_pickle=True)

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

# LOSS CURVE
plt.figure(figsize=(8, 6))
plt.plot(best_mlp.loss_curve_)
plt.title("MLP w/ Tuning Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("MLP_HT_Loss.pdf")