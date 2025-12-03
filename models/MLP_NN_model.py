import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

# SCALING FOR NEURAL NETWORK
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(128,64), # two hidden layers
    activation="relu",
    solver="adam",
    alpha=0.001,    #L2 regularization
    batch_size=256,
    learning_rate="adaptive",
    max_iter=50,
    random_state=42
)
mlp.fit(X_train_scaled, y_train)

# EVALUATE THE MODEL
pred = mlp.predict(X_test_scaled)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, zero_division=0)
recall = recall_score(y_test, pred, zero_division=0)
f1 = f1_score(y_test, pred, zero_division=0)

print("Multi-Layer Perceptron Neural Network Performance Evaluation")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# LOSS CURVE
plt.figure(figsize=(8, 6))
plt.plot(mlp.loss_curve_)
plt.title("MLP w/o Tuning Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("MLP_Loss.pdf")