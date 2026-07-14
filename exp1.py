import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
housing = fetch_california_housing(as_frame=True)
data = housing.frame
X = data[['AveRooms']].values
y = data['MedHouseVal'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
theta_ne = np.linalg.inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train
y_pred_ne = X_test_b @ theta_ne
mse_ne = mean_squared_error(y_test, y_pred_ne)
r2_ne = r2_score(y_test, y_pred_ne)
m = 0.0
b = 0.0
alpha = 0.01
epochs = 1000
n = len(X_train)
X_train_flat = X_train.flatten()
X_test_flat = X_test.flatten()
for _ in range(epochs):
    y_pred = m * X_train_flat + b
    dm = (-2 / n) * np.sum(X_train_flat * (y_train - y_pred))
    db = (-2 / n) * np.sum(y_train - y_pred)
    m = m - alpha * dm
    b = b - alpha * db
y_pred_gd = m * X_test_flat + b
mse_gd = mean_squared_error(y_test, y_pred_gd)
r2_gd = r2_score(y_test, y_pred_gd)
sorted_idx = X_test[:, 0].argsort()
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color='blue', alpha=0.4, label='Actual Data')
plt.plot(
    X_test[sorted_idx],
    y_pred_ne[sorted_idx],
    color='red',
    linewidth=2,
    label='Normal Equation Line'
)
plt.title("Normal Equation Fit")
plt.xlabel("Average Number of Rooms (AveRooms)")
plt.ylabel("Median House Value")
plt.legend()
plt.grid(True)
plt.text(
    0.02, 0.95,
    f"MSE = {mse_ne:.3f}\nR² = {r2_ne:.3f}",
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
)
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='blue', alpha=0.4, label='Actual Data')
plt.plot(
    X_test[sorted_idx],
    y_pred_gd[sorted_idx],
    color='green',
    linewidth=2,
    label='Gradient Descent Line'
)
plt.title("Gradient Descent Fit")
plt.xlabel("Average Number of Rooms (AveRooms)")
plt.ylabel("Median House Value")
plt.legend()
plt.grid(True)
plt.text(
    0.02, 0.95,
    f"MSE = {mse_gd:.3f}\nR² = {r2_gd:.3f}",
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
)
plt.tight_layout()
plt.show()