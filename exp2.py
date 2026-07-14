import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
names = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Year", "Origin", "Name"]
df = pd.read_csv(url, names=names, na_values="?", sep=r"\s+").dropna()

X = df[["Displacement"]]
y = df["MPG"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

poly_transformer = PolynomialFeatures(degree=2)
X_train_poly = poly_transformer.fit_transform(X_train)
X_test_poly = poly_transformer.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

lin_mse = mean_squared_error(y_test, y_pred_lin)
lin_r2 = r2_score(y_test, y_pred_lin)

poly_mse = mean_squared_error(y_test, y_pred_poly)
poly_r2 = r2_score(y_test, y_pred_poly)

print("=" * 45)
print("Model Performance")
print("=" * 45)
print("Linear Regression")
print(f"MSE      : {lin_mse:.2f}")
print(f"R² Score : {lin_r2:.2f}")
print()
print("Polynomial Regression (Degree 2)")
print(f"MSE      : {poly_mse:.2f}")
print(f"R² Score : {poly_r2:.2f}")
print("=" * 45)

X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.scatter(X_test, y_test, color="gray", alpha=0.6, label="Test Data")
ax1.plot(X_line, lin_model.predict(X_line), color="red", linewidth=2, label="Linear Fit")
ax1.set_title("Linear Regression")
ax1.set_xlabel("Displacement")
ax1.set_ylabel("MPG")
ax1.legend()
ax1.grid(True)
ax1.text(0.02, 1.05, f"MSE = {lin_mse:.2f}\nR² = {lin_r2:.2f}", transform=ax1.transAxes, fontsize=10, bbox=dict(facecolor="white", edgecolor="black"))

ax2.scatter(X_test, y_test, color="gray", alpha=0.6, label="Test Data")
ax2.plot(X_line, poly_model.predict(poly_transformer.transform(X_line)), color="blue", linewidth=2, label="Polynomial Fit")
ax2.set_title("Polynomial Regression (Degree 2)")
ax2.set_xlabel("Displacement")
ax2.set_ylabel("MPG")
ax2.legend()
ax2.grid(True)
ax2.text(0.02, 1.05, f"MSE = {poly_mse:.2f}\nR² = {poly_r2:.2f}", transform=ax2.transAxes, fontsize=10, bbox=dict(facecolor="white", edgecolor="black"))

plt.tight_layout()
plt.show()