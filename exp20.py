import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


# --------------------------------------------------
# Step 1 & 2: Load Boston Housing Dataset
# --------------------------------------------------

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"

try:
    df = pd.read_csv(url)
    print("Dataset loaded successfully from URL.")

except Exception as e:
    print(f"URL load failed ({e}). Generating fallback dataset...")

    np.random.seed(42)

    rm = np.random.uniform(3.5, 9.0, 506)

    medv = 5 * rm + np.random.normal(0, 5, 506)

    df = pd.DataFrame({
        "rm": rm,
        "medv": medv
    })


print("\nDataset Head:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())


# --------------------------------------------------
# Step 3: Select Input Feature and Target
# --------------------------------------------------

X = df[["rm"]]
y = df["medv"]

print("\nInput Feature Head:")
print(X.head())

print("\nTarget Head:")
print(y.head())


# --------------------------------------------------
# Step 4: Split the Dataset
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")


# --------------------------------------------------
# Step 5: Define Polynomial Degrees
# --------------------------------------------------

degrees = [1, 2, 3, 4, 5, 7, 10, 15]

train_errors = []
validation_errors = []


# --------------------------------------------------
# Step 6: Train Polynomial Regression Models
# --------------------------------------------------

for degree in degrees:

    model = Pipeline([
        ("scaler", StandardScaler()),

        ("poly", PolynomialFeatures(
            degree=degree,
            include_bias=False
        )),

        ("regressor", LinearRegression())
    ])

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    # Calculate MSE
    train_mse = mean_squared_error(
        y_train,
        train_pred
    )

    val_mse = mean_squared_error(
        y_val,
        val_pred
    )

    # Store errors
    train_errors.append(train_mse)
    validation_errors.append(val_mse)


# --------------------------------------------------
# Step 7: Create Comparison Table
# --------------------------------------------------

results = pd.DataFrame({
    "Polynomial Degree": degrees,
    "Training MSE": train_errors,
    "Validation MSE": validation_errors
})

print("\nPerformance Comparison Table:")

print(
    results.to_string(
        index=False,
        float_format="%.4f"
    )
)


# --------------------------------------------------
# Step 8: Plot Training and Validation Errors
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    degrees,
    train_errors,
    marker="o",
    label="Training Error"
)

plt.plot(
    degrees,
    validation_errors,
    marker="o",
    label="Validation Error"
)

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")

plt.title("Bias-Variance Tradeoff")

plt.legend()

plt.grid(True)

plt.show()


# --------------------------------------------------
# Step 9: Find Best Polynomial Degree
# --------------------------------------------------

best_index = np.argmin(validation_errors)

best_degree = degrees[best_index]

best_validation_error = validation_errors[best_index]

print(
    f"\nBest polynomial degree: {best_degree}"
)

print(
    f"Minimum validation MSE: "
    f"{best_validation_error:.4f}"
)
