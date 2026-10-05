import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
data = pd.read_csv("iris.csv")
X = data[
    ["sepal_length", "sepal_width", "petal_length", "petal_width"]
].values
y = data["species"].map({
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2
}).values
print("Dataset shape:", X.shape)
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

print("\nTarget classes:")
print(["setosa", "versicolor", "virginica"])
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("\nModel pipeline:")
print(model)


# --------------------------------------------------
# Step 4: Implement Bootstrapping
# --------------------------------------------------

rng = np.random.default_rng(42)

n_iterations = 30
n_samples = len(X)

bootstrap_results = []

for i in range(n_iterations):

    # Generate bootstrap sample
    train_indices = rng.choice(
        n_samples,
        size=n_samples,
        replace=True
    )

    # Find Out-of-Bag samples
    selected = np.zeros(n_samples, dtype=bool)
    selected[train_indices] = True

    test_indices = np.where(~selected)[0]

    # Train and evaluate if OOB samples exist
    if len(test_indices) > 0:

        model.fit(X[train_indices], y[train_indices])

        y_pred = model.predict(X[test_indices])

        # Accuracy
        accuracy = accuracy_score(
            y[test_indices],
            y_pred
        )

        # Macro F1-score
        f1 = f1_score(
            y[test_indices],
            y_pred,
            average="macro"
        )

        bootstrap_results.append([
            i + 1,
            accuracy,
            f1,
            len(test_indices)
        ])


# Convert results into DataFrame
bootstrap_results = pd.DataFrame(
    bootstrap_results,
    columns=[
        "Iteration",
        "Accuracy",
        "F1-Score",
        "OOB Samples"
    ]
)


# --------------------------------------------------
# Step 5: Display First 10 Bootstrap Results
# --------------------------------------------------

print("\nBootstrap Results (First 10 iterations):")

print(
    bootstrap_results.head(10).to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "F1-Score": "{:.4f}".format
        }
    )
)


# --------------------------------------------------
# Step 6: Calculate Bootstrap Mean
# --------------------------------------------------

bootstrap_accuracy = bootstrap_results["Accuracy"].mean()
bootstrap_f1 = bootstrap_results["F1-Score"].mean()

print("\nBootstrap Mean Accuracy:",
      f"{bootstrap_accuracy:.4f}")

print("Bootstrap Mean F1-Score:",
      f"{bootstrap_f1:.4f}")


# --------------------------------------------------
# Step 7: Implement 5-Fold Cross-Validation
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=["accuracy", "f1_macro"]
)


# --------------------------------------------------
# Step 8: Display Cross-Validation Results
# --------------------------------------------------

print("\nAccuracy for each fold:")

print(
    np.round(
        cv_results["test_accuracy"],
        4
    )
)

print("\nF1-Score for each fold:")

print(
    np.round(
        cv_results["test_f1_macro"],
        4
    )
)


# --------------------------------------------------
# Step 9: Calculate Cross-Validation Mean
# --------------------------------------------------

cv_accuracy = cv_results["test_accuracy"].mean()
cv_f1 = cv_results["test_f1_macro"].mean()

print("\nCross-Validation Mean Accuracy:",
      f"{cv_accuracy:.4f}")

print("Cross-Validation Mean F1-Score:",
      f"{cv_f1:.4f}")


# --------------------------------------------------
# Step 10: Compare Both Methods
# --------------------------------------------------

comparison = pd.DataFrame({
    "Method": [
        "Bootstrapping",
        "5-Fold Cross-Validation"
    ],

    "Mean Accuracy": [
        bootstrap_accuracy,
        cv_accuracy
    ],

    "Mean F1-Score": [
        bootstrap_f1,
        cv_f1
    ]
})


# --------------------------------------------------
# Step 11: Display Final Comparison
# --------------------------------------------------

print("\nPerformance Comparison:")

print(
    comparison.to_string(
        index=False,
        formatters={
            "Mean Accuracy": "{:.4f}".format,
            "Mean F1-Score": "{:.4f}".format
        }
    )
)
