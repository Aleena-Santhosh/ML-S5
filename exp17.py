import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Load Digits Dataset
digits = load_digits()

X = digits.data

print("Dataset shape:", X.shape)
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])


# Standardize the data
X_scaled = StandardScaler().fit_transform(X)


# Values of K to test
k_values = [2, 3, 4, 5, 6, 8, 10, 12]

inertia = []
silhouette = []


# K-Means for different K values
for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertia.append(model.inertia_)

    silhouette.append(
        silhouette_score(X_scaled, labels)
    )


# Create results table
results = pd.DataFrame({
    "K": k_values,
    "Inertia": inertia,
    "Silhouette Score": silhouette
})


print("\nPerformance Comparison:")
print(results.to_string(index=False))


# Find best K using Silhouette Score
best = silhouette.index(max(silhouette))

print("\nBest K:", k_values[best])
print("Best Silhouette Score:", silhouette[best])


# ------------------------------------------------
# Plot graphs
# ------------------------------------------------

plt.figure(figsize=(12, 4))


# Sample digit
plt.subplot(1, 3, 1)

plt.imshow(
    digits.images[0],
    cmap="gray"
)

plt.title("Sample Digit")
plt.axis("off")


# K vs Inertia
plt.subplot(1, 3, 2)

plt.plot(
    k_values,
    inertia,
    marker="o"
)

plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("K vs Inertia")
plt.grid()


# K vs Silhouette Score
plt.subplot(1, 3, 3)

plt.plot(
    k_values,
    silhouette,
    marker="o"
)

plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.title("K vs Silhouette")
plt.grid()


plt.tight_layout()
plt.show()
