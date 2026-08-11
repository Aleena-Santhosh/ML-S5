import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
mnist = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
X, y = mnist.data / 255.0, mnist.target.astype(int)
X_train, X_test = X[:10000], X[60000:62000]
y_train, y_test = y[:10000], y[60000:62000]
k_values = [1, 3, 5, 7, 9]
accuracies, times = [], []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    start = time.time()
    pred = model.predict(X_test)
    times.append(time.time() - start)
    accuracies.append(accuracy_score(y_test, pred) * 100)
    print(f"K={k}  Accuracy={accuracies[-1]:.2f}%  Time={times[-1]:.3f}s")
print("\nK Value | Accuracy | Time")
for k, a, t in zip(k_values, accuracies, times):
    print(f"{k:^7} | {a:.2f}%   | {t:.3f}s")
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(k_values, accuracies, "bo-")
plt.xlabel("K")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs K")
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(k_values, times, "rs-")
plt.xlabel("K")
plt.ylabel("Time (s)")
plt.title("Time vs K")
plt.grid()
plt.tight_layout()
plt.show()