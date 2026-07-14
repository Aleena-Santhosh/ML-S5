import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
alphas = {'alpha': [0.01, 0.1, 1, 10, 100]}
lr = LinearRegression().fit(X_train, y_train)
ridge = GridSearchCV(Ridge(), alphas, cv=5).fit(X_train, y_train)
lasso = GridSearchCV(Lasso(), alphas, cv=5).fit(X_train, y_train)
models = {
    "Linear Regression": lr,
    f"Ridge (α={ridge.best_params_['alpha']})": ridge.best_estimator_,
    f"Lasso (α={lasso.best_params_['alpha']})": lasso.best_estimator_
}
results = []
for name, model in models.items():
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results.append({
        "Model": name,
        "MSE": mse,
        "R2 Score": r2
    })
df = pd.DataFrame(results)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#ff6666', '#66b2ff', '#99ff99']
df_mse = df.sort_values(by="MSE", ascending=True)
bars1 = ax1.bar(df_mse["Model"], df_mse["MSE"], color=colors, alpha=0.85, edgecolor='black', width=0.5)
ax1.set_title("Mean Squared Error (MSE)\n[Lower is Better]", fontsize=12, fontweight='bold')
ax1.set_ylabel("MSE Value", fontsize=11)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 30, f"{yval:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
df_r2 = df.sort_values(by="R2 Score", ascending=True)
bars2 = ax2.bar(df_r2["Model"], df_r2["R2 Score"], color=colors, alpha=0.85, edgecolor='black', width=0.5)
ax2.set_title("Coefficient of Determination (R² Score)\n[Higher is Better]", fontsize=12, fontweight='bold')
ax2.set_ylabel("R² Score Value", fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.set_ylim(0, df["R2 Score"].max() * 1.15) 
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig("model_metrics_comparison.png", dpi=300)
plt.show() 
print("\n======================= MODEL PERFORMANCE METRICS =======================")
print(df.to_string(index=False))
print("=========================================================================\n")