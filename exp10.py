import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
columns = ["age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex","capital-gain", "capital-loss", "hours-per-week","native-country", "income"]
df = pd.read_csv("adult.csv")
df = df.replace("?", np.nan).dropna()
df["income"] = df["income"].str.strip().map({"<=50K": 0,">50K": 1})
X = df.drop("income", axis=1)
y = df["income"]
num = ["age", "fnlwgt", "education-num","capital-gain", "capital-loss", "hours-per-week"]
cat = ["workclass", "education", "marital-status", "occupation","relationship", "race", "sex", "native-country"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
preprocessor = ColumnTransformer([("num", StandardScaler(), num),("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
lr = Pipeline([("preprocessor", preprocessor),("classifier", LogisticRegression(max_iter=1000))])
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
dt = Pipeline([("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42))])
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)
def evaluate(name, y_test, pred):
    print("\n", name)
    print("Accuracy :", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall   :", recall_score(y_test, pred))
    print("F1-score :", f1_score(y_test, pred))
evaluate("Logistic Regression", y_test, pred_lr)
evaluate("Decision Tree", y_test, pred_dt)
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree"],
    "Accuracy": [accuracy_score(y_test, pred_lr),accuracy_score(y_test, pred_dt)],
    "Precision": [precision_score(y_test, pred_lr),precision_score(y_test, pred_dt)],
    "Recall": [recall_score(y_test, pred_lr),recall_score(y_test, pred_dt)],
    "F1-score": [f1_score(y_test, pred_lr),f1_score(y_test, pred_dt)]})
print("\nModel Comparison:")
print(results)
results.set_index("Model").plot(kind="bar", figsize=(8, 5))
plt.title("Logistic Regression vs Decision Tree")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()