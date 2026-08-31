import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_excel("Online Retail.xlsx")
df = df.dropna(subset=["CustomerID"])
df = df[(df.Quantity > 0) & (df.UnitPrice > 0)]
df = df[~df.InvoiceNo.astype(str).str.startswith("C")]
df["CustomerID"] = df.CustomerID.astype(int)
df["InvoiceDate"] = pd.to_datetime(df.InvoiceDate)
df["Total"] = df.Quantity * df.UnitPrice

# Customer features
c = df.groupby("CustomerID").agg(
    Spending=("Total","sum"),
    Quantity=("Quantity","sum"),
    Invoices=("InvoiceNo","nunique"),
    Products=("StockCode","nunique"),
    AvgValue=("Total","mean"),
    Last=("InvoiceDate","max")
).reset_index()

c["Recency"] = (df.InvoiceDate.max() + pd.Timedelta(days=1) - c.Last).dt.days
c["Segment"] = pd.qcut(c.Spending, 3, labels=["Low","Medium","High"])

# Features and target
features = ["Quantity","Invoices","Products","AvgValue","Recency"]
X, y = c[features], c.Segment

# Train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42, stratify=y)

# ID3 Decision Tree
dt = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
dt.fit(X_train, y_train)
pred = dt.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

# Decision Tree
plt.figure(figsize=(14,7))
plot_tree(dt, feature_names=features, class_names=dt.classes_, filled=True)
plt.show()

# Feature Importance
importance = pd.Series(dt.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importance:\n", importance)

importance.plot(kind="bar", color="steelblue")
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()