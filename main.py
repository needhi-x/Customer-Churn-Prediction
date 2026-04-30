import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ================= LOAD DATA =================
df = pd.read_csv("data/churn.csv")

print("Dataset Preview:")
print(df.head())

# ================= CLEANING =================
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# ================= EDA =================

# Churn Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.savefig("outputs/churn_distribution.png")
plt.close()

# Tenure vs Churn
plt.figure(figsize=(6,4))
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title("Tenure vs Churn")
plt.savefig("outputs/tenure_vs_churn.png")
plt.close()

# Charges vs Churn
plt.figure(figsize=(6,4))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Charges vs Churn")
plt.savefig("outputs/charges_vs_churn.png")
plt.close()

# Heatmap (only numeric)
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("outputs/heatmap.png")
plt.close()

# ================= SPLIT DATA =================
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= MULTIPLE MODELS =================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n{name} Results:")
    print("Accuracy:", acc)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

print("\nBest Model Selected:")
print(best_model)
print("Best Accuracy:", best_accuracy)

# ================= FINAL EVALUATION =================
y_pred = best_model.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ================= ROC CURVE =================
y_probs = best_model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("outputs/roc_curve.png")
plt.close()

# ================= FEATURE IMPORTANCE =================
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    feat_names = X.columns

    plt.figure(figsize=(10,5))
    sns.barplot(x=importances, y=feat_names)
    plt.title("Feature Importance")
    plt.savefig("outputs/feature_importance.png")
    plt.close()

# ================= SAVE MODEL =================
pickle.dump(best_model, open("models/churn_model.pkl", "wb"))

# Save column names
with open("models/columns.json", "w") as f:
    json.dump(list(X.columns), f)

# ================= BUSINESS INSIGHTS =================
print("\nBusiness Insights:")
print("- Customers with low tenure are more likely to churn")
print("- High monthly charges increase churn probability")
print("- Contract type plays a key role in retention")