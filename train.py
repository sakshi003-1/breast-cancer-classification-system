
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots without opening windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             recall_score, accuracy_score, roc_auc_score,
                             make_scorer)

os.makedirs("results", exist_ok=True)

# 1. Load data (0 = malignant, 1 = benign)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# 2. EDA
print("Shape:", X.shape, "| missing values:", X.isnull().sum().sum())
print(y.value_counts())
plt.figure(figsize=(14, 10))
sns.heatmap(X.corr(), cmap="coolwarm")
plt.savefig("results/correlation.png", bbox_inches="tight")
plt.close()

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Scale (fit on training data only)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 5. Compare models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}
rows = []
for name, m in models.items():
    m.fit(X_train_s, y_train)
    p = m.predict(X_test_s)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, p),
        "Malignant Recall": recall_score(y_test, p, pos_label=0),
        "AUC": roc_auc_score(y_test, m.predict_proba(X_test_s)[:, 1]),
    })
results = pd.DataFrame(rows).round(4)
print(results)
results.to_csv("results/model_comparison.csv", index=False)

# 6. Tune the SVM (optimise recall for malignant = class 0)
malignant_recall = make_scorer(recall_score, pos_label=0)
grid = GridSearchCV(SVC(probability=True, random_state=42),
                    {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
                    cv=5, scoring=malignant_recall)
grid.fit(X_train_s, y_train)
best = grid.best_estimator_
print("Best params:", grid.best_params_)

# 7. Final evaluation
pred = best.predict(X_test_s)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=data.target_names))
plt.figure()
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("results/confusion_matrix.png", bbox_inches="tight")
plt.close()

# 8. Save everything the app needs
joblib.dump(best, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(data.feature_names), "features.pkl")
print("Saved model, scaler and features.")