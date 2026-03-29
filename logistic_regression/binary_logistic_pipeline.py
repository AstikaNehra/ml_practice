import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# -------------------- Load Data --------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# -------------------- EDA --------------------
print("Shape:", X.shape)
print("Class distribution:\n", pd.Series(y).value_counts())
print("Missing values:\n", X.isnull().sum().sum())

# -------------------- Train Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------- Pipeline --------------------
pipeline_l2 = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000))
])

pipeline_l1 = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000))
])

# -------------------- Train --------------------
pipeline_l2.fit(X_train, y_train)
pipeline_l1.fit(X_train, y_train)

# -------------------- Evaluate --------------------
for name, model in [("L2", pipeline_l2), ("L1", pipeline_l1)]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {name} Regularization ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("Report:\n", classification_report(y_test, y_pred))