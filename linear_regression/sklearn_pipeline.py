# ============================================================
# LINEAR REGRESSION PIPELINE — INTERVIEW VERSION
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ====================================================================
# STEP 1: Load Data
# ====================================================================
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="MedHouseVal")

# ====================================================================
# STEP 2: Quick EDA
# ====================================================================
print(X.shape)                              # rows, cols
print(X.isnull().sum())                     # missing values
print(X.duplicated().sum())                 # duplicates
print(X.describe())                         # stats — spot skew, outliers

# feature-target correlation — quick way to spot useless features
print(X.apply(lambda col: col.corr(y)).sort_values(ascending=False))

# ====================================================================
# STEP 3: Preprocessing
# ====================================================================

# remove capped target values (data artifact — spike at 5.0)
X = X[y < 5.0].copy()
y = y[y < 5.0].copy()

# log transform skewed features
for col in ['MedInc', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']:
    X[col] = np.log1p(X[col])

# log transform target
y = np.log1p(y)

# ====================================================================
# STEP 4: VIF Check — detect multicollinearity
# ====================================================================
# VIF > 10 = severe multicollinearity = drop that feature
# drop the one with lower target correlation

vif = pd.DataFrame()
vif["Feature"] = X.columns
vif["VIF"]     = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif.sort_values("VIF", ascending=False))

# drop AveBedrms — highest VIF + weakest target correlation (-0.046)
X = X.drop(columns=["AveBedrms"])

# ====================================================================
# STEP 5: Train/Test Split
# ====================================================================
# split AFTER preprocessing, BEFORE scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====================================================================
# STEP 6: Scale — fit on train only (prevent data leakage)
# ====================================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)         # no fit here!

# ====================================================================
# STEP 7: Train Models
# ====================================================================
models = {
    "Linear Regression" : LinearRegression(),
    "Ridge (L2)"        : Ridge(alpha=1.0),
    "Lasso (L1)"        : Lasso(alpha=0.01)
}

for name, model in models.items():
    model.fit(X_train, y_train)

# ====================================================================
# STEP 8: Cross Validation
# ====================================================================
print("\n--- 5-Fold Cross Validation ---")
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"{name:20s} → mean R²: {scores.mean():.4f}  std: {scores.std():.4f}")

# ====================================================================
# STEP 9: Metrics on Test Set
# ====================================================================
print("\n--- Test Set Metrics ---")
for name, model in models.items():
    y_pred = model.predict(X_test)
    r2     = r2_score(y_test, y_pred)
    n, p   = X_test.shape
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print(f"{name:20s} → MSE: {mean_squared_error(y_test, y_pred):.4f} "
          f"| RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f} "
          f"| MAE: {mean_absolute_error(y_test, y_pred):.4f} "
          f"| R²: {r2:.4f} | Adj R²: {adj_r2:.4f}")

param_grid = {'alpha': [0.1, 1, 10, 100, 1000]}
grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
grid.fit(X_train, y_train)
print("Best alpha:", grid.best_params_)
print("Best R²:", grid.best_score_)

# ====================================================================
# STEP 10: Residual Plot
# ====================================================================
# good model → random scatter around 0
# bad model  → fan shape (heteroscedasticity) or curve (non-linearity)
y_pred    = models["Linear Regression"].predict(X_test)
residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.4)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values (ŷ)")
plt.ylabel("Residuals (y - ŷ)")
plt.title("Residual Plot")
plt.savefig("residual_plot.png", dpi=150, bbox_inches='tight')
# plt.show()

# ====================================================================
# STEP 11: Q-Q Plot
# ====================================================================
# checks if residuals are normally distributed (LR assumption)
# good → points on diagonal line
# bad  → points curve away from diagonal
plt.figure(figsize=(6, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.savefig("qq_plot.png", dpi=150, bbox_inches='tight')
# plt.show()