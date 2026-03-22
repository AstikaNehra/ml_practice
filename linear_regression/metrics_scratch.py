import numpy as np

def mse(y_true, y_pred):
    # MSE = (1/m) * Σ(y - ŷ)²
    mse = np.mean((y_true - y_pred)**2)
    return mse

def rmse(y_true, y_pred):
    # RMSE = √MSE → reuse mse() function
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    # MAE = (1/m) * Σ|y - ŷ|
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def r2_score(y_true, y_pred):
    # R² = 1 - (SS_res / SS_tot)
    # SS_res = Σ(y - ŷ)²
    # SS_tot = Σ(y - ȳ)²
    r_2 = 1 - np.mean((y_true - y_pred)**2) / np.mean((y_true - np.mean(y_true))**2)
    return r_2

def adjusted_r2(y_true, y_pred, n_features):
    # Adjusted R²
    #       = 1 - [(1 - R²)(n - 1) / (n - p - 1)]
    #       where:
    #           n = number of samples
    #           p = number of features
    adj_r2 = 1 - r2_score(y_true, y_pred) * (n_features - 1) / (len(y_true) - n_features - 1)
    return adj_r2


if __name__ == "__main__":
    np.random.seed(42)

    X = np.random.randn(100, 2) * 10
    y_true = 3 * X[:, 0] + 2 * X[:, 1] + 7 + np.random.randn(100) * 2
    y_pred = 7 * X[:, 0] + 9 * X[:, 1] + 3 + np.random.randn(100) * 4

    mse_value = mse(y_true, y_pred)
    rmse_value = rmse(y_true, y_pred)
    mae_value = mae(y_true, y_pred)
    r2_value = r2_score(y_true, y_pred)
    adj_r2_value = adjusted_r2(y_true, y_pred, n_features=X.shape[1])

    # compare against sklearn metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_error, r2_score, mean_absolute_error, mean_squared_error, r2_score
    assert np.isclose(mse_value, mean_squared_error(y_true, y_pred))
    assert np.isclose(rmse_value, np.sqrt(mean_squared_error(y_true, y_pred)))
    assert np.isclose(mae_value, mean_absolute_error(y_true, y_pred))
    assert np.isclose(r2_value, r2_score(y_true, y_pred))
