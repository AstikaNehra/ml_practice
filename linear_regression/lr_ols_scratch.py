# ============================================================
# LINEAR REGRESSION FROM SCRATCH USING NUMPY
# ============================================================
# GOAL: Implement LR without any ML libraries (no sklearn)
# Only numpy allowed for math operations
# ============================================================

import numpy as np

class LinearRegressionOLS:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # ============================================================
        # GOAL: find best weights and bias using OLS formula:
        #       theta = (XᵀX)⁻¹ Xᵀy
        #
        # ⚠️ THETA CONFUSION — two different contexts:
        #
        #   Simple notation  →  ŷ = θx + b   →  theta = just the slope
        #   Matrix notation  →  ŷ = X_b · θ  →  theta = [bias, w1, w2]
        #
        #   Here we use MATRIX notation — theta contains EVERYTHING
        #   bias is packed inside theta, not separate!
        #
        # ============================================================
        # WHY ADD A COLUMN OF 1s?
        # ============================================================
        #
        # Our equation:  ŷ = w1*x1 + w2*x2 + b
        # Rewrite as:    ŷ = w1*x1 + w2*x2 + b*1  ← bias * fake feature
        #
        # If we add a column of 1s to X, bias becomes just another weight!
        # OLS can then solve weights AND bias together in one shot.
        #
        # Original X:          X_b (after adding 1s):     theta (OLS output):
        #
        # x1      x2           1       x1      x2          [7.02, 3.01, 1.99]
        # ――――――――――――  →      ――――――――――――――――――――  →       ↑     ↑     ↑
        # 3.2     1.5          1       3.2     1.5          bias   w1    w2
        # 2.1     4.3          1       2.1     4.3
        # 5.0     2.2          1       5.0     2.2
        #                      ↑
        #                 always 1
        #          bias is just a weight for this column!
        #
        # ============================================================
        # THETA vs WEIGHTS — what's the difference?
        # ============================================================
        #
        #   theta   = [bias, w1, w2]  → complete package (3 elements)
        #                                used with X_b (has 1s column)
        #
        #   weights = [w1, w2]        → partial (2 elements)
        #                                used with original X (no 1s column)
        #
        #   Think of theta as a backpack containing everything.
        #   Weights = same backpack but bias taken out separately.
        #
        # ============================================================
        # EXTRACTION — why theta[0] is bias and theta[1:] are weights?
        # ============================================================
        #
        #   We added 1s column at the FRONT of X_b
        #   OLS solves left to right → first parameter = bias
        #
        #   theta = [7.02,   3.01,   1.99]
        #            ↑        ↑       ↑
        #          index 0  index 1  index 2
        #           bias     w1       w2
        #
        #   self.bias    = theta[0]   → scalar 7.02
        #   self.weights = theta[1:]  → array [3.01, 1.99]
        # ============================================================

        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))                          # add bias column at front
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y    # OLS closed form solution
        self.bias = theta[0]                                 # first element → bias
        self.weights = theta[1:]                             # rest → weights per feature

    def predict(self, X):
        # ŷ = X · weights + bias
        # X is original (no 1s column) so bias added separately
        return X @ self.weights + self.bias


if __name__ == "__main__":
    np.random.seed(42)

    # 100 samples, 2 features — values between -10 and 10
    X = np.random.randn(100, 2) * 10

    # True relationship: y = 3*x1 + 2*x2 + 7 + noise
    # So true weights = [3, 2], true bias = 7
    # After training, learned values should be close to these
    y = 3 * X[:, 0] + 2 * X[:, 1] + 7 + np.random.randn(100) * 2

    lr = LinearRegressionOLS()
    lr.fit(X, y)

    print("Learned weights:", lr.weights)   # should be close to [3, 2]
    print("Learned bias:", lr.bias)         # should be close to 7

    y_pred = lr.predict(X)
    print("Predictions:", y_pred[:5])