import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.weights = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.bias = None

    def fit(self, X, y):
        m,n= X.shape
        self.weights = np.zeros(n)  # initialize weights to 0
        self.bias = 0
        for i in range(self.n_iterations):
            y_hat= X @ self.weights + self.bias  # predictions
            error = y_hat - y  # error
            dw = (1/m) * X.T @ error  # gradient for weights
            db = (1/m) * np.sum(error)  # gradient for bias
            self.weights = self.weights - self.learning_rate * dw  # update weights
            self.bias = self.bias - self.learning_rate * db  # update bias
        # self.weights = self.weights - self.learning_rate @ (self.weights @ X - y) @ X.T # for 1 sample

    def predict(self, X):
        # ŷ = X · weights + bias
        return X @ self.weights + self.bias


if __name__ == "__main__":
    np.random.seed(42)

    X = np.random.randn(100, 2) * 10
    y = 3 * X[:, 0] + 2 * X[:, 1] + 7 + np.random.randn(100) * 2

    lr = LinearRegressionGD()
    lr.fit(X, y)

    print("Learned weights:", lr.weights)
    print("Learned bias:", lr.bias)

    y_pred = lr.predict(X)
    print("Predictions:", y_pred[:5])
