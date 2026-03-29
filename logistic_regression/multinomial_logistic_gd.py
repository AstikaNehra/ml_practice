import numpy as np


# -------------------- Softmax --------------------
def softmax(z):
    # numerical stability
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# -------------------- One-hot encoding --------------------
def one_hot(y, num_classes):
    m = len(y)
    y_onehot = np.zeros((m, num_classes))
    y_onehot[np.arange(m), y] = 1
    return y_onehot


# -------------------- Model --------------------
class MultinomialLogisticRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.W = None  # weights
        self.b = None  # bias

    def fit(self, X, y):
        m, n = X.shape
        num_classes = len(np.unique(y))

        # init
        self.W = np.zeros((n, num_classes))
        self.b = np.zeros((1, num_classes))

        y_onehot = one_hot(y, num_classes)

        for i in range(self.n_iterations):
            # forward
            logits = X @ self.W + self.b
            y_hat = softmax(logits)

            # error
            error = y_hat - y_onehot

            # gradients
            dW = (1 / m) * X.T @ error
            db = (1 / m) * np.sum(error, axis=0, keepdims=True)

            # update
            self.W -= self.lr * dW
            self.b -= self.lr * db

            # optional: print loss
            if i % 100 == 0:
                loss = -np.mean(np.sum(y_onehot * np.log(y_hat + 1e-9), axis=1))
                print(f"Iteration {i}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        logits = X @ self.W + self.b
        return softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


# -------------------- Example --------------------
if __name__ == "__main__":
    np.random.seed(42)

    # synthetic dataset (3 classes)
    X = np.random.randn(200, 2)

    # create 3 classes
    z1 = X[:, 0] + X[:, 1]
    z2 = X[:, 0] - X[:, 1]

    y = np.zeros(200, dtype=int)
    y[z1 > 1] = 1
    y[z2 > 1] = 2

    model = MultinomialLogisticRegressionGD(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    preds = model.predict(X)

    print("\nPredictions:", preds[:10])
    print("Actual:", y[:10])