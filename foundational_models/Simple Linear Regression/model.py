import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate 
        self.n_iterations = n_iterations   
        self.weights = None                
        self.bias = None                   

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0


        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


if __name__ == '__main__':
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1) # y = 4 + 3x + ノイズ

    model = SimpleLinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)

    y_pred_line = model.predict(X)

    print(f"学習後の重み (w): {model.weights[0]:.4f}")
    print(f"学習後のバイアス (b): {model.bias:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Actual data')
    plt.plot(X, y_pred_line, color='red', linewidth=2, label='Fitted line')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Simple Linear Regression")
    plt.legend()
    plt.show()