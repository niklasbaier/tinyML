import numpy as np
import matplotlib.pyplot as plt


class BaseLinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y_true):
        raise NotImplementedError("Method must be implemented in subclass.")

    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained yet.")
        return np.dot(X, self.weights) + self.bias

    def visualize_regression_line(self, X, y_truth):
        plt.scatter(X, y_truth, color="blue", label="Data Points")
        y_pred = self.predict(X)
        plt.plot(X, y_pred, color="red", label="Regression Line")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Linear Regression Fit")
        plt.legend()
        plt.show()
