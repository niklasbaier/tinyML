import logging
import numpy as np
import typing as t
import matplotlib.pyplot as plt

from tinyML import commons

commons.setup_logging()
logger = logging.getLogger(__name__)


class BaseLinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y_true):
        raise NotImplementedError("Method must be implemented in subclass.")

    def predict(self, X):
        if None in (self.weights, self.bias):
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


class LinearRegressionGD(BaseLinearRegression):
    def __init__(
        self, learning_rate: float = 0.01, epochs: int = 1000, verbose: bool = False
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.loss_history: t.List[float] = []

    def fit(self, X, y_true):
        if self.verbose:
            logger.info(
                f"Fitting model with learning rate {self.learning_rate} on {self.epochs} epochs."
            )

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            loss = commons.mse(y_true, y_pred)
            self.loss_history.append(loss)

            dw = -(2 / n_samples) * np.dot(X.T, (y_true - y_pred))
            db = -(2 / n_samples) * np.sum(y_true - y_pred)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.verbose and (epoch % 100 == 0 or epoch == self.epochs - 1):
                logger.info(f"Epoch: {epoch}/{self.epochs} | Loss: {loss:.4f}")

    def visualize_loss(self):
        plt.plot(range(self.epochs), self.loss_history, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.legend()
        plt.show()


class LinearRegressionOLS(BaseLinearRegression):
    def fit(self, X, y_true):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        self.weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_true)

        self.bias = self.weights[0]
        self.weights = self.weights[1:]
