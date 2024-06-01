import logging
import numpy as np
import matplotlib.pyplot as plt

from tinyML import commons

commons.setup_logging()
logger = logging.getLogger(__name__)


class LinearRegressionGD:
    def __init__(self, lr: float = 0.001, epochs: int = 1000, verbose: bool = False):
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = [float]

    def fit(self, X, y_true):
        logger.info(
            f"Fitting model with learning rate {self.lr} on {self.epochs} epochs."
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

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if self.verbose and (epoch % 100 == 0 or epoch == self.epochs - 1):
                logger.info(f"Epoch: {epoch}/{self.epochs} | Loss: {loss:.4f}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def visualize_loss(self):
        plt.plot(range(self.epochs), self.loss_history, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.legend()
        plt.show()

    def visualize_regression_line(self, X, y_truth):
        plt.scatter(X, y_truth, color="blue", label="Data Points")
        y_pred = self.predict(X)
        plt.plot(X, y_pred, color="red", label="Regression Line")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Linear Regression Fit")
        plt.legend()
        plt.show()


class LinearRegressionOLS:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y_true):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        self.weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_true)

        self.bias = self.weights[0]
        self.weights = self.weights[1:]

    def predict(self, X):
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
