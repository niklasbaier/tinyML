import logging
import numpy as np
import typing as t
import matplotlib.pyplot as plt

from tinyML import commons

commons.setup_logging()
logger = logging.getLogger(__name__)


class LogisticRegression:
    def __init__(
        self, learning_rate: float = 0.01, epochs: int = 1000, verbose: bool = False
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.loss_history: t.List[float] = []
        self.accuracy_history: t.List[float] = []
        self.weights = None
        self.bias = None

    def fit(self, X, y_true):
        if self.verbose:
            logger.info(
                f"Fitting model with learning rate {self.learning_rate} on {self.epochs} epochs."
            )

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = commons.sigmoid(linear_pred)
            loss = commons.binary_cross_entropy(y_true, y_pred)
            self.loss_history.append(loss)
            accuracy = commons.accuracy(y_true, y_pred)
            self.accuracy_history.append(accuracy)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y_true))
            db = (1 / n_samples) * np.sum(y_pred - y_true)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.verbose and (epoch % 100 == 0 or epoch == self.epochs - 1):
                logger.info(
                    f"Epoch: {epoch}/{self.epochs} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f}"
                )

    def predict_probability(self, X):
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained yet.")
        linear_pred = np.dot(X, self.weights) + self.bias
        return commons.sigmoid(linear_pred)

    def predict_binary(self, X):
        y_pred_probability = self.predict_probability(X)
        return np.where(y_pred_probability >= 0.5, 1, 0)

    def visualize_loss_accuracy(self):
        plt.subplot(1, 2, 1)
        plt.plot(range(self.epochs), self.loss_history, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(
            range(self.epochs),
            self.accuracy_history,
            label="Training Accuracy",
            color="orange",
        )
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy over Epochs")
        plt.legend()

        plt.tight_layout()
        plt.show()
