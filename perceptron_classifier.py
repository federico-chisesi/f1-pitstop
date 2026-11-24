"""
Perceptron semplice, come nel repository del prof (github.com/alessandrobruno10/perceptron).
Lo usiamo come BASELINE: il modello da battere.
"""

import numpy as np


class PerceptronClassifier:

    def __init__(self, learning_rate=0.01, max_iterations=100, random_state=42):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.weights = rng.normal(0, 0.01, X.shape[1])
        self.bias = 0.0
        y_signed = np.where(y == 1, 1, -1)               # il perceptron lavora con -1 e +1

        for _ in range(self.max_iterations):
            for xi, yi in zip(X, y_signed):
                prediction = 1 if xi @ self.weights + self.bias > 0 else -1
                if prediction != yi:                      # aggiorna i pesi solo se sbaglia
                    self.weights += self.learning_rate * yi * xi
                    self.bias += self.learning_rate * yi
        return self

    def predict(self, X):
        return (X @ self.weights + self.bias > 0).astype(int)
