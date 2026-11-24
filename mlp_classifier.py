"""
Multilayer Perceptron con UN strato nascosto, scritto da zero con NumPy.

    ingresso (8 feature) -> strato nascosto (ReLU) -> uscita (sigmoide = probabilita' di pit)

Differenze rispetto al perceptron del prof:
  - c'e' uno strato nascosto con funzione di attivazione ReLU
  - l'uscita e' una probabilita' (sigmoide) invece di un si'/no secco
  - i pesi si correggono con la backpropagation, usando la cross-entropy come errore
"""

import pickle

import numpy as np


class MLPClassifier:

    def __init__(self, hidden_size=8, learning_rate=0.1, epochs=1000, pos_weight=1.0, random_state=42):
        self.hidden_size = hidden_size        # neuroni dello strato nascosto
        self.learning_rate = learning_rate    # quanto grandi sono le correzioni dei pesi
        self.epochs = epochs                  # quante volte la rete ripassa tutti i dati
        self.pos_weight = pos_weight          # quanto pesa un esempio di pit (classe rara)
        self.random_state = random_state
        self.loss_history = []

    def _forward(self, X):
        """Passo in avanti: dai dati alla probabilita' di pit."""
        self.z1 = X @ self.W1 + self.b1           # somma pesata nello strato nascosto
        self.a1 = np.maximum(0, self.z1)          # ReLU: tiene i positivi, azzera i negativi
        z2 = self.a1 @ self.W2 + self.b2          # somma pesata nell'uscita
        return 1 / (1 + np.exp(-z2))              # sigmoide: numero tra 0 e 1

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)

        # Pesi iniziali piccoli e casuali, bias a zero
        self.W1 = rng.normal(0, 0.5, (n_features, self.hidden_size))
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = rng.normal(0, 0.5, (self.hidden_size, 1))
        self.b2 = np.zeros(1)

        # Peso di ogni esempio: i pit valgono di piu' perche' sono pochi
        weights = np.where(y == 1, self.pos_weight, 1.0)

        for epoch in range(self.epochs):
            # 1. Passo in avanti
            p = self._forward(X)

            # 2. Errore (binary cross-entropy pesata)
            p_safe = np.clip(p, 1e-7, 1 - 1e-7)
            loss = -np.mean(weights * (y * np.log(p_safe) + (1 - y) * np.log(1 - p_safe)))
            self.loss_history.append(loss)

            # 3. Backpropagation: dall'uscita all'indietro
            d_z2 = weights * (p - y) / n_samples       # errore in uscita
            d_W2 = self.a1.T @ d_z2
            d_b2 = d_z2.sum(axis=0)
            d_a1 = d_z2 @ self.W2.T                    # errore riportato allo strato nascosto
            d_z1 = d_a1 * (self.z1 > 0)                # derivata della ReLU
            d_W1 = X.T @ d_z1
            d_b1 = d_z1.sum(axis=0)

            # 4. Discesa del gradiente: correggiamo i pesi
            self.W1 -= self.learning_rate * d_W1
            self.b1 -= self.learning_rate * d_b1
            self.W2 -= self.learning_rate * d_W2
            self.b2 -= self.learning_rate * d_b2

            if (epoch + 1) % 200 == 0:
                print(f"Epoca {epoch + 1}: loss {loss:.4f}")
        return self

    def predict_proba(self, X):
        return self._forward(X).ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
