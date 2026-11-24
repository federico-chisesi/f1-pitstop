"""
Controlla se i dati sono linearmente separabili, cioe' se esiste una "linea retta"
(un iperpiano, con 8 feature) che mette tutti i pit da una parte e tutti i no pit dall'altra.

Usiamo TUTTO il dataset, pit e no pit insieme, senza dividere in training e test:
la separabilita' e' una proprieta' dei dati, non del modello.

Uso:
    python check_separability.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from build_dataset import FEATURES, LABEL
from mlp_classifier import MLPClassifier
from perceptron_classifier import PerceptronClassifier

# 1. Carichiamo tutti i dati, pit e no pit insieme, e li mettiamo sulla stessa scala
df = pd.read_csv("data/dataset.csv")
X = StandardScaler().fit_transform(df[FEATURES].values)
y = df[LABEL].values
print(f"Righe: {len(df)}  |  pit: {(y == 1).sum()}  |  no pit: {(y == 0).sum()}")

# 2. Giri identici con risposta diversa: se esistono, nessun modello puo' separarli
groups = df.groupby(FEATURES)[LABEL].nunique()
print(f"Situazioni identiche con etichette diverse: {(groups > 1).sum()}")

# 3. Test esatto: esiste un iperpiano w*x + b con y*(w*x + b) >= 1 per ogni riga?
#    Lo chiediamo alla programmazione lineare: se trova una soluzione, i dati sono separabili.
y_signed = np.where(y == 1, 1, -1)
n, d = X.shape
A = -y_signed[:, None] * np.hstack([X, np.ones((n, 1))])
result = linprog(c=np.zeros(d + 1), A_ub=A, b_ub=-np.ones(n),
                 bounds=[(None, None)] * (d + 1), method="highs")
print(f"Linearmente separabili: {'SI' if result.status == 0 else 'NO'}\n")

# 4. Quanto si avvicina una retta, rispetto all'MLP? (tutti i dati, senza test)
pos_weight = (y == 0).sum() / (y == 1).sum()
models = {
    "Perceptron semplice": PerceptronClassifier(),
    "Migliore retta (SVM lineare)": LinearSVC(class_weight="balanced", max_iter=200000),
    "MLP (8 neuroni)": MLPClassifier(pos_weight=pos_weight),
}
print(f"{'Modello':32s}{'Pit trovati':>14s}{'Falsi allarmi':>16s}")
for name, model in models.items():
    if name.startswith("MLP"):
        print("(addestramento MLP in corso...)")
    model.fit(X, y)
    tn, fp, fn, tp = confusion_matrix(y, model.predict(X)).ravel()
    print(f"{name:32s}{f'{tp} su {tp + fn}':>14s}{f'{fp} su {fp + tn}':>16s}")
