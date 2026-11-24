"""
Chiede al modello se un pilota rientrera' ai box al giro successivo.
Stessa idea di classify_spam.py del prof.

Esempio:
    python predict.py --driver HAM --lap 26
"""

import argparse
import pickle

import pandas as pd

from build_dataset import FEATURES, LABEL
from mlp_classifier import MLPClassifier

parser = argparse.ArgumentParser(description="Previsione del pit stop al giro successivo")
parser.add_argument("--driver", type=str, required=True, help="Sigla del pilota, es. HAM")
parser.add_argument("--lap", type=int, required=True, help="Giro appena concluso")
args = parser.parse_args()

# Carichiamo modello, scaler e dataset
mlp = MLPClassifier.load("mlp_model.pkl")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
df = pd.read_csv("data/dataset.csv")

# Cerchiamo la riga del pilota a quel giro
row = df[(df["driver"] == args.driver.upper()) & (df["lap_number"] == args.lap)]
if row.empty:
    raise SystemExit("Giro non trovato (giro di rientro, ultimo giro o pilota ritirato)")

# Previsione
proba = mlp.predict_proba(scaler.transform(row[FEATURES].values))[0]
print(f"{args.driver.upper()}, fine giro {args.lap}: gomma {row['compound'].iloc[0]} "
      f"con {row['tyre_age'].iloc[0]} giri")
print(f"Probabilita' di pit al giro {args.lap + 1}: {proba:.2f}")
print(f"Previsione: {'RIENTRA AI BOX' if proba >= 0.5 else 'RESTA IN PISTA'}")
print(f"Realta':    {'e rientrato' if row[LABEL].iloc[0] == 1 else 'non e rientrato'}")
