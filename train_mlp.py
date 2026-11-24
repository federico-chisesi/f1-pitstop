"""
Addestra il perceptron (baseline) e l'MLP, li valuta e salva l'MLP.
Stessa struttura di train_perceptron.py del prof.
"""

import argparse
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from build_dataset import FEATURES, LABEL
from mlp_classifier import MLPClassifier
from perceptron_classifier import PerceptronClassifier

# I 6 piloti tenuti da parte per il test: il modello non li vede durante l'addestramento
TEST_DRIVERS = ["HAM", "ANT", "LEC", "VER", "NOR", "RUS"]


def evaluate(name, y_test, y_pred):
    # Una sola misura per pit e no pit insieme: media delle due classi con lo stesso peso
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro",
                                                               zero_division=0)
    print(f"\n===== {name} =====")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    print("Confusion matrix (righe = reale, colonne = previsto):")
    print(confusion_matrix(y_test, y_pred))


def main():
    parser = argparse.ArgumentParser(description="Addestra l'MLP per prevedere i pit stop")
    parser.add_argument("--hidden_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # 1. Carichiamo il dataset
    df = pd.read_csv("data/dataset.csv")
    X, y = df[FEATURES].values, df[LABEL].values
    print(f"Righe: {len(df)}  |  pit stop: {y.sum()}")

    # 2. Dividiamo per PILOTA: i piloti del test non si vedono durante l'addestramento
    is_test = df["driver"].isin(TEST_DRIVERS).values
    X_train, X_test, y_train, y_test = X[~is_test], X[is_test], y[~is_test], y[is_test]
    print(f"Piloti nel test: {', '.join(TEST_DRIVERS)}")

    # 3. Standardizziamo: media e deviazione standard calcolate SOLO sul training
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    # 4. Baseline: il perceptron del prof
    perceptron = PerceptronClassifier(random_state=args.random_state).fit(X_train, y_train)
    evaluate("Perceptron (baseline)", y_test, perceptron.predict(X_test))

    # 5. Il nostro MLP. I pit sono pochi: ognuno pesa quanto (no pit / pit) esempi normali
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    mlp = MLPClassifier(hidden_size=args.hidden_size, learning_rate=args.learning_rate,
                        epochs=args.epochs, pos_weight=pos_weight, random_state=args.random_state)
    print("\nAddestramento MLP...")
    mlp.fit(X_train, y_train)
    evaluate("MLP", y_test, mlp.predict(X_test))

    # 6. Salviamo modello e scaler (servono a predict.py)
    mlp.save("mlp_model.pkl")
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # 7. Grafico della loss: se scende, la rete sta imparando
    plt.plot(mlp.loss_history)
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.title("Curva di apprendimento dell'MLP")
    plt.savefig("loss_curve.png", dpi=150)
    print("\nSalvati: mlp_model.pkl, scaler.pkl, loss_curve.png")


if __name__ == "__main__":
    main()
