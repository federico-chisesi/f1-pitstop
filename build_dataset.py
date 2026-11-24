"""
Costruisce il dataset: una riga = un pilota alla fine di un giro.
Etichetta pit_next_lap = 1 se il pilota rientra ai box nel giro successivo.
"""

import json

import pandas as pd

FEATURES = ["SOFT", "MEDIUM", "HARD", "tyre_age", "stint_number",
            "laps_remaining", "pace_loss", "neutralized"]
LABEL = "pit_next_lap"
TOTAL_LAPS = 66


def load(table):
    with open(f"data/raw/{table}.json") as f:
        return pd.DataFrame(json.load(f))


def build_dataset():
    laps = load("laps")[["driver_number", "lap_number", "lap_duration"]]
    stints = load("stints")
    pits = load("pit")
    race_control = load("race_control")
    drivers = load("drivers")

    # 1. Per ogni giro: mescola, numero dello stint ed eta' della gomma
    laps = laps.merge(stints, on="driver_number")
    laps = laps[(laps["lap_number"] >= laps["lap_start"]) & (laps["lap_number"] <= laps["lap_end"])]
    laps["tyre_age"] = laps["tyre_age_at_start"] + laps["lap_number"] - laps["lap_start"] + 1

    # 2. Mescola in tre colonne 0/1 (one-hot)
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        laps[compound] = (laps["compound"] == compound).astype(int)

    # 3. Giri che mancano alla fine
    laps["laps_remaining"] = TOTAL_LAPS - laps["lap_number"]

    # 4. Degrado: secondi persi rispetto al giro migliore fatto con queste gomme
    laps = laps.sort_values(["driver_number", "lap_number"])
    best = laps.groupby(["driver_number", "stint_number"])["lap_duration"].cummin()
    laps["pace_loss"] = (laps["lap_duration"] - best).fillna(0).clip(0, 10)

    # 5. Giri con Safety Car o Virtual Safety Car
    sc = race_control[race_control["category"] == "SafetyCar"]
    starts = sc[sc["message"].str.contains("DEPLOYED")]["lap_number"].tolist()
    ends = sc[sc["message"].str.contains("ENDING|IN THIS LAP")]["lap_number"].tolist()
    neutralized_laps = [lap for s, e in zip(starts, ends) for lap in range(s, e + 1)]
    laps["neutralized"] = laps["lap_number"].isin(neutralized_laps).astype(int)

    # 6. Etichetta: c'e' un pit stop nel giro successivo?
    pit_laps = set(zip(pits["driver_number"], pits["lap_number"]))
    laps[LABEL] = [int((d, n + 1) in pit_laps) for d, n in zip(laps["driver_number"], laps["lap_number"])]

    # 7. Togliamo i giri di rientro (il pilota e' gia' ai box) e l'ultimo giro di ogni pilota
    is_in_lap = [(d, n) in pit_laps for d, n in zip(laps["driver_number"], laps["lap_number"])]
    last_lap = laps.groupby("driver_number")["lap_number"].transform("max")
    laps = laps[~pd.Series(is_in_lap, index=laps.index) & (laps["lap_number"] < last_lap)]

    # 8. Sigla del pilota, per leggere meglio la tabella
    laps["driver"] = laps["driver_number"].map(drivers.set_index("driver_number")["name_acronym"])
    return laps[["driver", "lap_number", "compound"] + FEATURES + [LABEL]]


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv("data/dataset.csv", index=False)
    print(f"Righe: {len(df)}  |  pit stop: {df[LABEL].sum()}  ({df[LABEL].mean():.1%})")
