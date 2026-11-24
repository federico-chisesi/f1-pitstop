"""Scarica da OpenF1 i dati del GP di Barcellona 2026 (session_key 11307)."""

import json
import os
import time

import requests

SESSION_KEY = 11307
TABLES = ["laps", "stints", "pit", "race_control", "drivers"]

os.makedirs("data/raw", exist_ok=True)

for table in TABLES:
    url = f"https://api.openf1.org/v1/{table}?session_key={SESSION_KEY}"
    data = requests.get(url, timeout=60).json()
    with open(f"data/raw/{table}.json", "w") as f:
        json.dump(data, f)
    print(f"{table}: {len(data)} righe")
    time.sleep(1)  # OpenF1 accetta poche richieste al secondo
