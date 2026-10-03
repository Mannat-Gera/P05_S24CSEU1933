"""Predict one delivery, from the command line."""

import pandas as pd
from pathlib import Path

from delivery import load_model


# Folder containing predict.py
WORK = Path(__file__).resolve().parent


def main():
    # Load the saved model
    model = load_model(WORK / "model.joblib")

    # Create one-row DataFrame for the required order
    order = pd.DataFrame([{
        "distance_km": 7,
        "prep_time_min": 25,
        "traffic_level": 3,
        "rain": 0
    }])

    # Make prediction
    minutes = float(model.predict(order)[0])

    # Print the required output
    print(f"PREDICTION: {minutes:.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
