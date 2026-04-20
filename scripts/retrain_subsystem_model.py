import argparse
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


WINDOW = 10

SUBSYSTEMS = {
    "air_turbo": {
        "label": "Air & Turbocharger",
        "weight": 0.25,
        "columns": [
            "MAP (kPa)",
            "Total Air Mass Flow Into Engine (kg/h)",
            "HFM Temperature",
            "Boost Temperature (kg/h)",
        ],
    },
    "thermal_cooling": {
        "label": "Thermal & Cooling",
        "weight": 0.20,
        "columns": [
            "Coolant Temperature (degree C)",
            "Fuel Temperature",
            "Cabin Temperature",
        ],
    },
    "fuel_injection": {
        "label": "Fuel & Injection",
        "weight": 0.20,
        "columns": [
            "Fuel Temperature",
            "Total Air Mass Flow Into Engine (kg/h)",
            "MAP (kPa)",
        ],
    },
    "electrical_ambient": {
        "label": "Electrical & Ambient",
        "weight": 0.15,
        "columns": [
            "Ambient Temperature (degree C)",
            "Ambient Pressure (bar)",
            "HFM Temperature",
        ],
    },
    "drive_dynamics": {
        "label": "Drive Dynamics",
        "weight": 0.20,
        "columns": [
            "Throttle Position (%)",
            "MAP (kPa)",
            "Total Air Mass Flow Into Engine (kg/h)",
        ],
    },
}


def normalize_and_harmonize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "Accelerator Pedal Position (%)" in df.columns and "Throttle Position (%)" not in df.columns:
        df["Throttle Position (%)"] = pd.to_numeric(df["Accelerator Pedal Position (%)"], errors="coerce")

    if "Coolant Temperature Raw" in df.columns and "Coolant Temperature (degree C)" not in df.columns:
        df["Coolant Temperature (degree C)"] = pd.to_numeric(df["Coolant Temperature Raw"], errors="coerce") / 16.0

    if "Boost Pressure (mV)" in df.columns and "MAP (kPa)" not in df.columns:
        df["MAP (kPa)"] = pd.to_numeric(df["Boost Pressure (mV)"], errors="coerce") * 100.0

    if "Total Air Mass Flow Into Engine" in df.columns and "Total Air Mass Flow Into Engine (kg/h)" not in df.columns:
        df["Total Air Mass Flow Into Engine (kg/h)"] = pd.to_numeric(
            df["Total Air Mass Flow Into Engine"], errors="coerce"
        )

    return df


def build_subsystem_features(df: pd.DataFrame, columns: list[str]):
    available = [c for c in columns if c in df.columns]
    if not available:
        return None, []

    feat = pd.DataFrame(index=df.index)
    feat_names = []

    for col in available:
        s = pd.to_numeric(df[col], errors="coerce").ffill().bfill().fillna(0)
        safe = (
            col.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace("%", "pct")
            .replace("°", "")
        )
        feat[f"{safe}_val"] = s.values
        feat[f"{safe}_std"] = s.rolling(WINDOW, min_periods=1).std().fillna(0).values
        feat[f"{safe}_mean"] = s.rolling(WINDOW, min_periods=1).mean().fillna(0).values
        feat[f"{safe}_roc"] = s.diff().fillna(0).values
        feat_names += [f"{safe}_val", f"{safe}_std", f"{safe}_mean", f"{safe}_roc"]

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    if feat.empty:
        return None, []
    return feat[feat_names], feat_names


def train(csv_path: str, output_dir: str):
    df = pd.read_csv(csv_path)
    df = normalize_and_harmonize(df)

    trained_models = {}
    subsystem_columns = {}

    for key, sub in SUBSYSTEMS.items():
        X, feat_names = build_subsystem_features(df, sub["columns"])
        if X is None or len(X) < 100:
            continue

        pipe = Pipeline(
            steps=[
                ("scaler", RobustScaler()),
                (
                    "iforest",
                    IsolationForest(
                        n_estimators=300,
                        contamination=0.07,
                        random_state=42,
                    ),
                ),
            ]
        )
        pipe.fit(X.values)

        # Keep threshold metadata for diagnostics; API currently uses predict() ratio.
        scores = pipe.named_steps["iforest"].score_samples(
            pipe.named_steps["scaler"].transform(X.values)
        )
        threshold = float(np.percentile(scores, 7))

        trained_models[key] = {
            "pipeline": pipe,
            "feat_names": feat_names,
            "label": sub["label"],
            "weight": sub["weight"],
            "columns": sub["columns"],
            "threshold": threshold,
            "n_train": int(len(X)),
        }
        subsystem_columns[key] = sub["columns"]

    if not trained_models:
        raise RuntimeError("No subsystem could be trained. Check CSV schema.")

    bundle = {"subsystems": trained_models, "window": WINDOW, "version": "new-data-v1"}
    joblib.dump(bundle, os.path.join(output_dir, "vehicle_risk_model.pkl"))
    joblib.dump(subsystem_columns, os.path.join(output_dir, "subsystem_columns.pkl"))

    print(f"Trained subsystems: {list(trained_models.keys())}")
    for k, info in trained_models.items():
        print(f" - {k}: {info['n_train']} rows, {len(info['feat_names'])} features")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to training CSV")
    parser.add_argument("--out", default=".", help="Output directory")
    args = parser.parse_args()
    train(args.csv, args.out)


if __name__ == "__main__":
    main()
