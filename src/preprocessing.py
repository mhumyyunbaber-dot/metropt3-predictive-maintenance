
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metropt+3+dataset"
    / "MetroPT3(AirCompressor).csv"
)

CONTINUOUS_COLS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]

FAILURE_WINDOWS = [
    {
        "start": "2020-04-18 00:00:00",
        "end": "2020-04-18 23:59:00",
        "event": "Air leak / high stress",
    },
    {
        "start": "2020-05-29 23:30:00",
        "end": "2020-05-30 06:00:00",
        "event": "Air leak / high stress",
    },
    {
        "start": "2020-06-05 10:00:00",
        "end": "2020-06-07 14:30:00",
        "event": "Air leak / high stress",
    },
    {
        "start": "2020-07-15 14:30:00",
        "end": "2020-07-15 19:00:00",
        "event": "Air leak / high stress",
    },
]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required_cols = ["timestamp", *CONTINUOUS_COLS]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df["timestamp"].isna().any():
        raise ValueError("Dataset contains missing timestamps.")

    if df[CONTINUOUS_COLS].isna().any().any():
        raise ValueError("Continuous sensor features contain missing values.")

    return df


def add_failure_window_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label observations inside documented event windows as 1.

    Label 0 means outside these windows, not confirmed healthy.
    """
    result = df.copy()
    timestamps = result["timestamp"]
    failure_mask = pd.Series(False, index=result.index)

    for window in FAILURE_WINDOWS:
        start = pd.Timestamp(window["start"])
        end = pd.Timestamp(window["end"])

        failure_mask |= timestamps.between(start, end)

    result["failure_window"] = failure_mask.astype("int8")
    return result


def load_and_prepare_data() -> pd.DataFrame:
    """Load the dataset and add provisional failure-window labels."""
    df = load_data()
    return add_failure_window_labels(df)


if __name__ == "__main__":
    df = load_and_prepare_data()

    print("Dataset shape:", df.shape)
    print("Timestamp range:", df["timestamp"].min(), "to", df["timestamp"].max())
    print("\nFailure-window label counts:")
    print(df["failure_window"].value_counts().sort_index())
