from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.preprocessing import CONTINUOUS_COLS, PROJECT_ROOT


RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"

N_CLUSTERS = 3
RANDOM_STATE = 42


def run_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """Fit K-Means on the complete dataset for retrospective analysis."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[CONTINUOUS_COLS])

    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=3,
    )

    result = df.copy()
    result["cluster"] = model.fit_predict(X_scaled)

    joblib.dump(scaler, MODELS_DIR / "sensor_scaler.joblib")
    joblib.dump(model, MODELS_DIR / "kmeans_model.joblib")

    cluster_sizes = (
        result["cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="observations")
    )
    cluster_sizes.to_csv(
        METRICS_DIR / "cluster_sizes.csv", index=False
    )

    profiles = result.groupby("cluster")[CONTINUOUS_COLS].mean().round(4)
    profiles.to_csv(METRICS_DIR / "cluster_profiles.csv")

    print("Full-dataset retrospective clustering completed.")
    print("\nCluster sizes:")
    print(cluster_sizes.to_string(index=False))
    print("\nCluster profiles:")
    print(profiles)

    return result


def run_chronological_evaluation(
    df: pd.DataFrame,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on earlier observations and assign clusters to later observations.

    This evaluates unseen-data cluster assignment, not failure prediction.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    required = {"timestamp", *CONTINUOUS_COLS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    ordered = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(ordered) * train_fraction)

    train_df = ordered.iloc[:split_idx].copy()
    test_df = ordered.iloc[split_idx:].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Both training and test periods must contain rows.")

    # Fit preprocessing and clustering on training data only.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[CONTINUOUS_COLS])

    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=3,
    )
    train_df["cluster"] = model.fit_predict(X_train)

    # Never fit the scaler or K-Means on the test period.
    X_test = scaler.transform(test_df[CONTINUOUS_COLS])
    test_df["cluster"] = model.predict(X_test)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save evaluation-specific artifacts separately from the full-data model.
    joblib.dump(
        scaler, MODELS_DIR / "chronological_sensor_scaler.joblib"
    )
    joblib.dump(
        model, MODELS_DIR / "chronological_kmeans_model.joblib"
    )

    def save_percentages(
        frame: pd.DataFrame, filename: str
    ) -> None:
        percentages = (
            frame["cluster"]
            .value_counts(normalize=True)
            .mul(100)
            .sort_index()
            .rename("percentage")
            .rename_axis("cluster")
            .reset_index()
        )
        percentages.to_csv(METRICS_DIR / filename, index=False)
        print(f"\n{filename}")
        print(percentages.to_string(index=False))

    save_percentages(
        train_df, "chronological_train_cluster_percentages.csv"
    )
    save_percentages(
        test_df, "chronological_test_cluster_percentages.csv"
    )

    train_profiles = (
        train_df.groupby("cluster")[CONTINUOUS_COLS].mean().round(4)
    )
    test_profiles = (
        test_df.groupby("cluster")[CONTINUOUS_COLS].mean().round(4)
    )

    train_profiles.to_csv(
        METRICS_DIR / "chronological_train_cluster_profiles.csv"
    )
    test_profiles.to_csv(
        METRICS_DIR / "chronological_test_cluster_profiles.csv"
    )

    print("\nTraining period:")
    print(train_df["timestamp"].min(), "to", train_df["timestamp"].max())

    print("\nUnseen test period:")
    print(test_df["timestamp"].min(), "to", test_df["timestamp"].max())

    print("\nTraining cluster sizes:")
    print(train_df["cluster"].value_counts().sort_index())

    print("\nUnseen test cluster sizes:")
    print(test_df["cluster"].value_counts().sort_index())

    print(
        "\nEvaluation complete. This tests cluster assignment on later "
        "data; it does not validate failure prediction."
    )

    return train_df, test_df
