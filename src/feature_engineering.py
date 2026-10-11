
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.preprocessing import CONTINUOUS_COLS, PROJECT_ROOT


METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def save_failure_window_analysis(df: pd.DataFrame) -> None:
    """Save cluster comparisons and sensor mean comparisons."""
    required = {"cluster", "failure_window", *CONTINUOUS_COLS}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(f"Missing analysis columns: {sorted(missing)}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Counts and percentages by provisional label
    counts = pd.crosstab(df["cluster"], df["failure_window"])
    counts.to_csv(
        METRICS_DIR / "cluster_counts_by_failure_window.csv"
    )

    percentages = pd.crosstab(
        df["cluster"],
        df["failure_window"],
        normalize="columns",
    ).mul(100).round(2)

    percentages.to_csv(
        METRICS_DIR / "cluster_percentages_by_failure_window.csv"
    )

    # Sensor means, with explicitly named columns
    other_means = df.loc[
        df["failure_window"] == 0, CONTINUOUS_COLS
    ].mean()

    failure_means = df.loc[
        df["failure_window"] == 1, CONTINUOUS_COLS
    ].mean()

    comparison = pd.DataFrame({
        "Other_periods_mean": other_means,
        "Failure_window_mean": failure_means,
    })

    comparison.index.name = "sensor"
    comparison.to_csv(
        METRICS_DIR / "sensor_mean_comparison.csv"
    )

    # Figure: cluster composition within each label group
    ax = percentages.rename(columns={
        0: "Outside documented windows",
        1: "Inside documented windows",
    }).plot(kind="bar", figsize=(9, 5))

    ax.set_title("Cluster Distribution by Failure-Window Label")
    ax.set_xlabel("K-Means cluster")
    ax.set_ylabel("Observations within label group (%)")
    ax.legend(title="Provisional label")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "cluster_distribution_by_label.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    print("Failure-window analysis saved.")
    print("\nCluster counts:")
    print(counts)
    print("\nCluster distribution (%):")
    print(percentages)
    print("\nSensor mean comparison:")
    print(comparison.round(3))


if __name__ == "__main__":
    from src.clustering import run_clustering
    from src.preprocessing import load_and_prepare_data

    prepared_df = load_and_prepare_data()
    clustered_df = run_clustering(prepared_df)
    save_failure_window_analysis(clustered_df)
