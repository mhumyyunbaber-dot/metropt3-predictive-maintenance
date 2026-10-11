
from src.clustering import run_clustering
from src.feature_engineering import save_failure_window_analysis
from src.preprocessing import load_and_prepare_data


def main() -> None:
    print("Step 1/3: Loading dataset and creating provisional labels...")
    df = load_and_prepare_data()

    print("\nStep 2/3: Running K-Means clustering...")
    clustered_df = run_clustering(df)

    print("\nStep 3/3: Saving failure-window analysis...")
    save_failure_window_analysis(clustered_df)

    print("\nAnalysis pipeline completed.")
    print(
        "Note: This pipeline performs retrospective analysis; "
        "it does not train a failure-prediction classifier."
    )


if __name__ == "__main__":
    main()
