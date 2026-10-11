
def main() -> None:
    print("No validated failure-prediction model is available.")
    print(
        "The saved K-Means model groups sensor operating patterns; "
        "it is not a supervised failure classifier."
    )
    print(
        "The chronological test period contains no observations "
        "inside the documented failure windows, so failure-detection "
        "performance could not be evaluated on that period."
    )


if __name__ == "__main__":
    main()
