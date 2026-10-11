
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.clustering as clustering
from src.preprocessing import CONTINUOUS_COLS


class TestChronologicalClustering(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        rows = 120

        self.df = pd.DataFrame({
            "timestamp": pd.date_range(
                "2020-01-01", periods=rows, freq="10s"
            )
        })

        # Create three distinct synthetic sensor patterns.
        for col in CONTINUOUS_COLS:
            self.df[col] = np.concatenate([
                rng.normal(0, 0.2, 40),
                rng.normal(5, 0.2, 40),
                rng.normal(10, 0.2, 40),
            ])

    def test_chronological_split_and_cluster_assignment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with (
                patch.object(clustering, "METRICS_DIR", temp_path / "metrics"),
                patch.object(clustering, "MODELS_DIR", temp_path / "models"),
            ):
                train_df, test_df = (
                    clustering.run_chronological_evaluation(self.df)
                )

        self.assertEqual(len(train_df), 96)
        self.assertEqual(len(test_df), 24)

        self.assertLess(
            train_df["timestamp"].max(),
            test_df["timestamp"].min(),
        )

        self.assertIn("cluster", train_df.columns)
        self.assertIn("cluster", test_df.columns)

        self.assertTrue(train_df["cluster"].between(0, 2).all())
        self.assertTrue(test_df["cluster"].between(0, 2).all())

    def test_invalid_train_fraction_raises_error(self):
        with self.assertRaises(ValueError):
            clustering.run_chronological_evaluation(
                self.df, train_fraction=1.0
            )


if __name__ == "__main__":
    unittest.main()
