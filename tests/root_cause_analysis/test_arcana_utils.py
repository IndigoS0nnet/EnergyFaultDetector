import unittest

import numpy as np
import pandas as pd

from energy_fault_detector.root_cause_analysis.arcana_utils import (
    calculate_arcana_importance_time_series,
    calculate_mean_arcana_importances,
)


class TestArcanaUtils(unittest.TestCase):

    def _sample_bias(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="h")
        # Two rows where "high" dominates, two rows where the two features tie.
        return pd.DataFrame(
            {"low": [1.0, 1.0, 5.0, 5.0],
             "high": [9.0, 9.0, 5.0, 5.0]},
            index=idx,
        )

    def test_time_series_rows_sum_to_one(self):
        bias = self._sample_bias()
        result = calculate_arcana_importance_time_series(bias)

        np.testing.assert_allclose(result.sum(axis=1).to_numpy(), np.ones(len(bias)))

    def test_time_series_normalises_per_row(self):
        bias = self._sample_bias()
        result = calculate_arcana_importance_time_series(bias)

        np.testing.assert_allclose(result["low"].to_numpy(), [0.1, 0.1, 0.5, 0.5])
        np.testing.assert_allclose(result["high"].to_numpy(), [0.9, 0.9, 0.5, 0.5])

    def test_mean_importances_returns_series_sorted_ascending(self):
        bias = self._sample_bias()
        result = calculate_mean_arcana_importances(bias)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(list(result.index), ["low", "high"])
        # overall mean: low = (0.1+0.1+0.5+0.5)/4 = 0.3, high = 0.7
        np.testing.assert_allclose(result.to_numpy(), [0.3, 0.7])

    def test_mean_importances_respects_time_window(self):
        bias = self._sample_bias()
        # Only the first two rows → mean should be 0.1 / 0.9.
        window = calculate_mean_arcana_importances(
            bias, start="2024-01-01 00:00", end="2024-01-01 01:00"
        )

        np.testing.assert_allclose(window.to_numpy(), [0.1, 0.9])

    def test_mean_importances_uses_absolute_values(self):
        # Negative bias values must contribute positively to importance.
        idx = pd.date_range("2024-01-01", periods=2, freq="h")
        bias = pd.DataFrame({"a": [-3.0, -3.0], "b": [1.0, 1.0]}, index=idx)

        result = calculate_mean_arcana_importances(bias)
        np.testing.assert_allclose(sorted(result.to_numpy()), [0.25, 0.75])
