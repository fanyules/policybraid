import math
import unittest
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from policybraid.adjudication import (
    ess_fraction_from_log_weights,
    hierarchical_resample_counts,
    quantile_lower,
    sequence_mixture_log_weight,
    spearman_correlation,
)


class AdjudicationTests(unittest.TestCase):
    def test_equal_sequence_weights_have_unit_ess_fraction(self):
        self.assertAlmostEqual(ess_fraction_from_log_weights([7.0] * 8), 1.0)

    def test_dominant_sequence_reduces_ess_fraction(self):
        observed = ess_fraction_from_log_weights([100.0] + [0.0] * 7)
        self.assertAlmostEqual(observed, 1.0 / 8.0)

    def test_mixture_weight_uses_log_space_average(self):
        observed = sequence_mixture_log_weight(
            [math.log(0.5)], [math.log(0.25)], [math.log(0.75)]
        )
        self.assertAlmostEqual(observed, 0.0)

    def test_spearman_handles_ties_and_direction(self):
        self.assertAlmostEqual(
            spearman_correlation(
                np.array([1.0, 1.0, 2.0, 3.0]),
                np.array([4.0, 4.0, 2.0, 1.0]),
            ),
            -1.0,
        )

    def test_hierarchical_counts_preserve_draw_size(self):
        counts = hierarchical_resample_counts(
            ["a", "a", "b", "b"], restart_count=3, replicates=20, seed=11
        )
        self.assertEqual(counts.shape, (20, 12))
        np.testing.assert_array_equal(counts.sum(axis=1), 12.0)

    def test_lower_quantile_uses_registered_finite_sample_rule(self):
        self.assertEqual(quantile_lower(np.array([0.1, 0.2, 0.3, 0.4]), 0.05), 0.1)


if __name__ == "__main__":
    unittest.main()
