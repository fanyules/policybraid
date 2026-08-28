import unittest
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from policybraid.noise import (
    aggregate_distance_from_gram,
    paired_stratified_bootstrap_distances,
)


class NoiseTests(unittest.TestCase):
    def test_gram_aggregate_matches_direct_distance(self):
        left = np.array([[1.0, 0.0], [0.0, 1.0]])
        right = np.array([[0.5, 0.0], [0.0, 0.5]])
        combined = np.concatenate([left, right], axis=0)
        gram = combined @ combined.T
        observed = aggregate_distance_from_gram(gram, 2, 1e-12)
        expected = np.linalg.norm(left.mean(0) - right.mean(0)) / np.linalg.norm(
            left.mean(0)
        )
        self.assertAlmostEqual(observed, expected)

    def test_paired_bootstrap_is_reproducible_and_finite(self):
        left = np.eye(4)
        right = left * 0.8
        combined = np.concatenate([left, right], axis=0)
        gram = combined @ combined.T
        first = paired_stratified_bootstrap_distances(
            gram, ["a", "a", "b", "b"], 100, 9, 1e-12
        )
        second = paired_stratified_bootstrap_distances(
            gram, ["a", "a", "b", "b"], 100, 9, 1e-12
        )
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(first).all())


if __name__ == "__main__":
    unittest.main()

