import unittest
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from policybraid.grpo import gradient_distance, group_advantages, quantile_higher


class GrpoTests(unittest.TestCase):
    def test_advantages_are_centered_and_population_normalized(self):
        advantages = group_advantages([0.0, 0.5, 1.0], 1e-12)
        self.assertAlmostEqual(float(advantages.mean()), 0.0)
        self.assertAlmostEqual(float(np.mean(advantages * advantages)), 1.0)

    def test_constant_rewards_produce_zero_advantage(self):
        np.testing.assert_array_equal(group_advantages([0.5] * 8, 1e-4), 0.0)

    def test_gradient_distance_uses_left_norm(self):
        self.assertAlmostEqual(
            gradient_distance(np.array([3.0, 4.0]), np.array([0.0, 0.0]), 1e-12),
            1.0,
        )

    def test_higher_quantile_is_conservative_for_four_values(self):
        self.assertEqual(quantile_higher([0.1, 0.2, 0.3, 0.4], 0.95), 0.4)


if __name__ == "__main__":
    unittest.main()

