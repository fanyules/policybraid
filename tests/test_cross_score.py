import math
import unittest
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


class CrossScoreCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import vllm  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("vLLM is unavailable in the local test environment")

    def test_records_pre_force_logprob_and_forces_target(self):
        from policybraid.cross_score import (
            record_and_force,
            start_capture,
            take_capture,
        )

        logits = torch.tensor([0.0, 1.0, 2.0])
        expected = float(torch.log_softmax(logits, dim=0)[1])
        start_capture(["sample"])
        record_and_force(logits, 1, "sample")
        captured = take_capture()
        self.assertAlmostEqual(captured["sample"][0], expected)
        self.assertEqual(float(logits[1]), 0.0)
        self.assertTrue(math.isinf(float(logits[0])))
        self.assertTrue(math.isinf(float(logits[2])))


if __name__ == "__main__":
    unittest.main()

