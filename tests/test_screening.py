import unittest
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from policybraid.prompts import FAMILIES
from policybraid.screening import adjudicate_screening


def _fixture(eligible_per_family: int):
    candidates = []
    rows = []
    for family_index, family in enumerate(FAMILIES):
        for prompt_index in range(40):
            prompt_id = f"f{family_index}-{prompt_index:03d}"
            candidates.append(
                {"prompt_id": prompt_id, "task_family": family, "prompt": "x"}
            )
            varied = prompt_index < eligible_per_family
            rewards = [0, 1] * 4 if varied else [1] * 8
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "task_family": family,
                    "samples": [
                        {
                            "sample_id": sample_id,
                            "reward": reward,
                            "verifier_status": "completed",
                        }
                        for sample_id, reward in enumerate(rewards)
                    ],
                }
            )
    return candidates, {"status": "success", "candidates": rows}


class ScreeningTests(unittest.TestCase):
    def test_selects_first_32_eligible_per_family(self):
        candidates, screening = _fixture(35)
        adjudication, selected = adjudicate_screening(candidates, screening, 8, 32)
        self.assertEqual(adjudication["status"], "passed")
        self.assertEqual(len(selected), 128)
        for family_index, family in enumerate(FAMILIES):
            ids = [
                record["prompt_id"]
                for record in selected
                if record["task_family"] == family
            ]
            self.assertEqual(ids, [f"f{family_index}-{index:03d}" for index in range(32)])

    def test_insufficient_family_fails_closed(self):
        candidates, screening = _fixture(31)
        adjudication, selected = adjudicate_screening(candidates, screening, 8, 32)
        self.assertEqual(adjudication["status"], "workload_insufficient")
        self.assertEqual(selected, [])


if __name__ == "__main__":
    unittest.main()

