import unittest
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from policybraid.prompts import FAMILIES
from policybraid.reentry import select_maximal_balanced


class ReentrySelectionTests(unittest.TestCase):
    def _fixture(self):
        candidates = []
        eligibility = []
        counts = [30, 27, 29, 28]
        for family_index, (family, eligible_count) in enumerate(
            zip(FAMILIES, counts, strict=True)
        ):
            for prompt_index in range(32):
                prompt_id = f"f{family_index}-{prompt_index:03d}"
                candidates.append(
                    {"prompt_id": prompt_id, "task_family": family, "prompt": "x"}
                )
                eligible = prompt_index < eligible_count
                eligibility.append(
                    {
                        "prompt_id": prompt_id,
                        "task_family": family,
                        "eligible": eligible,
                        "verifier_healthy": True,
                        "nonzero_reward_variance": eligible,
                    }
                )
        return candidates, {"status": "workload_insufficient", "eligibility": eligibility}

    def test_selects_first_27_per_family_and_seals_surplus(self):
        candidates, adjudication = self._fixture()
        selected, metadata = select_maximal_balanced(candidates, adjudication, 27)
        self.assertEqual(len(selected), 108)
        self.assertEqual(metadata["surplus_count"], 6)
        self.assertFalse(metadata["surplus_replacement_allowed"])
        for family_index, family in enumerate(FAMILIES):
            ids = metadata["selected_ids_by_family"][family]
            self.assertEqual(ids, [f"f{family_index}-{index:03d}" for index in range(27)])
            self.assertEqual(metadata["anchor_ids_by_family"][family], ids[:8])

    def test_expected_balance_mismatch_fails_closed(self):
        candidates, adjudication = self._fixture()
        with self.assertRaises(ValueError):
            select_maximal_balanced(candidates, adjudication, 26)


if __name__ == "__main__":
    unittest.main()

