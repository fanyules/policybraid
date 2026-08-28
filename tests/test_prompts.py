import hashlib
import unittest
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from policybraid.prompts import (
    CANDIDATES_PER_FAMILY,
    FAMILIES,
    build_candidates,
    jsonl_bytes,
)
from policybraid.verifiers import validate_candidate


class PromptGenerationTests(unittest.TestCase):
    def test_generator_is_deterministic_and_balanced(self):
        first = build_candidates(240831)
        second = build_candidates(240831)
        self.assertEqual(jsonl_bytes(first), jsonl_bytes(second))
        self.assertEqual(len(first), CANDIDATES_PER_FAMILY * len(FAMILIES))
        for family in FAMILIES:
            self.assertEqual(
                sum(record["task_family"] == family for record in first),
                CANDIDATES_PER_FAMILY,
            )
        self.assertEqual(len({record["prompt_id"] for record in first}), len(first))
        for record in first:
            validate_candidate(record)

    def test_seed_changes_candidate_file(self):
        left = hashlib.sha256(jsonl_bytes(build_candidates(240831))).digest()
        right = hashlib.sha256(jsonl_bytes(build_candidates(240832))).digest()
        self.assertNotEqual(left, right)


if __name__ == "__main__":
    unittest.main()

