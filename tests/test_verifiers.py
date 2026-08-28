import unittest
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from policybraid.verifiers import verify_output


class VerifierTests(unittest.TestCase):
    def test_exact_integer_requires_registered_final_line(self):
        candidate = {"verifier": {"kind": "exact_integer", "expected": 42}}
        self.assertEqual(verify_output(candidate, "work\nFINAL: 42").reward, 1)
        self.assertEqual(verify_output(candidate, "42").reward, 0)

    def test_multiple_choice_is_case_insensitive(self):
        candidate = {"verifier": {"kind": "multiple_choice", "expected": "C"}}
        self.assertEqual(verify_output(candidate, "FINAL: c").reward, 1)
        self.assertEqual(verify_output(candidate, "FINAL: A").reward, 0)

    def test_json_is_exact_and_allows_one_fence(self):
        candidate = {
            "verifier": {"kind": "exact_json", "expected": {"a": 1, "b": [2]}}
        }
        self.assertEqual(
            verify_output(candidate, '```json\n{"b":[2],"a":1}\n```').reward,
            1,
        )
        self.assertEqual(
            verify_output(candidate, '{"a":1,"b":[2],"extra":0}').reward,
            0,
        )

    def test_code_runs_registered_tests(self):
        candidate = {
            "verifier": {
                "kind": "python_unit_tests",
                "function": "solve",
                "tests": [
                    {"args": [[1, 2, 3]], "expected": 6},
                    {"args": [[]], "expected": 0},
                    {"args": [[-2, 5]], "expected": 3},
                ],
            }
        }
        correct = "```python\ndef solve(values):\n    return sum(values)\n```"
        wrong = "def solve(values):\n    return len(values)"
        malicious = "def solve(values):\n    return (1).__class__"
        self.assertEqual(verify_output(candidate, correct).reward, 1)
        self.assertEqual(verify_output(candidate, wrong).reward, 0)
        self.assertEqual(verify_output(candidate, malicious).reward, 0)


if __name__ == "__main__":
    unittest.main()

