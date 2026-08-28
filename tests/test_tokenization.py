import unittest
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from policybraid.tokenization import token_ids_from_chat_template


class TokenizationTests(unittest.TestCase):
    def test_accepts_batch_encoding_style_mapping(self):
        self.assertEqual(
            token_ids_from_chat_template({"input_ids": [1, 2, 3]}), [1, 2, 3]
        )

    def test_unwraps_one_batched_sequence(self):
        self.assertEqual(token_ids_from_chat_template([[4, 5]]), [4, 5])

    def test_rejects_missing_or_multiple_sequences(self):
        with self.assertRaises(ValueError):
            token_ids_from_chat_template({"attention_mask": [1]})
        with self.assertRaises(ValueError):
            token_ids_from_chat_template([[1], [2]])


if __name__ == "__main__":
    unittest.main()

