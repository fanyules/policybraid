from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def token_ids_from_chat_template(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("chat-template mapping omitted input_ids")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, Sequence)
        and value
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], (str, bytes))
    ):
        if len(value) != 1:
            raise ValueError("chat template returned more than one token sequence")
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("chat template did not return a token sequence")
    tokens = [int(token) for token in value]
    if not tokens:
        raise ValueError("chat template returned an empty token sequence")
    if any(token < 0 for token in tokens):
        raise ValueError("chat template returned a negative token ID")
    return tokens

