from __future__ import annotations

from typing import Any

import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


CROSS_SCORE_ARG = "policybraid_cross_score"
_capture_enabled = False
_captures: dict[str, list[torch.Tensor]] = {}


def start_capture(keys: list[str]) -> None:
    global _capture_enabled, _captures
    if _capture_enabled:
        raise RuntimeError("cross-score capture is already active")
    if len(keys) != len(set(keys)):
        raise ValueError("cross-score keys must be unique")
    _captures = {key: [] for key in keys}
    _capture_enabled = True


def take_capture() -> dict[str, list[float]]:
    global _capture_enabled, _captures
    if not _capture_enabled:
        raise RuntimeError("cross-score capture is not active")
    payload = {
        key: [float(value) for value in torch.stack(values).to("cpu").tolist()]
        if values
        else []
        for key, values in _captures.items()
    }
    _capture_enabled = False
    _captures = {}
    return payload


def stop_capture() -> None:
    global _capture_enabled, _captures
    _capture_enabled = False
    _captures = {}


def record_and_force(logits: torch.Tensor, target: int, key: str) -> torch.Tensor:
    if not _capture_enabled or key not in _captures:
        raise RuntimeError(f"cross-score capture key is not active: {key}")
    if logits.ndim != 1:
        raise ValueError("cross-score logits processor expects a one-dimensional row")
    if target < 0 or target >= logits.numel():
        raise ValueError("cross-score target token is outside the vocabulary")
    row = logits.to(torch.float32)
    logprob = row[target] - torch.logsumexp(row, dim=0)
    if not bool(torch.isfinite(logprob)):
        raise RuntimeError("cross-score produced a non-finite logprob")
    _captures[key].append(logprob.detach())
    logits.fill_(-torch.inf)
    logits[target] = 0.0
    return logits


class CrossScoreLogitsProcessor(AdapterLogitsProcessor):
    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        extra_args = sampling_params.extra_args or {}
        request = extra_args.get(CROSS_SCORE_ARG)
        if request is None:
            return
        if not isinstance(request, dict) or set(request) != {"key", "token_ids"}:
            raise ValueError(f"{CROSS_SCORE_ARG} requires key and token_ids")
        if not isinstance(request["key"], str) or not request["key"]:
            raise ValueError("cross-score key must be a nonempty string")
        token_ids = request["token_ids"]
        if not isinstance(token_ids, list) or not token_ids or any(
            not isinstance(token_id, int) or token_id < 0 for token_id in token_ids
        ):
            raise ValueError("cross-score token_ids must be nonempty nonnegative integers")
        if (
            float(sampling_params.temperature) != 1.0
            or float(sampling_params.top_p) != 1.0
            or int(sampling_params.top_k) != 0
        ):
            raise ValueError("cross-score is frozen to full-support temperature-1 sampling")

    def new_req_logits_processor(self, params: SamplingParams):
        extra_args = params.extra_args or {}
        request = extra_args.get(CROSS_SCORE_ARG)
        if request is None:
            return None
        key = request["key"]
        targets = tuple(request["token_ids"])

        def apply(output_token_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
            step = len(output_token_ids)
            if step >= len(targets):
                return logits
            return record_and_force(logits, targets[step], key)

        return apply

    def is_argmax_invariant(self) -> bool:
        return False

