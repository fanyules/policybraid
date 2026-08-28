from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


def logsumexp(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("logsumexp requires a finite nonempty vector")
    maximum = float(array.max())
    return maximum + math.log(float(np.exp(array - maximum).sum()))


def sequence_mixture_log_weight(
    trainer_logprobs: list[float],
    a100_logprobs: list[float],
    npu_logprobs: list[float],
) -> float:
    trainer = np.asarray(trainer_logprobs, dtype=np.float64)
    a100 = np.asarray(a100_logprobs, dtype=np.float64)
    npu = np.asarray(npu_logprobs, dtype=np.float64)
    if trainer.ndim != 1 or trainer.size == 0:
        raise ValueError("sequence logprobs must be nonempty vectors")
    if trainer.shape != a100.shape or trainer.shape != npu.shape:
        raise ValueError("sequence logprob vectors must be aligned")
    if not (
        np.isfinite(trainer).all()
        and np.isfinite(a100).all()
        and np.isfinite(npu).all()
    ):
        raise ValueError("sequence logprobs must be finite")
    mixture = np.logaddexp(a100, npu) - math.log(2.0)
    return float(np.sum(trainer - mixture, dtype=np.float64))


def ess_fraction_from_log_weights(log_weights: list[float]) -> float:
    values = np.asarray(log_weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("ESS requires finite nonempty log weights")
    log_ess = 2.0 * logsumexp(values) - logsumexp(2.0 * values)
    fraction = math.exp(log_ess) / values.size
    return min(max(fraction, 0.0), 1.0)


def rankdata_average(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("rank data must be a finite nonempty vector")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = rankdata_average(left)
    right_ranks = rankdata_average(right)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    if denominator == 0.0:
        return 0.0
    return float(left_centered @ right_centered / denominator)


def quantile_lower(values: np.ndarray, probability: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("quantile values must be a finite nonempty vector")
    if not 0 <= probability <= 1:
        raise ValueError("probability must lie in [0,1]")
    return float(np.quantile(array, probability, method="lower"))


def hierarchical_resample_counts(
    task_families: list[str],
    restart_count: int,
    replicates: int,
    seed: int,
) -> np.ndarray:
    prompt_count = len(task_families)
    if prompt_count == 0 or restart_count <= 0 or replicates <= 0:
        raise ValueError("bootstrap dimensions must be positive")
    by_family: dict[str, list[int]] = defaultdict(list)
    for index, family in enumerate(task_families):
        by_family[family].append(index)
    if len(by_family) < 2:
        raise ValueError("hierarchical bootstrap requires at least two families")
    rng = np.random.default_rng(seed)
    restart_draws = rng.integers(
        0, restart_count, size=(replicates, restart_count), endpoint=False
    )
    counts = np.zeros((replicates, restart_count * prompt_count), dtype=np.float64)
    row_ids_by_family = {
        family: np.repeat(np.arange(replicates), len(indices))
        for family, indices in by_family.items()
    }
    for slot in range(restart_count):
        selected_restarts = restart_draws[:, slot]
        for family, indices in by_family.items():
            index_array = np.asarray(indices, dtype=np.int64)
            draws = rng.choice(
                index_array, size=(replicates, len(indices)), replace=True
            )
            columns = (
                np.repeat(selected_restarts, len(indices)) * prompt_count
                + draws.reshape(-1)
            )
            np.add.at(counts, (row_ids_by_family[family], columns), 1.0)
    expected = float(restart_count * prompt_count)
    if not np.all(counts.sum(axis=1) == expected):
        raise RuntimeError("hierarchical bootstrap produced invalid draw counts")
    return counts
