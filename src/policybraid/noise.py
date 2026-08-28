from __future__ import annotations

from collections import defaultdict

import numpy as np


def paired_stratified_bootstrap_distances(
    combined_gram: np.ndarray,
    task_families: list[str],
    replicates: int,
    seed: int,
    epsilon: float,
    batch_size: int = 256,
) -> np.ndarray:
    gram = np.asarray(combined_gram, dtype=np.float64)
    prompt_count = len(task_families)
    if gram.shape != (2 * prompt_count, 2 * prompt_count):
        raise ValueError("combined Gram matrix has the wrong shape")
    if not np.isfinite(gram).all():
        raise ValueError("combined Gram matrix must be finite")
    if replicates <= 0 or epsilon <= 0 or batch_size <= 0:
        raise ValueError("replicates, epsilon, and batch size must be positive")
    by_family: dict[str, list[int]] = defaultdict(list)
    for index, family in enumerate(task_families):
        by_family[family].append(index)
    if len(by_family) < 2:
        raise ValueError("task-stratified bootstrap needs at least two families")

    rng = np.random.default_rng(seed)
    counts = np.zeros((replicates, prompt_count), dtype=np.float64)
    for indices in by_family.values():
        indices_array = np.asarray(indices, dtype=np.int64)
        draws = rng.choice(indices_array, size=(replicates, len(indices)), replace=True)
        for replicate in range(replicates):
            counts[replicate] += np.bincount(
                draws[replicate], minlength=prompt_count
            )
    weights = counts / prompt_count
    distances = np.empty(replicates, dtype=np.float64)
    left_gram = gram[:prompt_count, :prompt_count]
    for start in range(0, replicates, batch_size):
        stop = min(replicates, start + batch_size)
        left_weights = weights[start:stop]
        difference_weights = np.concatenate(
            [left_weights, -left_weights], axis=1
        )
        numerator_squared = np.einsum(
            "bi,ij,bj->b",
            difference_weights,
            gram,
            difference_weights,
            optimize=True,
        )
        denominator_squared = np.einsum(
            "bi,ij,bj->b", left_weights, left_gram, left_weights, optimize=True
        )
        distances[start:stop] = np.sqrt(np.maximum(numerator_squared, 0.0)) / (
            np.sqrt(np.maximum(denominator_squared, 0.0)) + epsilon
        )
    return distances


def aggregate_distance_from_gram(
    combined_gram: np.ndarray, prompt_count: int, epsilon: float
) -> float:
    gram = np.asarray(combined_gram, dtype=np.float64)
    if gram.shape != (2 * prompt_count, 2 * prompt_count):
        raise ValueError("combined Gram matrix has the wrong shape")
    weights = np.full(prompt_count, 1.0 / prompt_count, dtype=np.float64)
    difference = np.concatenate([weights, -weights])
    numerator = np.sqrt(max(float(difference @ gram @ difference), 0.0))
    denominator = np.sqrt(
        max(float(weights @ gram[:prompt_count, :prompt_count] @ weights), 0.0)
    )
    return numerator / (denominator + epsilon)

