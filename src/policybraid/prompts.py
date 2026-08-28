from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any


FAMILIES = (
    "exact_math",
    "choice_logic",
    "code_unit_tests",
    "json_schema_tool",
)
CANDIDATES_PER_FAMILY = 64
SCHEMA = "policybraid.pm_a.prompt.v1"


def _number(seed: int, *parts: object, modulus: int) -> int:
    material = ":".join(str(part) for part in (seed, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % modulus


def _record(
    prompt_id: str,
    family: str,
    prompt: str,
    verifier: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "prompt_id": prompt_id,
        "task_family": family,
        "prompt": prompt,
        "verifier": verifier,
    }


def _integer_options(
    seed: int, candidate_index: int, item_index: int, expected: int
) -> list[int]:
    offsets = (1, -1, 2, -2, 5, -5, 10, -10)
    distractors = []
    for offset in offsets:
        value = expected + offset
        if value not in distractors and value != expected:
            distractors.append(value)
        if len(distractors) == 3:
            break
    options = [expected, *distractors]
    return sorted(
        options,
        key=lambda value: _number(
            seed,
            "math-r3-option",
            candidate_index,
            item_index,
            value,
            modulus=2**32,
        ),
    )


def _math_candidates(seed: int) -> list[dict[str, Any]]:
    records = []
    coprime_moduli = ((5, 7), (7, 9), (8, 11), (11, 13))
    for index in range(CANDIDATES_PER_FAMILY):
        questions: list[str] = []
        expected: list[int] = []

        left = 23 + _number(seed, "math-r3", index, "left", modulus=79)
        right = 11 + _number(seed, "math-r3", index, "right", modulus=43)
        offset = 7 + _number(seed, "math-r3", index, "offset", modulus=101)
        divisor = 3 + _number(seed, "math-r3", index, "divisor", modulus=11)
        answer = (left * right + offset) // divisor
        expected.append(answer)
        questions.append(
            f"floor((({left}*{right})+{offset})/{divisor}); choose from "
            f"{_integer_options(seed, index, 0, answer)}"
        )

        value = 2 + _number(seed, "math-r3", index, "value", modulus=19)
        multiplier = 2 + _number(seed, "math-r3", index, "mul", modulus=5)
        increment = 3 + _number(seed, "math-r3", index, "inc", modulus=17)
        modulus = 71 + _number(seed, "math-r3", index, "mod", modulus=80)
        steps = 4 + _number(seed, "math-r3", index, "steps", modulus=4)
        initial = value
        for _ in range(steps):
            value = (multiplier * value + increment) % modulus
        expected.append(value)
        questions.append(
            f"x_0={initial}, x_(t+1)=({multiplier}*x_t+{increment}) mod "
            f"{modulus}; find x_{steps}; choose from "
            f"{_integer_options(seed, index, 1, value)}"
        )

        n = 10 + _number(seed, "math-r3", index, "n", modulus=8)
        k = 3 + _number(seed, "math-r3", index, "k", modulus=2)
        bonus_n = 6 + _number(seed, "math-r3", index, "bn", modulus=5)
        answer = math.comb(n, k) - math.comb(n - 2, k - 2) + bonus_n
        expected.append(answer)
        questions.append(
            f"C({n},{k})-C({n - 2},{k - 2})+{bonus_n}; choose from "
            f"{_integer_options(seed, index, 2, answer)}"
        )

        first_modulus, second_modulus = coprime_moduli[index % 4]
        target = 1 + _number(
            seed,
            "math-r3",
            index,
            "target",
            modulus=first_modulus * second_modulus - 1,
        )
        first_remainder = target % first_modulus
        second_remainder = target % second_modulus
        answer = next(
            candidate
            for candidate in range(first_modulus * second_modulus)
            if candidate % first_modulus == first_remainder
            and candidate % second_modulus == second_remainder
        )
        expected.append(answer)
        questions.append(
            f"smallest nonnegative x with x mod {first_modulus}={first_remainder} "
            f"and x mod {second_modulus}={second_remainder}; choose from "
            f"{_integer_options(seed, index, 3, answer)}"
        )

        prompt = "Solve four independent exact-integer items:\n" + "\n".join(
            f"{item_index + 1}. {question}"
            for item_index, question in enumerate(questions)
        )
        prompt += (
            "\nReturn only one line `FINAL: [a,b,c,d]` in item order. "
            "Do not include reasoning or other text."
        )
        records.append(
            _record(
                f"math-{index:03d}",
                "exact_math",
                prompt,
                {"kind": "exact_integer_list", "expected": expected},
            )
        )
    return records


def _constraint_holds(order: tuple[str, ...], constraint: tuple[Any, ...]) -> bool:
    kind, *values = constraint
    positions = {item: index for index, item in enumerate(order)}
    if kind == "before":
        return positions[values[0]] < positions[values[1]]
    if kind == "immediately_before":
        return positions[values[1]] == positions[values[0]] + 1
    if kind == "position":
        return positions[values[0]] == values[1]
    if kind == "not_adjacent":
        return abs(positions[values[0]] - positions[values[1]]) != 1
    raise ValueError(f"unknown logic constraint: {kind}")


def _constraint_text(constraint: tuple[Any, ...]) -> str:
    kind, *values = constraint
    if kind == "before":
        return f"{values[0]} appears before {values[1]}."
    if kind == "immediately_before":
        return f"{values[0]} appears immediately before {values[1]}."
    if kind == "position":
        return f"{values[0]} is in position {values[1] + 1}."
    if kind == "not_adjacent":
        return f"{values[0]} is not adjacent to {values[1]}."
    raise ValueError(f"unknown logic constraint: {kind}")


def _logic_puzzle(seed: int, puzzle_index: int) -> tuple[str, str]:
    labels = "ABCD"
    items = tuple("PQRST")
    target = tuple(
        sorted(
            items,
            key=lambda item: _number(
                seed, "logic-r3", puzzle_index, "target", item, modulus=2**32
            ),
        )
    )
    swaps = ((0, 1), (1, 2), (3, 4))
    options = [target]
    for left, right in swaps:
        candidate = list(target)
        candidate[left], candidate[right] = candidate[right], candidate[left]
        options.append(tuple(candidate))
    options = sorted(
        options,
        key=lambda option: _number(
            seed,
            "logic-r3",
            puzzle_index,
            "option",
            "".join(option),
            modulus=2**32,
        ),
    )

    true_constraints: list[tuple[Any, ...]] = []
    for left in range(len(target)):
        for right in range(left + 1, len(target)):
            true_constraints.append(("before", target[left], target[right]))
            if right - left > 1:
                true_constraints.append(("not_adjacent", target[left], target[right]))
    true_constraints.extend(
        ("immediately_before", target[position], target[position + 1])
        for position in range(len(target) - 1)
    )
    true_constraints.extend(
        ("position", item, position) for position, item in enumerate(target)
    )
    true_constraints.sort(
        key=lambda constraint: _number(
            seed,
            "logic-r3",
            puzzle_index,
            "constraint",
            repr(constraint),
            modulus=2**32,
        )
    )
    selected: list[tuple[Any, ...]] = []
    survivors = list(options)
    for constraint in true_constraints:
        reduced = [
            option for option in survivors if _constraint_holds(option, constraint)
        ]
        if len(reduced) < len(survivors):
            selected.append(constraint)
            survivors = reduced
        if len(survivors) == 1 and len(selected) >= 3:
            break
    for constraint in true_constraints:
        if len(selected) >= 4:
            break
        if constraint not in selected:
            selected.append(constraint)
    if survivors != [target]:
        raise RuntimeError(f"logic puzzle {puzzle_index} lacks a unique option")

    answer = labels[options.index(target)]
    rules = " ".join(
        f"R{rule_index + 1}: {_constraint_text(constraint)}"
        for rule_index, constraint in enumerate(selected)
    )
    option_text = "; ".join(
        f"{label}={' '.join(option)}"
        for label, option in zip(labels, options, strict=True)
    )
    return (
        "Five jobs P,Q,R,S,T are ordered first to fifth. "
        f"{rules} Which option satisfies all rules? {option_text}",
        answer,
    )


def _choice_candidates(seed: int) -> list[dict[str, Any]]:
    records = []
    for index in range(CANDIDATES_PER_FAMILY):
        puzzles = [
            _logic_puzzle(seed, index * 4 + item_index)
            for item_index in range(4)
        ]
        prompt = "Solve four independent ordering questions:\n" + "\n".join(
            f"{item_index + 1}. {question}"
            for item_index, (question, _answer) in enumerate(puzzles)
        )
        prompt += (
            "\nReturn only one line `FINAL: [A,B,C,D]` in question order. "
            "Do not include reasoning or other text."
        )
        records.append(
            _record(
                f"logic-{index:03d}",
                "choice_logic",
                prompt,
                {
                    "kind": "multiple_choice_list",
                    "expected": [answer for _question, answer in puzzles],
                },
            )
        )
    return records


def _sequence(seed: int, *parts: object, length: int, low: int, high: int) -> list[int]:
    width = high - low + 1
    return [
        low + _number(seed, *parts, position, modulus=width)
        for position in range(length)
    ]


def _balanced_cut(values: list[int]) -> int:
    total = sum(values)
    prefix = 0
    for index in range(1, len(values)):
        prefix += values[index - 1]
        if prefix == total - prefix:
            return index
    return -1


def _cyclic_peaks(values: list[int]) -> int:
    return sum(
        value > values[(index - 1) % len(values)]
        and value >= values[(index + 1) % len(values)]
        for index, value in enumerate(values)
    )


def _code_candidates(seed: int) -> list[dict[str, Any]]:
    records = []
    for index in range(CANDIDATES_PER_FAMILY):
        task = index // 16
        variant = index % 16
        tests: list[dict[str, Any]] = []
        if task == 0:
            window = 2 + variant % 5
            specification = (
                f"For a nonempty integer list, return the smallest zero-based start "
                f"index of a length-{window} window with maximum sum."
            )
            for test in range(5):
                values = _sequence(
                    seed,
                    "code-r2",
                    index,
                    test,
                    length=window + 5 + test,
                    low=-10,
                    high=15,
                )
                sums = [
                    sum(values[start : start + window])
                    for start in range(len(values) - window + 1)
                ]
                tests.append({"args": [values], "expected": sums.index(max(sums))})
        elif task == 1:
            specification = (
                "Return the smallest cut index i with 1 <= i < len(values) for "
                "which sum(values[:i]) equals sum(values[i:]); return -1 if absent."
            )
            for test in range(5):
                values = _sequence(
                    seed, "code-r2", index, test, length=6 + test, low=-5, high=9
                )
                if test == variant % 5:
                    values[-1] = sum(values[:-1])
                tests.append({"args": [values], "expected": _balanced_cut(values)})
        elif task == 2:
            shift = 1 + variant
            specification = (
                f"Split a string on single spaces, rotate every nonempty word left by "
                f"{shift} positions modulo its length, and rejoin with single spaces."
            )
            words = ("alpha", "braid", "runtime", "policy", "tensor", "group")
            for test in range(5):
                chosen = [
                    words[(variant + test + step * 2) % len(words)]
                    for step in range(3 + test % 2)
                ]
                text = " ".join(chosen)
                expected = " ".join(
                    word[shift % len(word) :] + word[: shift % len(word)]
                    for word in chosen
                )
                tests.append({"args": [text], "expected": expected})
        else:
            specification = (
                "Treat the list as circular. Return the number of indices i for which "
                "values[i] is strictly greater than its previous neighbor and greater "
                "than or equal to its next neighbor. The first and last elements are "
                "neighbors. The input has at least three elements."
            )
            for test in range(5):
                values = _sequence(
                    seed, "code-r2", index, test, length=7 + test, low=-5, high=12
                )
                tests.append({"args": [values], "expected": _cyclic_peaks(values)})

        prompt = (
            "Write Python 3 code defining exactly `def solve(values):` or "
            "`def solve(text):` as implied below. Do not import modules, read input, "
            "print, or access files. Return only one Python code block.\n\n"
            f"Specification: {specification}"
        )
        records.append(
            _record(
                f"code-{index:03d}",
                "code_unit_tests",
                prompt,
                {
                    "kind": "python_unit_tests",
                    "function": "solve",
                    "tests": tests,
                },
            )
        )
    return records


def _json_candidates(seed: int) -> list[dict[str, Any]]:
    records = []
    names = ("Ada", "Bo", "Chen", "Dina", "Eli", "Faye", "Gus", "Hana")
    for index in range(CANDIDATES_PER_FAMILY):
        task = index // 8
        variant = index % 8
        if task == 0:
            units = 7 + _number(seed, "json", index, "units", modulus=35)
            unit_price = 3 + variant
            sku = f"PB-{index:03d}"
            expected = {
                "tool": "inventory_summary",
                "arguments": {
                    "sku": sku,
                    "units": units,
                    "subtotal": units * unit_price,
                },
            }
            task_text = (
                f"SKU {sku} has {units} units at integer unit price {unit_price}. "
                "Create an inventory_summary tool call with sku, units, and subtotal. "
                "Required shape: {\"tool\":\"inventory_summary\",\"arguments\":"
                "{\"sku\":<string>,\"units\":<integer>,\"subtotal\":<integer>}}."
            )
        elif task == 1:
            values = _sequence(
                seed, "json", index, length=9, low=-8, high=21
            )
            evens = sorted({value for value in values if value % 2 == 0})
            expected = {"even_values": evens, "count": len(evens), "sum": sum(evens)}
            task_text = (
                f"For values {values}, deduplicate the even values, sort ascending, "
                "and report even_values, count, and sum. Required shape: "
                "{\"even_values\":<array>,\"count\":<integer>,\"sum\":<integer>}."
            )
        elif task == 2:
            raw_name = f"  {names[variant].upper()}-{names[(variant + 3) % 8].lower()}  "
            scores = _sequence(
                seed, "json", index, length=4, low=40, high=99
            )
            normalized = raw_name.strip().lower().replace("-", " ")
            expected = {
                "profile": {"name": normalized, "score_max": max(scores)},
                "passed": max(scores) >= 70,
            }
            task_text = (
                f"Normalize the name {raw_name!r} by trimming, lowercasing, and "
                f"replacing '-' with one space. Scores are {scores}. Return profile "
                "with name and score_max, plus passed (score_max >= 70). Required "
                "shape: {\"profile\":{\"name\":<string>,\"score_max\":<integer>},"
                "\"passed\":<boolean>}."
            )
        elif task == 3:
            latency = 40 + _number(seed, "json", index, "latency", modulus=180)
            errors = _number(seed, "json", index, "errors", modulus=5)
            route = "fallback" if errors >= 3 or latency > 150 else "primary"
            expected = {
                "tool": "route_request",
                "arguments": {
                    "route": route,
                    "observed_ms": latency,
                    "errors": errors,
                },
            }
            task_text = (
                f"Observed latency is {latency} ms and errors is {errors}. Route to "
                "fallback iff errors >= 3 or latency > 150; otherwise primary. "
                "Required shape: {\"tool\":\"route_request\",\"arguments\":"
                "{\"route\":<string>,\"observed_ms\":<integer>,"
                "\"errors\":<integer>}}."
            )
        elif task == 4:
            matrix = [
                _sequence(seed, "json", index, row, length=4, low=-5, high=12)
                for row in range(3)
            ]
            row_sums = [sum(row) for row in matrix]
            expected = {
                "row_sums": row_sums,
                "largest_row": row_sums.index(max(row_sums)),
            }
            task_text = (
                f"For matrix {matrix}, return each row sum and the zero-based index "
                "of the first row with the largest sum. Required shape: "
                "{\"row_sums\":<array>,\"largest_row\":<integer>}."
            )
        elif task == 5:
            roles = ["reader", "writer", "auditor", "operator"]
            assigned = [roles[(variant + step * 2) % 4] for step in range(6)]
            permissions = {
                "reader": {"read"},
                "writer": {"read", "write"},
                "auditor": {"read", "audit"},
                "operator": {"read", "execute"},
            }
            effective = sorted(set().union(*(permissions[role] for role in assigned)))
            expected = {
                "roles": sorted(set(assigned)),
                "effective_permissions": effective,
            }
            task_text = (
                f"Assigned roles are {assigned}. Permissions: reader=[read], "
                "writer=[read,write], auditor=[read,audit], "
                "operator=[read,execute]. Deduplicate and sort roles and the union "
                "of permissions. Required shape: {\"roles\":<array>,"
                "\"effective_permissions\":<array>}."
            )
        elif task == 6:
            start = 9 + variant
            durations = _sequence(
                seed, "json", index, length=4, low=3, high=17
            )
            offsets = []
            cursor = start
            for duration in durations:
                offsets.append({"start": cursor, "end": cursor + duration})
                cursor += duration
            expected = {"segments": offsets, "total_duration": sum(durations)}
            task_text = (
                f"Starting at offset {start}, lay out consecutive segments with "
                f"durations {durations}. Required shape: "
                "{\"segments\":"
                "[{\"start\":<integer>,\"end\":<integer>},...],"
                "\"total_duration\":<integer>}."
            )
        else:
            values = _sequence(
                seed, "json", index, length=12, low=0, high=9
            )
            threshold = 4 + variant % 3
            low = [value for value in values if value < threshold]
            high = [value for value in values if value >= threshold]
            expected = {
                "threshold": threshold,
                "buckets": {
                    "low": {"count": len(low), "sum": sum(low)},
                    "high": {"count": len(high), "sum": sum(high)},
                },
            }
            task_text = (
                f"Split values {values} at threshold {threshold}: low is < threshold "
                "and high is >= threshold. Return threshold and nested low/high count "
                "and sum objects. Required shape: {\"threshold\":<integer>,"
                "\"buckets\":{\"low\":{\"count\":<integer>,\"sum\":<integer>},"
                "\"high\":{\"count\":<integer>,\"sum\":<integer>}}}."
            )
        prompt = (
            task_text
            + " Return exactly one JSON object with exactly the requested keys, no prose."
        )
        records.append(
            _record(
                f"json-{index:03d}",
                "json_schema_tool",
                prompt,
                {"kind": "exact_json", "expected": expected},
            )
        )
    return records


def build_candidates(seed: int) -> list[dict[str, Any]]:
    records = [
        *_math_candidates(seed),
        *_choice_candidates(seed),
        *_code_candidates(seed),
        *_json_candidates(seed),
    ]
    ids = [record["prompt_id"] for record in records]
    if len(records) != len(FAMILIES) * CANDIDATES_PER_FAMILY:
        raise RuntimeError("candidate generator produced the wrong record count")
    if len(ids) != len(set(ids)):
        raise RuntimeError("candidate generator produced duplicate prompt IDs")
    return records


def jsonl_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        for record in records
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return records
