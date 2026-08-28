from __future__ import annotations

import hashlib
import itertools
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


def _math_candidates(seed: int) -> list[dict[str, Any]]:
    records = []
    coprime_moduli = ((5, 7), (7, 9), (8, 11), (11, 13))
    for index in range(CANDIDATES_PER_FAMILY):
        kind = index // 16
        if kind == 0:
            left = 37 + _number(seed, "math", index, "left", modulus=163)
            right = 19 + _number(seed, "math", index, "right", modulus=97)
            offset = 11 + _number(seed, "math", index, "offset", modulus=211)
            divisor = 3 + _number(seed, "math", index, "divisor", modulus=15)
            expected = (left * right + offset) // divisor
            question = (
                f"Compute floor((({left} * {right}) + {offset}) / {divisor})."
            )
        elif kind == 1:
            value = 2 + _number(seed, "math", index, "value", modulus=29)
            multiplier = 2 + _number(seed, "math", index, "mul", modulus=7)
            increment = 3 + _number(seed, "math", index, "inc", modulus=31)
            modulus = 101 + _number(seed, "math", index, "mod", modulus=198)
            steps = 7 + _number(seed, "math", index, "steps", modulus=6)
            initial = value
            for _ in range(steps):
                value = (multiplier * value + increment) % modulus
            expected = value
            question = (
                f"Let x_0={initial}. For t=0,...,{steps - 1}, set "
                f"x_(t+1)=({multiplier}*x_t+{increment}) mod {modulus}. "
                f"Compute x_{steps}."
            )
        elif kind == 2:
            n = 13 + _number(seed, "math", index, "n", modulus=12)
            k = 3 + _number(seed, "math", index, "k", modulus=4)
            bonus_n = 7 + _number(seed, "math", index, "bn", modulus=7)
            bonus_k = 2 + _number(seed, "math", index, "bk", modulus=3)
            expected = math.comb(n, k) - math.comb(n - 2, k - 2) + math.comb(
                bonus_n, bonus_k
            )
            question = (
                f"Compute C({n},{k}) - C({n - 2},{k - 2}) + "
                f"C({bonus_n},{bonus_k}), where C(n,k) is the binomial coefficient."
            )
        else:
            first_modulus, second_modulus = coprime_moduli[index % 4]
            target = 1 + _number(
                seed,
                "math",
                index,
                "target",
                modulus=first_modulus * second_modulus - 1,
            )
            first_remainder = target % first_modulus
            second_remainder = target % second_modulus
            expected = next(
                value
                for value in range(first_modulus * second_modulus)
                if value % first_modulus == first_remainder
                and value % second_modulus == second_remainder
            )
            question = (
                "Find the smallest nonnegative integer x such that "
                f"x mod {first_modulus} = {first_remainder} and "
                f"x mod {second_modulus} = {second_remainder}."
            )
        records.append(
            _record(
                f"math-{index:03d}",
                "exact_math",
                question
                + " Show any reasoning briefly, then end with exactly `FINAL: <integer>`. ",
                {"kind": "exact_integer", "expected": expected},
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


def _choice_candidates(seed: int) -> list[dict[str, Any]]:
    records = []
    labels = "ABCD"
    items = tuple("PQRSTU")
    for index in range(CANDIDATES_PER_FAMILY):
        target = tuple(
            sorted(
                items,
                key=lambda item: _number(
                    seed, "logic", index, "target", item, modulus=2**32
                ),
            )
        )
        swaps = ((0, 1), (2, 3), (4, 5))
        options = [target]
        for left, right in swaps:
            candidate = list(target)
            candidate[left], candidate[right] = candidate[right], candidate[left]
            options.append(tuple(candidate))
        options = sorted(
            options,
            key=lambda option: _number(
                seed, "logic", index, "option", "".join(option), modulus=2**32
            ),
        )

        true_constraints: list[tuple[Any, ...]] = []
        for left in range(len(target)):
            for right in range(left + 1, len(target)):
                true_constraints.append(("before", target[left], target[right]))
                if right - left > 1:
                    true_constraints.append(
                        ("not_adjacent", target[left], target[right])
                    )
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
                "logic",
                index,
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
            if len(survivors) == 1 and len(selected) >= 4:
                break
        for constraint in true_constraints:
            if len(selected) >= 5:
                break
            if constraint not in selected:
                selected.append(constraint)
        if survivors != [target]:
            raise RuntimeError(f"logic candidate {index} does not have a unique option")

        answer = labels[options.index(target)]
        rules = "\n".join(
            f"{rule_index + 1}. {_constraint_text(constraint)}"
            for rule_index, constraint in enumerate(selected)
        )
        option_text = "\n".join(
            f"{label}. {' '.join(option)}"
            for label, option in zip(labels, options, strict=True)
        )
        prompt = (
            "Six jobs P, Q, R, S, T, U are placed from first to sixth. "
            "Which proposed order satisfies every rule?\n"
            f"{rules}\nOptions:\n{option_text}\n"
            "Show any reasoning briefly, then end with exactly `FINAL: <A|B|C|D>`."
        )
        records.append(
            _record(
                f"logic-{index:03d}",
                "choice_logic",
                prompt,
                {"kind": "multiple_choice", "expected": answer},
            )
        )
    return records


def _sequence(seed: int, *parts: object, length: int, low: int, high: int) -> list[int]:
    width = high - low + 1
    return [
        low + _number(seed, *parts, position, modulus=width)
        for position in range(length)
    ]


def _longest_alternating(values: list[int]) -> int:
    best = current = 0
    previous_sign = 0
    for value in values:
        sign = 1 if value > 0 else -1 if value < 0 else 0
        if sign == 0:
            current = 0
        elif current and sign != previous_sign:
            current += 1
        else:
            current = 1
        previous_sign = sign
        best = max(best, current)
    return best


def _collapse_runs(text: str) -> str:
    return "".join(
        f"{character}{sum(1 for _ in group)}"
        for character, group in itertools.groupby(text)
    )


def _recent_dedupe(values: list[int], window: int) -> list[int]:
    return [
        value
        for index, value in enumerate(values)
        if value not in values[max(0, index - window) : index]
    ]


def _balanced_cut(values: list[int]) -> int:
    total = sum(values)
    prefix = 0
    for index in range(1, len(values)):
        prefix += values[index - 1]
        if prefix == total - prefix:
            return index
    return -1


def _code_candidates(seed: int) -> list[dict[str, Any]]:
    records = []
    for index in range(CANDIDATES_PER_FAMILY):
        task = index // 8
        variant = index % 8
        tests: list[dict[str, Any]] = []
        if task == 0:
            offset = 2 + variant
            modulus = 97 + 2 * variant
            specification = (
                "Return sum((i + OFFSET) * values[i]) modulo MODULUS, using "
                f"zero-based i, OFFSET={offset}, MODULUS={modulus}."
            )
            for test in range(5):
                values = _sequence(
                    seed, "code", index, test, length=5 + test, low=-12, high=19
                )
                expected = sum(
                    (position + offset) * value
                    for position, value in enumerate(values)
                ) % modulus
                tests.append({"args": [values], "expected": expected})
        elif task == 1:
            specification = (
                "Return the length of the longest contiguous run whose nonzero "
                "values alternate sign. A zero breaks a run."
            )
            for test in range(5):
                values = _sequence(
                    seed, "code", index, test, length=8 + test, low=-4, high=4
                )
                tests.append(
                    {"args": [values], "expected": _longest_alternating(values)}
                )
        elif task == 2:
            window = 2 + variant % 4
            specification = (
                f"For a nonempty integer list, return the smallest zero-based start "
                f"index of a length-{window} window with maximum sum."
            )
            for test in range(5):
                values = _sequence(
                    seed,
                    "code",
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
        elif task == 3:
            specification = (
                "Collapse each maximal run in the input string to the character "
                "followed by its decimal run length; for example, aaabb becomes a3b2."
            )
            alphabet = "abcde"
            for test in range(5):
                runs = []
                for run in range(4 + test):
                    character = alphabet[(variant + run * 2 + test) % len(alphabet)]
                    length = 1 + _number(
                        seed, "code", index, test, run, modulus=4
                    )
                    runs.append(character * length)
                text = "".join(runs)
                tests.append({"args": [text], "expected": _collapse_runs(text)})
        elif task == 4:
            window = 2 + variant % 4
            specification = (
                "Scan left to right and return a list containing an input value only "
                f"when it did not occur in the previous {window} input positions."
            )
            for test in range(5):
                values = _sequence(
                    seed, "code", index, test, length=9 + test, low=0, high=6
                )
                tests.append(
                    {"args": [values], "expected": _recent_dedupe(values, window)}
                )
        elif task == 5:
            specification = (
                "Return the smallest cut index i with 1 <= i < len(values) for "
                "which sum(values[:i]) equals sum(values[i:]); return -1 if absent."
            )
            for test in range(5):
                values = _sequence(
                    seed, "code", index, test, length=6 + test, low=-5, high=9
                )
                if test == 0:
                    values[-1] = sum(values[:-1])
                tests.append({"args": [values], "expected": _balanced_cut(values)})
        elif task == 6:
            shift = 1 + variant
            specification = (
                f"Split a string on single spaces, rotate every nonempty word left by "
                f"{shift} positions modulo its length, and rejoin with single spaces."
            )
            words = ("alpha", "braid", "runtime", "policy", "tensor", "group")
            for test in range(5):
                chosen = [words[(variant + test + j) % len(words)] for j in range(3)]
                text = " ".join(chosen)
                expected = " ".join(
                    word[shift % len(word) :] + word[: shift % len(word)]
                    for word in chosen
                )
                tests.append({"args": [text], "expected": expected})
        else:
            target = 3 * variant - 5
            specification = (
                "Return the number of index pairs (i,j) with i < j and "
                f"values[i] + values[j] == {target}. Count duplicate positions."
            )
            for test in range(5):
                values = _sequence(
                    seed, "code", index, test, length=8 + test, low=-9, high=12
                )
                expected = sum(
                    values[left] + values[right] == target
                    for left in range(len(values))
                    for right in range(left + 1, len(values))
                )
                tests.append({"args": [values], "expected": expected})

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
                "Create an inventory_summary tool call with sku, units, and subtotal."
            )
        elif task == 1:
            values = _sequence(
                seed, "json", index, length=9, low=-8, high=21
            )
            evens = sorted({value for value in values if value % 2 == 0})
            expected = {"even_values": evens, "count": len(evens), "sum": sum(evens)}
            task_text = (
                f"For values {values}, deduplicate the even values, sort ascending, "
                "and report even_values, count, and sum."
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
                "with name and score_max, plus passed (score_max >= 70)."
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
                "Return a route_request tool call with route, observed_ms, and errors."
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
                "of the first row with the largest sum."
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
                "of permissions."
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
                f"durations {durations}. Return each start/end pair and total_duration."
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
                "and sum objects."
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

