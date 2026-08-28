from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any


@dataclass(frozen=True)
class Verification:
    reward: int
    reason: str


_INTEGER_PATTERN = re.compile(r"FINAL\s*:\s*([-+]?\d+)\s*$", re.IGNORECASE)
_CHOICE_PATTERN = re.compile(r"FINAL\s*:\s*([ABCD])\s*$", re.IGNORECASE)
_CODE_FENCE_PATTERN = re.compile(
    r"```(?:python|py)?\s*\n(?P<code>.*?)```", re.IGNORECASE | re.DOTALL
)
_JSON_FENCE_PATTERN = re.compile(
    r"```(?:json)?\s*\n(?P<json>.*?)```", re.IGNORECASE | re.DOTALL
)

_FORBIDDEN_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.NamedExpr,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)
_SAFE_CALLS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "int",
    "len",
    "list",
    "max",
    "min",
    "range",
    "reversed",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}
_SAFE_METHODS = {
    "append",
    "clear",
    "copy",
    "count",
    "endswith",
    "extend",
    "get",
    "index",
    "insert",
    "items",
    "join",
    "keys",
    "lower",
    "pop",
    "remove",
    "replace",
    "reverse",
    "sort",
    "split",
    "startswith",
    "strip",
    "upper",
    "values",
}


def _extract_code(text: str) -> str:
    matches = list(_CODE_FENCE_PATTERN.finditer(text.strip()))
    if matches:
        return matches[0].group("code").strip()
    start = text.find("def solve")
    return text[start:].strip() if start >= 0 else ""


def _validate_code(code: str, function_name: str) -> None:
    if not code:
        raise ValueError("no Python function found")
    tree = ast.parse(code, mode="exec")
    top_level_functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    ]
    other_statements = [
        node
        for node in tree.body
        if not isinstance(node, ast.FunctionDef)
        and not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    if len(top_level_functions) != 1 or top_level_functions[0].name != function_name:
        raise ValueError(f"code must define exactly one function named {function_name}")
    if other_statements:
        raise ValueError("code contains a top-level executable statement")
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise ValueError(f"forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names are forbidden")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("private or dunder attributes are forbidden")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _SAFE_CALLS:
                    raise ValueError(f"call is not allowed: {node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr not in _SAFE_METHODS:
                    raise ValueError(f"method is not allowed: {node.func.attr}")
            else:
                raise ValueError("indirect calls are forbidden")


def _verify_code(text: str, verifier: dict[str, Any]) -> Verification:
    code = _extract_code(text)
    try:
        _validate_code(code, verifier["function"])
    except (SyntaxError, ValueError) as error:
        return Verification(0, f"invalid_code:{error}")

    worker = Path(__file__).with_name("code_worker.py")
    payload = {
        "code": code,
        "function": verifier["function"],
        "tests": verifier["tests"],
    }
    try:
        with tempfile.TemporaryDirectory(prefix="policybraid-code-") as directory:
            payload_path = Path(directory) / "payload.json"
            payload_path.write_text(
                json.dumps(payload, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(worker), str(payload_path)],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return Verification(0, "execution_timeout")
    if completed.returncode != 0:
        return Verification(0, "execution_failed")
    try:
        actual = json.loads(completed.stdout)["actual"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return Verification(0, "invalid_worker_result")
    expected = [test["expected"] for test in verifier["tests"]]
    return Verification(int(actual == expected), "passed" if actual == expected else "failed_tests")


def _exact_json(text: str) -> Any:
    stripped = text.strip()
    match = _JSON_FENCE_PATTERN.fullmatch(stripped)
    if match:
        stripped = match.group("json").strip()
    decoder = json.JSONDecoder()
    value, end = decoder.raw_decode(stripped)
    if stripped[end:].strip():
        raise ValueError("trailing content after JSON value")
    return value


def verify_output(candidate: dict[str, Any], output: str) -> Verification:
    verifier = candidate["verifier"]
    kind = verifier["kind"]
    if kind == "exact_integer":
        match = _INTEGER_PATTERN.search(output.strip())
        if not match:
            return Verification(0, "missing_final_integer")
        passed = int(match.group(1)) == int(verifier["expected"])
        return Verification(int(passed), "passed" if passed else "wrong_integer")
    if kind == "multiple_choice":
        match = _CHOICE_PATTERN.search(output.strip())
        if not match:
            return Verification(0, "missing_final_choice")
        passed = match.group(1).upper() == verifier["expected"]
        return Verification(int(passed), "passed" if passed else "wrong_choice")
    if kind == "exact_json":
        try:
            observed = _exact_json(output)
        except (json.JSONDecodeError, ValueError):
            return Verification(0, "invalid_json")
        passed = observed == verifier["expected"]
        return Verification(int(passed), "passed" if passed else "wrong_json")
    if kind == "python_unit_tests":
        return _verify_code(output, verifier)
    raise ValueError(f"unsupported verifier kind: {kind}")


def validate_candidate(candidate: dict[str, Any]) -> None:
    required = {"schema", "prompt_id", "task_family", "prompt", "verifier"}
    if set(candidate) != required:
        raise ValueError(f"candidate keys differ: {candidate.get('prompt_id')}")
    if not candidate["prompt"] or not isinstance(candidate["prompt"], str):
        raise ValueError("candidate prompt is empty")
    verifier = candidate["verifier"]
    kind = verifier.get("kind")
    if kind == "exact_integer" and not isinstance(verifier.get("expected"), int):
        raise ValueError("exact-integer verifier lacks an integer answer")
    if kind == "multiple_choice" and verifier.get("expected") not in "ABCD":
        raise ValueError("multiple-choice verifier lacks a valid answer")
    if kind == "exact_json":
        json.dumps(verifier["expected"], allow_nan=False)
    if kind == "python_unit_tests":
        if verifier.get("function") != "solve" or len(verifier.get("tests", [])) < 3:
            raise ValueError("code verifier has an invalid test contract")
        for test in verifier["tests"]:
            if set(test) != {"args", "expected"} or not isinstance(test["args"], list):
                raise ValueError("code verifier contains an invalid test")
            json.dumps(test, allow_nan=False)
    if kind not in {
        "exact_integer",
        "multiple_choice",
        "exact_json",
        "python_unit_tests",
    }:
        raise ValueError(f"unknown verifier kind: {kind}")

