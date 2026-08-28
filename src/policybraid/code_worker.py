from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys


SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def _limit_process() -> None:
    try:
        import resource
    except ImportError:
        return
    resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
    memory = 512 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))


def main() -> int:
    _limit_process()
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__builtins__": SAFE_BUILTINS}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        exec(compile(payload["code"], "<candidate>", "exec"), namespace, namespace)
        function = namespace[payload["function"]]
        actual = [function(*test["args"]) for test in payload["tests"]]
    print(json.dumps({"actual": actual}, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

