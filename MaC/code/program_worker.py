"""The short-lived bubblewrap worker for one generated search program."""

from __future__ import annotations

import argparse
import ast
import json
import resource
import sys
from pathlib import Path
from typing import Any

from .executable_memory import MemorySDK
from .memory_runtime import Evidence, MemoryValue
from .program_language import validate_program
from .utils import write_json


def _limit_resources(memory_limit_mb: int, cpu_limit_seconds: int) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_mb * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_seconds, cpu_limit_seconds + 1))


class _TracingMemory:
    """Proxy that records exact SDK call inputs and serialised outputs by line."""

    def __init__(self, memory: MemorySDK, calls: list[dict[str, Any]]):
        self._memory, self._calls = memory, calls

    def __getattr__(self, name: str):
        target = getattr(self._memory, name)
        if not callable(target):
            return target

        def wrapped(*args: Any, **kwargs: Any):
            caller = sys._getframe(1)
            output = target(*args, **kwargs)
            self._calls.append({
                "line": caller.f_lineno,
                "operation": name,
                "input": {"args": _jsonable(args), "kwargs": _jsonable(kwargs)},
                "output": _jsonable(output),
            })
            return output

        return wrapped


def execute_request(request: dict[str, Any]) -> dict[str, Any]:
    _limit_resources(int(request["memory_limit_mb"]), int(request["cpu_limit_seconds"]))
    program = str(request["program"])
    tree = validate_program(program)
    memory = MemorySDK(
        request["sample_memory_dir"],
        source_dialogue=request.get("sample") or {},
        retrieval_bundle=request.get("retrieval_bundle") or {},
    )
    calls: list[dict[str, Any]] = []
    executed_lines: list[int] = []
    namespace: dict[str, Any] = {"__builtins__": {}, "memory": _TracingMemory(memory, calls)}

    def line_tracer(frame, event, arg):
        del arg
        if frame.f_code.co_filename == "<generated-search-program>" and event == "line":
            executed_lines.append(frame.f_lineno)
        return line_tracer

    previous = sys.gettrace()
    sys.settrace(line_tracer)
    try:
        exec(compile(program, "<generated-search-program>", "exec"), namespace, namespace)
    except Exception as exc:
        _write_line_trace(request, program, tree, calls, executed_lines, namespace, error=f"{exc.__class__.__name__}: {exc}")
        return {"status": "error", "error": f"{exc.__class__.__name__}: {exc}"}
    finally:
        sys.settrace(previous)

    _write_line_trace(request, program, tree, calls, executed_lines, namespace)
    result = namespace.get("result")
    if isinstance(result, list):
        if not all(isinstance(item, Evidence) for item in result):
            return {"status": "error", "error": "result list must contain only Evidence"}
        payload: Any = [item.to_dict() for item in result]
    elif isinstance(result, Evidence):
        payload = result.to_dict()
    else:
        return {"status": "error", "error": "result must be Evidence or list[Evidence]"}
    return {"status": "ok", "result": payload}


def _write_line_trace(
    request: dict[str, Any],
    program: str,
    tree: ast.Module,
    calls: list[dict[str, Any]],
    executed_lines: list[int],
    namespace: dict[str, Any],
    error: str | None = None,
) -> None:
    lines = program.splitlines()
    calls_by_line: dict[int, list[dict[str, Any]]] = {}
    for call in calls:
        calls_by_line.setdefault(int(call["line"]), []).append(call)
    assignments = _assignment_names(tree)
    executed = set(executed_lines)
    records = []
    for number, source in enumerate(lines, start=1):
        if not source.strip() or source.lstrip().startswith("#"):
            continue
        item: dict[str, Any] = {
            "line": number,
            "source": source,
            "executed": number in executed,
        }
        if number in calls_by_line:
            item["calls"] = calls_by_line[number]
        if number in assignments and assignments[number] in namespace:
            item["assignment"] = {
                "name": assignments[number],
                "output": _jsonable(namespace[assignments[number]]),
            }
        condition = _condition_value(tree, number, namespace)
        if condition is not None:
            item["condition"] = condition
        records.append(item)
    payload = {"schema": "program_line_execution_trace_v1", "lines": records}
    if error:
        payload["error"] = error
    trace_dir = Path(str(request.get("trace_dir", "")))
    if trace_dir:
        write_json(payload, trace_dir / "line_execution_trace.json")


def _assignment_names(tree: ast.Module) -> dict[int, str]:
    output: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            output[node.lineno] = node.targets[0].id
    return output


def _condition_value(tree: ast.Module, line: int, namespace: dict[str, Any]) -> dict[str, Any] | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and node.lineno == line:
            try:
                value = eval(compile(ast.Expression(node.test), "<trace-condition>", "eval"), {"__builtins__": {}}, namespace)
                return {"value": bool(value)}
            except Exception:
                return {"value": None}
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Evidence):
        return value.to_dict()
    if isinstance(value, MemoryValue):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        output = execute_request(request)
    except Exception as exc:
        output = {"status": "error", "error": f"worker setup failed: {exc.__class__.__name__}: {exc}"}
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
