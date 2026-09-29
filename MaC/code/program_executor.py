"""Complete program planning, required retrieval validation, and sandbox execution."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .program_language import ProgramValidationError, validate_program
from .retrieval_pipeline import build_answer_context, prepare_retrieval_bundle
from .llm_client import LLMClient
from .utils import ensure_dir, read_json, write_json
from .prompt_contracts import QUERY_PLANNING_SYSTEM, QUERY_PLANNING_USER, QUERY_POLICY_VERSION


PROGRAM_SCHEMA_VERSION = 11
DEFAULT_TIMEOUT_SECONDS = 5
DEFAULT_MEMORY_MB = 512
MAX_REPAIRS = 2
PROGRAM_SYSTEM = QUERY_PLANNING_SYSTEM
PROGRAM_USER = QUERY_PLANNING_USER
PROGRAM_REPAIR_SUFFIX = """\n\nYour previous program failed validation/execution:
{{error}}
Rewrite the complete program for the question using the runtime interface contract.
Correct the reported error while preserving compatible evidence dependencies.
Do not use lambda or invented keyword arguments. Conditions must use direct report
indexing, such as if report["needs_source"]:, with report obtained from
memory.validate(). Use else for the opposite case, not .get(), not/and/or, or
comparisons. Ensure result is evidence on every path, and expand_scope receives
direct resolve evidence (optionally filtered by where/rank), not hydrated or merged evidence."""

# Emergency fallback only; never shown in normal planning prompts.
PROGRAM_TEMPLATE = """code = memory.resolve()
scoped = memory.expand_scope(code, scopes=["concept", "session"])
lexical = memory.match()
evidence = memory.union(memory.union(code, scoped), lexical)
result = memory.hydrate(evidence)"""


def deterministic_retrieval_program() -> str:
    """Category-independent fallback used only after planner exhaustion."""
    return PROGRAM_TEMPLATE


class ProgramExecutionError(RuntimeError):
    pass


def strip_code_fences(program: str) -> str:
    text = str(program or "").strip()
    match = re.fullmatch(r"```(?:python)?\s*(.*?)\s*```", text, flags=re.S | re.I)
    return match.group(1).strip() if match else text


def initial_candidates(
    sample_memory_dir: str | Path,
    sample: dict,
    question: str,
    llm: LLMClient,
    *,
    question_date: str = "",
) -> dict[str, Any]:
    """Prepare query-only candidate routes and their source context."""

    bundle = prepare_retrieval_bundle(
        sample_memory_dir,
        sample,
        question,
        llm,
        question_date=question_date,
    )
    functions = bundle.get("resolve", {}).get("candidate_functions", [])
    compact_functions = [_compact_function(item) for item in functions[:8]]
    final_facts = bundle.get("final", {}).get("facts", [])
    return {
        "schema": "program_candidates_v5",
        "retrieval_bundle": bundle,
        "candidate_functions": compact_functions,
        "candidate_fact_ids": [str(item.get("fact_id", "")) for item in final_facts],
    }


def _compact_function(item: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "entrypoint", "subject", "predicate", "operation", "args", "search", "_cross_encoder_score")
    return {key: item.get(key) for key in keys if item.get(key) not in (None, "", [], {})}


def _planner_fingerprint(system: str, user: str, bundle: dict, sample: dict, model: str) -> str:
    payload = {
        "policy": QUERY_POLICY_VERSION, "schema": PROGRAM_SCHEMA_VERSION,
        "system": system, "user": user, "bundle": bundle,
        "conversation": sample.get("conversation", {}), "model": model,
        "fallback": PROGRAM_TEMPLATE,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def plan_and_execute(
    *,
    sample_memory_dir: str | Path,
    sample: dict,
    question: str,
    llm: LLMClient,
    trace_dir: str | Path,
    candidates: dict[str, Any] | None = None,
    question_date: str = "",
) -> dict[str, Any]:
    """Generate a complete constrained program and execute it safely."""

    project_root = Path(__file__).resolve().parents[1]
    trace_path = Path(trace_dir).resolve()
    sample_path = Path(sample_memory_dir).resolve()
    for label, path in (("trace directory", trace_path), ("sample memory directory", sample_path)):
        try:
            path.relative_to(project_root)
        except ValueError as exc:
            raise ProgramExecutionError(f"{label} must be inside MaC") from exc
    ensure_dir(trace_path)
    if candidates is None:
        candidates = initial_candidates(
            sample_memory_dir,
            sample,
            question,
            llm,
            question_date=question_date,
        )
    bundle = dict(candidates.get("retrieval_bundle", {}))
    write_json(bundle, trace_path / "retrieval_bundle.json")
    write_json(
        {"schema": "function_index_candidates_v1", "functions": candidates.get("candidate_functions", [])},
        trace_path / "function_index_candidates.json",
    )
    write_json(
        {"schema": "match_trace_v1", **dict(bundle.get("match", {}))},
        trace_path / "match.json",
    )
    write_json(
        {"schema": "expand_scope_trace_v1", **dict(bundle.get("expand_scope", {}))},
        trace_path / "expand_scope.json",
    )
    write_json(
        {
            "schema": "program_planner_input_v4",
            "question": question,
            "candidate_functions": candidates.get("candidate_functions", []),
            "candidate_fact_ids": candidates.get("candidate_fact_ids", []),
            "query_policy": QUERY_POLICY_VERSION,
        },
        trace_path / "planner_input.json",
    )
    planner_system = PROGRAM_SYSTEM
    user = (
        PROGRAM_USER.replace("{{question}}", question)
        .replace("{{functions}}", json.dumps(candidates.get("candidate_functions", []), ensure_ascii=False))
    )
    if question_date:
        user += "\n\nQuestion date: " + question_date
    planner_fingerprint = _planner_fingerprint(
        planner_system, user, bundle, sample, str(getattr(llm, "model", ""))
    )

    prior_exhaustion = _read_prior_planner_exhaustion(trace_path, planner_fingerprint)
    if prior_exhaustion:
        # Only an exhausted attempt with identical current-policy inputs can
        # resume directly through the category-independent fallback.
        return _run_deterministic_fallback(
            attempts=prior_exhaustion,
            sample_memory_dir=sample_memory_dir,
            sample=sample,
            trace_path=trace_path,
            bundle=bundle,
            candidates=candidates,
            planner_fingerprint=planner_fingerprint,
        )

    attempts = []
    instruction = user
    for attempt_number in range(MAX_REPAIRS + 1):
        full_program = strip_code_fences(llm.chat(planner_system, instruction)) + "\n"
        body_path = trace_path / ("planner_program.py" if attempt_number == 0 else f"planner_program_repair_{attempt_number}.py")
        program_path = trace_path / ("search_program.py" if attempt_number == 0 else f"search_program_repair_{attempt_number}.py")
        body_path.write_text(full_program, encoding="utf-8")
        program_path.write_text(full_program, encoding="utf-8")
        try:
            validate_program(full_program)
            execution = run_in_sandbox(
                program=full_program,
                sample_memory_dir=sample_memory_dir,
                sample=sample,
                trace_dir=trace_path,
                retrieval_bundle=bundle,
            )
            attempts.append({"attempt": attempt_number, "program_path": program_path.name, "status": "ok"})
            write_json({"schema": "program_attempts_v3", "planner_fingerprint": planner_fingerprint, "attempts": attempts}, trace_path / "attempts.json")
            return {
                "status": "ok",
                "program": full_program,
                "planner_program": full_program,
                "attempts": attempts,
                "candidates": candidates,
                "execution": execution,
                "cost_metrics": None,
            }
        except (ProgramValidationError, ProgramExecutionError) as exc:
            attempts.append({"attempt": attempt_number, "program_path": program_path.name, "status": "failed", "error": str(exc)})
            write_json({"schema": "program_attempts_v3", "planner_fingerprint": planner_fingerprint, "attempts": attempts}, trace_path / "attempts.json")
            if attempt_number >= MAX_REPAIRS:
                return _run_deterministic_fallback(
                    attempts=attempts,
                    sample_memory_dir=sample_memory_dir,
                    sample=sample,
                    trace_path=trace_path,
                    bundle=bundle,
                    candidates=candidates,
                    planner_fingerprint=planner_fingerprint,
                )
            instruction = user + PROGRAM_REPAIR_SUFFIX.replace("{{error}}", str(exc))
    raise AssertionError("unreachable")


def _read_prior_planner_exhaustion(trace_path: Path, planner_fingerprint: str) -> list[dict[str, Any]]:
    """Reuse exhausted attempts only for the same policy and planning inputs."""

    path = trace_path / "attempts.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(payload, dict)
                or payload.get("schema") != "program_attempts_v3"
                or payload.get("planner_fingerprint") != planner_fingerprint):
            return []
        raw_attempts = payload.get("attempts", []) if isinstance(payload, dict) else []
    except (OSError, json.JSONDecodeError):
        return []
    attempts = [dict(item) for item in raw_attempts if isinstance(item, dict)]
    by_attempt = {
        item.get("attempt"): item
        for item in attempts
        if isinstance(item.get("attempt"), int)
    }
    required = tuple(range(MAX_REPAIRS + 1))
    if all(by_attempt.get(number, {}).get("status") == "failed" for number in required):
        return [by_attempt[number] for number in required]
    return []


def _run_deterministic_fallback(
    *,
    attempts: list[dict[str, Any]],
    sample_memory_dir: str | Path,
    sample: dict,
    trace_path: Path,
    bundle: dict[str, Any],
    candidates: dict[str, Any],
    planner_fingerprint: str,
) -> dict[str, Any]:
    """Execute the fixed baseline instead of abstaining after planner failures."""

    program = deterministic_retrieval_program() + "\n"
    program_path = trace_path / "search_program_deterministic_fallback.py"
    program_path.write_text(program, encoding="utf-8")
    try:
        validate_program(program)
        execution = run_in_sandbox(
            program=program,
            sample_memory_dir=sample_memory_dir,
            sample=sample,
            trace_dir=trace_path,
            retrieval_bundle=bundle,
        )
    except (ProgramValidationError, ProgramExecutionError) as exc:
        attempts.append(
            {
                "attempt": "deterministic_fallback",
                "program_path": program_path.name,
                "status": "failed",
                "error": str(exc),
            }
        )
        write_json({"schema": "program_attempts_v3", "planner_fingerprint": planner_fingerprint, "attempts": attempts}, trace_path / "attempts.json")
        raise ProgramExecutionError(
            "deterministic fallback failed after planner exhaustion"
        ) from exc

    attempts.append(
        {
            "attempt": "deterministic_fallback",
            "program_path": program_path.name,
            "status": "ok",
            "reason": "planner_generation_failed",
        }
    )
    write_json({"schema": "program_attempts_v3", "planner_fingerprint": planner_fingerprint, "attempts": attempts}, trace_path / "attempts.json")
    return {
        "status": "ok",
        "program": program,
        "planner_program": None,
        "fallback": {
            "used": True,
            "reason": "planner_generation_failed",
            "program_path": program_path.name,
        },
        "attempts": attempts,
        "candidates": candidates,
        "execution": execution,
        "cost_metrics": None,
    }


def run_in_sandbox(
    *,
    program: str,
    sample_memory_dir: str | Path,
    sample: dict,
    trace_dir: str | Path,
    retrieval_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise ProgramExecutionError("bubblewrap is required for generated program execution")
    project_root = Path(__file__).resolve().parents[1]
    sample_dir, trace_path = Path(sample_memory_dir).resolve(), Path(trace_dir).resolve()
    for label, path in (("sample memory directory", sample_dir), ("trace directory", trace_path)):
        try:
            path.relative_to(project_root)
        except ValueError as exc:
            raise ProgramExecutionError(f"{label} must be inside MaC") from exc
    ensure_dir(trace_path)
    request_path = trace_path / "sandbox_request.json"
    write_json(
        {
            "program": program,
            "sample_memory_dir": str(sample_dir),
            "sample": sample,
            "retrieval_bundle": retrieval_bundle or {},
            "trace_dir": str(trace_path),
            "memory_limit_mb": _positive_env("MAC_SANDBOX_MEMORY_MB", DEFAULT_MEMORY_MB),
            "cpu_limit_seconds": _positive_env("MAC_SANDBOX_CPU_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        },
        request_path,
    )
    python = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable
    command = [
        bwrap, "--die-with-parent", "--unshare-user", "--unshare-pid", "--unshare-net",
        "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", str(project_root), str(project_root), "--bind", str(trace_path), str(trace_path),
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--chdir", str(project_root),
        python, "-m", "code.program_worker", "--request", str(request_path),
    ]
    timeout = _positive_env("MAC_SANDBOX_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    started = time.monotonic()
    try:
        process = subprocess.run(command, cwd=project_root, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ProgramExecutionError(f"sandbox timed out after {timeout}s") from exc
    runtime = time.monotonic() - started
    if process.returncode != 0:
        write_json({"schema": "sandbox_failure_v1", "exit_code": process.returncode, "stdout": process.stdout, "stderr": process.stderr, "runtime_seconds": runtime}, trace_path / "sandbox_failure.json")
        raise ProgramExecutionError(f"sandbox failed (exit={process.returncode}): {(process.stderr or process.stdout).strip()[:800]}")
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise ProgramExecutionError(f"sandbox produced invalid JSON: {process.stdout[:400]!r}") from exc
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        write_json(payload if isinstance(payload, dict) else {"raw": payload}, trace_path / "sandbox_result.json")
        error = payload.get("error", "sandbox returned invalid payload") if isinstance(payload, dict) else "sandbox returned invalid payload"
        raise ProgramExecutionError(str(error))
    payload["runtime_seconds"] = runtime
    payload["answer_trace"] = _answer_safe_trace(trace_path, payload.get("result"))
    write_json(payload["answer_trace"], trace_path / "answer_context_manifest.json")
    write_json(payload, trace_path / "sandbox_result.json")
    return payload


def render_evidence_context(
    execution: dict[str, Any],
    retrieval_bundle: dict[str, Any],
    sample_memory_dir: str | Path | None = None,
) -> str:
    canonical_facts = {}
    if sample_memory_dir is not None:
        payload = read_json(Path(sample_memory_dir) / "facts_by_id.json")
        if isinstance(payload, dict):
            canonical_facts = {str(key): dict(value) for key, value in payload.items() if isinstance(value, dict)}
    return build_answer_context(
        retrieval_bundle,
        execution.get("result"),
        program_trace=execution.get("answer_trace"),
        canonical_facts_by_id=canonical_facts,
    )


def _answer_safe_trace(trace_path: Path, program_result: Any) -> dict[str, Any]:
    """Expose execution provenance to the answer model without derived values.

    The complete line-level inputs and outputs remain in
    ``line_execution_trace.json`` for auditing.  This compact projection is the
    only trace passed to the answer model, preventing an incomplete aggregate or
    a narrowly-filtered result from being mistaken for a final answer.
    """

    operations: list[str] = []
    try:
        line_trace = json.loads((trace_path / "line_execution_trace.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        line_trace = {}
    for line in line_trace.get("lines", []) if isinstance(line_trace, dict) else []:
        if not isinstance(line, dict):
            continue
        for call in line.get("calls", []):
            operation = str(call.get("operation", "")).strip() if isinstance(call, dict) else ""
            if operation and operation not in operations:
                operations.append(operation)
    return {
        "schema": "answer_safe_program_trace_v1",
        "operations": operations,
        "result_provenance": _evidence_provenance(program_result),
    }


def _evidence_provenance(value: Any) -> dict[str, list[str]]:
    """Collect only stable provenance fields; deliberately omit ``value``."""

    fields = {"fact_ids": [], "dialogue_ids": [], "rules": []}

    def collect(item: Any) -> None:
        if isinstance(item, dict):
            for field in fields:
                raw = item.get(field, [])
                values = raw if isinstance(raw, list) else [raw]
                for candidate in values:
                    text = str(candidate or "").strip()
                    if text and text not in fields[field]:
                        fields[field].append(text)
        elif isinstance(item, list):
            for child in item:
                collect(child)

    collect(value)
    return fields


def _positive_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default
