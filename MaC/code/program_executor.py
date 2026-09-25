"""Complete program planning, required retrieval validation, and sandbox execution."""

from __future__ import annotations

import json
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


PROGRAM_SCHEMA_VERSION = 9
CATEGORY_SEARCH_PROFILE_SCHEMA = "category_search_profile_v1"
DEFAULT_TIMEOUT_SECONDS = 5
DEFAULT_MEMORY_MB = 512
MAX_REPAIRS = 2


# LoCoMo's numeric labels are benchmark metadata, so preserve their original
# retrieval profiles instead of making the planner infer them from wording.
CATEGORY_SEARCH_PROFILES: dict[int, dict[str, Any]] = {
    1: {
        "name": "multi_hop",
        "default_scope": ["concept", "session"],
        "planner_guidance": (
            "This is a multi-hop, list, intersection, or count question. Expand direct code hits through both "
            "concept and session views before merging routes, so facts distributed across people, topics, and "
            "sessions remain available. Use expand_relation or select(complete=True) only when the "
            "wording actually requires a bridge, intersection, or exhaustive set."
        ),
    },
    2: {
        "name": "temporal",
        "default_scope": ["session"],
        "planner_guidance": (
            "This is a temporal question. Expand direct code hits through session views before merging routes, "
            "then hydrate to retain original dialogue wording and session timestamps. Use temporal_before only "
            "when distinct ordered events must be linked."
        ),
    },
    3: {
        "name": "open_domain",
        "default_scope": ["concept", "session"],
        "planner_guidance": (
            "This is an open-domain question. Expand direct code hits through concept and session views before "
            "merging routes, then hydrate the merged evidence. Do not add broad structural graph expansion by default."
        ),
    },
    4: {
        "name": "single_hop",
        "default_scope": [],
        "planner_guidance": (
            "This is a single-hop factual lookup. Use resolve/match/union/hydrate without default scope expansion, "
            "because unrelated facts about the same person or topic can distract from an exact attribute."
        ),
    },
}

PROGRAM_SYSTEM = """Write one complete module-level Python search program over a read-only executable-memory SDK.
Do not answer the question. Assign provenance-carrying memory Evidence to `result`.

Use the recommended default skeleton unless the question clearly does not need one or more
steps. `memory.resolve()` is the code-function route and `memory.match()` is the independent
lexical-fact route; both are recommended, but neither is mandatory for every question.
`memory.expand_scope(code, scopes=...)` is MaC2's optional concept/session expansion: it may
only consume direct `resolve()` evidence, or that evidence narrowed by `where()` or `rank()`.
Run it before `union()` or `hydrate()`, because merged/hydrated evidence no longer carries the
function locator needed for scope expansion. When two or more evidence routes are used, normally
merge them with `memory.union(...)`. `memory.hydrate(...)` is recommended before answering
because it materialises canonical facts and source windows, but direct Evidence may be returned
when already sufficient.

All other operations are question-specific choices: `expand_relation`, `select`, `where`,
`execute`, `aggregate`, `rank`, and `validate`. Use `execute(function_id)`
for compact function candidates. Never import, loop, define a function/class, access
files/network, or modify memory.

Allowed calls: resolve, expand_scope, match, union, expand_relation, rank, hydrate,
execute, select, search, search_functions, where, aggregate, validate. Every operation must
consume compatible earlier evidence, and `result` must be memory
Evidence rather than a constant, catalog record, or validation report.

STRICT SDK CALL CARD — use only these call shapes.  If no extra operation is needed, copy
the default skeleton verbatim.  `resolve()` and `match()` take ZERO arguments.
```
code = memory.resolve()
lexical = memory.match()
scoped = memory.expand_scope(code, scopes=["concept", "session"])
evidence = memory.union(code, lexical)
evidence = memory.union(evidence, scoped)
evidence = memory.where(evidence, start="YYYY-MM-DD", end="YYYY-MM-DD")
evidence = memory.rank(evidence, limit=36)
evidence = memory.expand_relation(evidence, relations=["temporal_before"])
evidence = memory.select(subject="Name", predicate="event", topics=["topic"], complete=True, limit=24)
evidence = memory.aggregate(evidence, operation="count")
report = memory.validate("question text", evidence)
result = memory.hydrate(evidence)
```
For `select`, every argument is keyword-only.  `where` supports only `start` and `end`;
it does NOT accept `predicate`.  `rank` supports only `limit`; `aggregate` supports only
`operation`; `validate` needs both the question string and evidence.  Do NOT use lambda,
comprehensions, loops, imports, `key=`, `fields=`, or any unlisted keyword.  Write the
complete program directly without prose."""

PROGRAM_TEMPLATE = """code = memory.resolve()
scoped = memory.expand_scope(code, scopes=["concept", "session"])
# RECOMMENDED_DEFAULT: retain an independent lexical route unless it adds no evidence.
lexical = memory.match()
evidence = memory.union(memory.union(code, scoped), lexical)
# OPTIONAL_QUESTION_SPECIFIC_OPERATIONS
result = memory.hydrate(evidence)"""

PROGRAM_USER = """Question:
{{question}}

Compact code candidates (index only; no copied fact text):
{{functions}}

Recommended default skeleton:
{{program_template}}

Write the complete Python program. Prefer the full default because independent code, scope, and
lexical routes usually improve recall, but omit any step that is clearly unnecessary for this
question. Scope expansion, when selected, must consume direct resolve evidence (or resolve
evidence filtered only by where/rank) before union or hydrate. If multiple routes are retained,
normally union them; hydrate before answering when the additional canonical fact/source context
is useful. Add zero or more allowed question-specific assignments only by copying a call shape
from the STRICT SDK CALL CARD.  Never invent Python filtering logic: especially never use a
lambda or `where(predicate=...)`.  If uncertain, return the supplied default skeleton unchanged.
The final `result` must be memory Evidence, never a textual answer or a catalog record."""

# Original LoCoMo planner prompts, selected for its numeric categories.
LOCOMO_PROGRAM_SYSTEM = """Write one complete module-level Python search program over a read-only executable-memory SDK.
Do not answer the question. Assign provenance-carrying memory Evidence to `result`.

Use the category-profile skeleton unless the question clearly makes an additional step
unnecessary. `memory.resolve()` is the code-function route and `memory.match()` is the
independent lexical-fact route. Recommend retaining both routes, unioning them, and hydrating
their merged evidence: this is the common recall-and-source-evidence base for every
LoCoMo category.
`memory.expand_scope(code, scopes=...)` is MaC2's optional concept/session expansion: it may
only consume direct `resolve()` evidence, or that evidence narrowed by `where()` or `rank()`.
Run it before `union()` or `hydrate()`, because merged/hydrated evidence no longer carries the
function locator needed for scope expansion. When two or more evidence routes are used, normally
merge them with `memory.union(...)`. `memory.hydrate(...)` is recommended before answering
because it materialises canonical facts and source windows, but direct Evidence may be returned
when already sufficient.

All other operations are question-specific choices: `expand_relation`, `select`, `where`,
`execute`, `aggregate`, `rank`, and `validate`. Use `execute(function_id)`
for compact function candidates. Never import, loop, define a function/class, access
files/network, or modify memory.

Allowed calls: resolve, expand_scope, match, union, expand_relation, rank, hydrate,
execute, select, search, search_functions, where, aggregate, validate. Every operation must
consume compatible earlier evidence, and `result` must be memory
Evidence rather than a constant, catalog record, or validation report. Write the complete
program directly without prose."""

LOCOMO_PROGRAM_USER = """Question:
{{question}}

Known LoCoMo category search profile:
{{category_search_profile}}

Compact code candidates (index only; no copied fact text):
{{functions}}

Recommended default skeleton:
{{program_template}}

Write the complete Python program. Prefer the full default because independent code, scope, and
lexical routes usually improve recall, but omit any step that is clearly unnecessary for this
question. Scope expansion, when selected, must consume direct resolve evidence (or resolve
evidence filtered only by where/rank) before union or hydrate. If multiple routes are retained,
normally union them; hydrate before answering when the additional canonical fact/source context
is useful. Add zero or more allowed question-specific assignments. The final `result` must be
memory Evidence, never a textual answer or a catalog record."""


def _locomo_category(category: object) -> int | None:
    value = str(category or "").strip()
    return int(value) if value in {"1", "2", "3", "4"} else None


def category_search_profile(category: object) -> dict[str, Any]:
    """Return LoCoMo's serialisable category profile, if applicable."""
    category_id = _locomo_category(category)
    if category_id is None:
        return {}
    profile = CATEGORY_SEARCH_PROFILES[category_id]
    return {
        "schema": CATEGORY_SEARCH_PROFILE_SCHEMA,
        "category": category_id,
        "name": str(profile["name"]),
        "default_scope": list(profile["default_scope"]),
        "planner_guidance": str(profile["planner_guidance"]),
        "common_recommendations": ["resolve", "match", "union", "hydrate"],
    }


def recommended_program_template(category: object = None) -> str:
    """Return the original category-aware LoCoMo fallback, or the LongMem default."""
    profile = category_search_profile(category)
    if not profile:
        return PROGRAM_TEMPLATE
    scope = profile["default_scope"]
    lines = ["code = memory.resolve()"]
    if scope:
        lines.append(f"scoped = memory.expand_scope(code, scopes={scope!r})")
    lines.append("lexical = memory.match()")
    lines.append(
        "evidence = memory.union(memory.union(code, scoped), lexical)"
        if scope else "evidence = memory.union(code, lexical)"
    )
    lines.append("result = memory.hydrate(evidence)")
    return "\n".join(lines)


def deterministic_retrieval_program(category: object = None) -> str:
    """Return the valid baseline program used only after planner exhaustion.

    Program synthesis is allowed to add question-specific calls, but a malformed
    program must never be converted into a false ``Not mentioned`` answer.  This
    baseline follows the recommended multi-route default and returns the
    resulting provenance-carrying context to the answer model.
    """

    return recommended_program_template(category)


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
    category: str = "",
    question_date: str = "",
) -> dict[str, Any]:
    """Prepare the two parallel recalls and mandatory graph/context bundle."""

    bundle = prepare_retrieval_bundle(
        sample_memory_dir,
        sample,
        question,
        llm,
        category=category,
        question_date=question_date,
    )
    functions = bundle.get("resolve", {}).get("candidate_functions", [])
    compact_functions = [_compact_function(item) for item in functions[:8]]
    final_facts = bundle.get("final", {}).get("facts", [])
    return {
        "schema": "program_candidates_v4",
        "retrieval_bundle": bundle,
        "candidate_functions": compact_functions,
        "candidate_fact_ids": [str(item.get("fact_id", "")) for item in final_facts],
        "category_search_profile": category_search_profile(category),
    }


def _compact_function(item: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "entrypoint", "subject", "predicate", "operation", "args", "search", "_cross_encoder_score")
    return {key: item.get(key) for key in keys if item.get(key) not in (None, "", [], {})}


def _program_guidance(category: str, question_date: str) -> str:
    """Return the relevant benchmark's planning hints without changing SDK semantics."""

    locomo_profile = category_search_profile(category)
    if locomo_profile:
        return (
            f"Benchmark: LoCoMo category {locomo_profile['category']} ({locomo_profile['name']})\n"
            f"Planner guidance: {locomo_profile['planner_guidance']}"
        )

    guidance = {
        "multi-session": "Use complete=True queries for counts or lists; deduplicate one repeated event before aggregate().",
        "temporal-reasoning": "Match every named event before temporal filtering; the question date is authoritative for relative-time questions.",
        "knowledge-update": "Retrieve complete value history and prefer an active-state derivation over an older mention.",
        "single-session-preference": "Retrieve durable user constraints before a recommendation.",
        "single-session-assistant": "Assistant-authored facts and visual captions are valid evidence when requested.",
        "single-session-user": "Prefer the exact user-authored fact matching all requested attributes.",
    }.get(str(category), "")
    return f"Benchmark category: {category or 'unknown'}\nQuestion date: {question_date or 'unknown'}\nPlanner guidance: {guidance}"


def plan_and_execute(
    *,
    sample_memory_dir: str | Path,
    sample: dict,
    question: str,
    llm: LLMClient,
    trace_dir: str | Path,
    candidates: dict[str, Any] | None = None,
    category: str = "",
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
            raise ProgramExecutionError(f"{label} must be inside MaC2") from exc
    ensure_dir(trace_path)
    if candidates is None:
        candidates = initial_candidates(
            sample_memory_dir,
            sample,
            question,
            llm,
            category=category,
            question_date=question_date,
        )
    bundle = dict(candidates.get("retrieval_bundle", {}))
    locomo_profile = dict(candidates.get("category_search_profile") or {})
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
            "category_search_profile": candidates.get("category_search_profile", {}),
        },
        trace_path / "planner_input.json",
    )
    prompt_template = LOCOMO_PROGRAM_USER if locomo_profile else PROGRAM_USER
    planner_system = LOCOMO_PROGRAM_SYSTEM if locomo_profile else PROGRAM_SYSTEM
    user = (
        prompt_template.replace("{{question}}", question)
        .replace("{{functions}}", json.dumps(candidates.get("candidate_functions", []), ensure_ascii=False))
        .replace("{{category_search_profile}}", json.dumps(candidates.get("category_search_profile", {}), ensure_ascii=False))
        .replace("{{program_template}}", recommended_program_template(category))
    )
    user += "\n\n" + _program_guidance(category, question_date)

    prior_exhaustion = _read_prior_planner_exhaustion(trace_path)
    if prior_exhaustion:
        # This question was previously written as a false ``Not mentioned``
        # solely because all planner repairs had failed.  Re-execute its
        # retrieval deterministically instead of repeating the same costly
        # planner failure loop.
        return _run_deterministic_fallback(
            attempts=prior_exhaustion,
            sample_memory_dir=sample_memory_dir,
            sample=sample,
            trace_path=trace_path,
            bundle=bundle,
            candidates=candidates,
            category=category,
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
            write_json({"schema": "program_attempts_v2", "attempts": attempts}, trace_path / "attempts.json")
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
            write_json({"schema": "program_attempts_v2", "attempts": attempts}, trace_path / "attempts.json")
            if attempt_number >= MAX_REPAIRS:
                return _run_deterministic_fallback(
                    attempts=attempts,
                    sample_memory_dir=sample_memory_dir,
                    sample=sample,
                    trace_path=trace_path,
                    bundle=bundle,
                    candidates=candidates,
                    category=category,
                )
            instruction = (
                user
                + "\n\nYour previous program failed validation/execution:\n"
                + str(exc)
                + "\nRewrite the complete program. Start by copying the recommended default skeleton "
                "exactly; add an optional operation only if its call shape appears verbatim in the "
                "STRICT SDK CALL CARD. Do not use lambda or invented keyword arguments."
            )
    raise AssertionError("unreachable")


def _read_prior_planner_exhaustion(trace_path: Path) -> list[dict[str, Any]]:
    """Return a legacy three-repair failure trace, if this question has one."""

    path = trace_path / "attempts.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
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
    category: object = None,
) -> dict[str, Any]:
    """Execute the fixed baseline instead of abstaining after planner failures."""

    program = deterministic_retrieval_program(category) + "\n"
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
        write_json({"schema": "program_attempts_v2", "attempts": attempts}, trace_path / "attempts.json")
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
    write_json({"schema": "program_attempts_v2", "attempts": attempts}, trace_path / "attempts.json")
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
            raise ProgramExecutionError(f"{label} must be inside MaC2") from exc
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
