"""Six-stage token and runtime accounting for Memory-as-Code on LongMemEval."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from . import benchmark
from .benchmark import LM_CATEGORIES, LM_CATEGORY_NAMES
from .utils import write_json


SUMMARY_STAGES = (
    "fact_extraction",
    "memory_build",
    "candidate_retrieval",
    "program_execution",
)
ALL_STAGES = SUMMARY_STAGES + ("answer_generation", "evaluation")

EVALUATED_CATEGORIES = LM_CATEGORIES
CATEGORY_NAMES = LM_CATEGORY_NAMES

SUMMARY_SCOPE = "memory_construction_and_retrieval"
ALL_SCOPE = "full_pipeline_including_answer_and_evaluation"
SUMMARY_FILENAME = "cost_summary.json"
ALL_FILENAME = "cost_all.json"

_INTERNAL_KEYS = (
    "token_consumption",
    "runtime_seconds",
    "request_count",
    "token_usage_complete",
    "attempts",
    "unknown_attempts",
)

_DIAGNOSTIC_KEYS = (
    "failed_attempts",
    "failed_runtime_seconds",
    "retry_backoff_seconds",
    "discarded_attempts",
    "discarded_input_tokens",
    "discarded_runtime_seconds",
    "excluded_queue_wait_seconds",
    "failed_logical_runs",
    "failed_logical_runtime_seconds",
)


@dataclass(frozen=True)
class LocalTimingSnapshot:
    excluded_queue_wait_seconds: float = 0.0


class LocalTimingAccumulator:
    """Track resource-acquisition waits inside one local pipeline stage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._excluded_queue_wait_seconds = 0.0

    def record_queue_wait(self, runtime_seconds: float) -> None:
        runtime = max(0.0, float(runtime_seconds))
        with self._lock:
            self._excluded_queue_wait_seconds += runtime

    def snapshot(self) -> LocalTimingSnapshot:
        with self._lock:
            return LocalTimingSnapshot(
                excluded_queue_wait_seconds=self._excluded_queue_wait_seconds
            )


_active_local_timing: ContextVar[LocalTimingAccumulator | None] = ContextVar(
    "active_local_timing", default=None
)


@contextmanager
def capture_local_timing():
    """Capture explicit lock/semaphore waits for one local stage."""
    accumulator = LocalTimingAccumulator()
    token = _active_local_timing.set(accumulator)
    try:
        yield accumulator
    finally:
        _active_local_timing.reset(token)


@contextmanager
def lock_excluding_queue_wait(lock: Any):
    """Acquire a lock while excluding only the time spent waiting for it."""
    if lock.acquire(blocking=False):
        waited = 0.0
    else:
        started = time.monotonic()
        lock.acquire()
        waited = time.monotonic() - started
    accumulator = _active_local_timing.get()
    if accumulator is not None:
        accumulator.record_queue_wait(waited)
    try:
        yield
    finally:
        lock.release()


def usage_metric(usage: Any, runtime_seconds: float) -> dict[str, Any]:
    """Convert accepted provider responses into a successful-only metric.

    Providers that expose ``accepted_runtime_seconds`` are timed at the actual
    request boundary. The Qwen client starts that timer only after its
    process-wide worker slot has been acquired, so outer-stage queue time is
    intentionally excluded from the published runtime.
    """
    stage_runtime = _nonnegative_float(runtime_seconds, "runtime_seconds")
    attempts = _nonnegative_int(_field(usage, "attempts", 0), "attempts")
    request_count = _nonnegative_int(
        _field(usage, "accepted_attempts", 1), "accepted_attempts"
    )
    unknown = _nonnegative_int(
        _field(usage, "unknown_attempts", 0), "unknown_attempts"
    )
    reported_complete = _field(usage, "usage_complete", unknown == 0)
    complete = bool(reported_complete) and unknown == 0
    raw_tokens = _field(
        usage,
        "input_tokens",
        _field(usage, "prompt_tokens", None),
    )
    tokens = _nonnegative_int(raw_tokens, "input_tokens") if complete else None
    diagnostics = _usage_diagnostics(usage)
    excluded_runtime = (
        diagnostics["failed_runtime_seconds"]
        + diagnostics["retry_backoff_seconds"]
        + diagnostics["discarded_runtime_seconds"]
    )
    accepted_runtime = _field(usage, "accepted_runtime_seconds", None)
    if accepted_runtime is None:
        published_runtime = max(0.0, stage_runtime - excluded_runtime)
    else:
        published_runtime = _nonnegative_float(
            accepted_runtime, "accepted_runtime_seconds"
        )
    return {
        "token_consumption": tokens,
        "runtime_seconds": published_runtime,
        "request_count": request_count,
        "token_usage_complete": complete,
        "attempts": attempts,
        "unknown_attempts": unknown,
        "logical_success": True,
        "retry_diagnostics": diagnostics,
    }


def local_metric(
    runtime_seconds: float,
    excluded_queue_wait_seconds: float = 0.0,
) -> dict[str, Any]:
    elapsed = _nonnegative_float(runtime_seconds, "runtime_seconds")
    excluded = _nonnegative_float(
        excluded_queue_wait_seconds, "excluded_queue_wait_seconds"
    )
    diagnostics = _empty_diagnostics()
    diagnostics["excluded_queue_wait_seconds"] = excluded
    return {
        "token_consumption": 0,
        "runtime_seconds": max(0.0, elapsed - excluded),
        "request_count": 0,
        "token_usage_complete": True,
        "attempts": 0,
        "unknown_attempts": 0,
        "logical_success": True,
        "retry_diagnostics": diagnostics,
    }


def failed_usage_metric(usage: Any, runtime_seconds: float) -> dict[str, Any]:
    """Keep failed logical work as diagnostics without charging the main metric."""
    _nonnegative_float(runtime_seconds, "runtime_seconds")
    attempts = _nonnegative_int(_field(usage, "attempts", 0), "attempts")
    diagnostics = _usage_diagnostics(usage)
    active_runtime = (
        _nonnegative_float(
            _field(usage, "accepted_runtime_seconds", 0.0),
            "accepted_runtime_seconds",
        )
        + float(diagnostics["failed_runtime_seconds"])
        + float(diagnostics["discarded_runtime_seconds"])
    )
    diagnostics["failed_logical_runs"] += 1
    diagnostics["failed_logical_runtime_seconds"] += active_runtime
    return {
        "token_consumption": 0,
        "runtime_seconds": 0.0,
        "request_count": 0,
        "token_usage_complete": True,
        "attempts": attempts,
        "unknown_attempts": 0,
        "logical_success": False,
        "retry_diagnostics": diagnostics,
    }


def failed_local_metric(
    runtime_seconds: float,
    excluded_queue_wait_seconds: float = 0.0,
) -> dict[str, Any]:
    runtime = _nonnegative_float(runtime_seconds, "runtime_seconds")
    excluded = _nonnegative_float(
        excluded_queue_wait_seconds, "excluded_queue_wait_seconds"
    )
    diagnostics = _empty_diagnostics()
    diagnostics["failed_logical_runs"] = 1
    diagnostics["failed_logical_runtime_seconds"] = max(0.0, runtime - excluded)
    diagnostics["excluded_queue_wait_seconds"] = excluded
    return {
        "token_consumption": 0,
        "runtime_seconds": 0.0,
        "request_count": 0,
        "token_usage_complete": True,
        "attempts": 0,
        "unknown_attempts": 0,
        "logical_success": False,
        "retry_diagnostics": diagnostics,
    }


def metric_is_complete(metric: Mapping[str, Any] | None) -> bool:
    if not isinstance(metric, Mapping) or any(key not in metric for key in _INTERNAL_KEYS):
        return False
    try:
        _nonnegative_int(metric["token_consumption"], "token_consumption")
        _nonnegative_float(metric["runtime_seconds"], "runtime_seconds")
        attempts = _nonnegative_int(metric["attempts"], "attempts")
        request_count = _nonnegative_int(metric["request_count"], "request_count")
        unknown = _nonnegative_int(metric["unknown_attempts"], "unknown_attempts")
    except (TypeError, ValueError):
        return False
    return bool(
        metric.get("logical_success", True) is True
        and
        metric["token_usage_complete"] is True
        and unknown == 0
        and unknown <= attempts
        and request_count <= attempts
    )


def merge_metrics(*metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge work while excluding failed logical runs from the main totals."""
    if not metrics:
        raise ValueError("at least one metric is required")
    structurally_valid = True
    complete = True
    tokens = 0
    runtime = 0.0
    attempts = 0
    request_count = 0
    unknown = 0
    has_success = False
    diagnostics = _empty_diagnostics()
    for metric in metrics:
        if not isinstance(metric, Mapping):
            structurally_valid = False
            complete = False
            continue
        try:
            runtime += _nonnegative_float(
                metric.get("runtime_seconds"), "runtime_seconds"
            )
            attempts += _nonnegative_int(metric.get("attempts"), "attempts")
            request_count += _nonnegative_int(
                metric.get("request_count"), "request_count"
            )
            _add_diagnostics(diagnostics, metric.get("retry_diagnostics"))
            if metric.get("logical_success", True) is True:
                has_success = True
                unknown += _nonnegative_int(
                    metric.get("unknown_attempts"), "unknown_attempts"
                )
                if metric_is_complete(metric):
                    tokens += _nonnegative_int(
                        metric.get("token_consumption"), "token_consumption"
                    )
                else:
                    complete = False
        except (TypeError, ValueError):
            structurally_valid = False
            complete = False
    exact = structurally_valid and complete and has_success
    return {
        "token_consumption": tokens if exact else None,
        "runtime_seconds": runtime if structurally_valid else None,
        "request_count": request_count,
        "token_usage_complete": exact,
        "attempts": attempts,
        "unknown_attempts": unknown,
        "logical_success": has_success,
        "retry_diagnostics": diagnostics,
    }


def _empty_diagnostics() -> dict[str, int | float]:
    return {
        key: 0.0 if key.endswith("_seconds") else 0
        for key in _DIAGNOSTIC_KEYS
    }


def _usage_diagnostics(usage: Any) -> dict[str, int | float]:
    diagnostics = _empty_diagnostics()
    for key in _DIAGNOSTIC_KEYS:
        if key in {"failed_logical_runs", "failed_logical_runtime_seconds"}:
            continue
        raw = _field(usage, key, 0)
        diagnostics[key] = (
            _nonnegative_float(raw, key)
            if key.endswith("_seconds")
            else _nonnegative_int(raw, key)
        )
    return diagnostics


def _add_diagnostics(
    target: dict[str, int | float],
    source: Mapping[str, Any] | None,
) -> None:
    if not isinstance(source, Mapping):
        return
    for key in _DIAGNOSTIC_KEYS:
        raw = source.get(key, 0)
        target[key] += (
            _nonnegative_float(raw, key)
            if key.endswith("_seconds")
            else _nonnegative_int(raw, key)
        )


def public_stage_metric(metric: Mapping[str, Any]) -> dict[str, int | float]:
    if not metric_is_complete(metric):
        raise ValueError("cannot publish an incomplete stage metric")
    return {
        "token_consumption": int(metric["token_consumption"]),
        "runtime_seconds": _round_two(float(metric["runtime_seconds"])),
        "request_count": int(metric["request_count"]),
    }


def build_cost_reports(
    records: Iterable[Mapping[str, Any]],
    *,
    model: str,
    dataset: str | None = None,
    method: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = list(records)
    dataset = benchmark.DATASET_NAME if dataset is None else dataset
    method = benchmark.METHOD_NAME if method is None else method
    return (
        _build_report(rows, SUMMARY_STAGES, SUMMARY_SCOPE, model, dataset, method),
        _build_report(rows, ALL_STAGES, ALL_SCOPE, model, dataset, method),
    )


def write_cost_reports(
    records: Iterable[Mapping[str, Any]],
    *,
    model: str,
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary, full = build_cost_reports(records, model=model)
    output = Path(output_dir) / "predictions"
    write_json(summary, output / SUMMARY_FILENAME)
    write_json(full, output / ALL_FILENAME)
    return summary, full


def _build_report(
    records: list[Mapping[str, Any]],
    stage_names: tuple[str, ...],
    scope: str,
    model: str,
    dataset: str,
    method: str,
) -> dict[str, Any]:
    complete_rows: list[dict[str, Any]] = []
    incomplete: list[str] = []
    seen: set[str] = set()
    category_question_counts = {category: 0 for category in EVALUATED_CATEGORIES}

    for raw in records:
        sample_id = str(raw.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError("every cost record requires a sample_id")
        if sample_id in seen:
            raise ValueError(f"duplicate cost record for {sample_id!r}")
        seen.add(sample_id)
        question_count = _positive_int(raw.get("question_count", 0), "question_count")
        raw_category_counts = raw.get("category_question_counts") or {}
        normalized_category_counts = {
            category: _nonnegative_int(
                raw_category_counts.get(category, 0),
                f"category_question_counts[{category}]",
            )
            for category in EVALUATED_CATEGORIES
        }
        if sum(normalized_category_counts.values()) != question_count:
            raise ValueError(
                f"category question counts do not sum to question_count for {sample_id!r}"
            )
        for category, count in normalized_category_counts.items():
            category_question_counts[category] += count

        stage_map = raw.get("stages")
        if not isinstance(stage_map, Mapping) or not all(
            metric_is_complete(stage_map.get(name)) for name in stage_names
        ):
            incomplete.append(sample_id)
            continue
        public_stages = {
            name: public_stage_metric(stage_map[name]) for name in stage_names
        }
        complete_rows.append(
            {
                "sample_id": sample_id,
                "question_count": question_count,
                "category_question_counts": normalized_category_counts,
                "stages": public_stages,
                "token_consumption": sum(
                    int(metric["token_consumption"])
                    for metric in public_stages.values()
                ),
                "runtime_seconds": sum(
                    float(metric["runtime_seconds"])
                    for metric in public_stages.values()
                ),
                "request_count": sum(
                    int(metric["request_count"])
                    for metric in public_stages.values()
                ),
                "four_stage_costs": _four_stage_costs(public_stages),
            }
        )

    samples = {
        row["sample_id"]: {
            "question_count": row["question_count"],
            "category_question_counts": row["category_question_counts"],
            "token_consumption": row["token_consumption"],
            "runtime_seconds": _round_two(row["runtime_seconds"]),
            "request_count": row["request_count"],
            "stages": row["stages"],
            "four_stage_costs": row["four_stage_costs"],
        }
        for row in complete_rows
    }
    return {
        "dataset": dataset,
        "method": method,
        "model": str(model),
        "metric_scope": scope,
        f"all_{len(EVALUATED_CATEGORIES)}_categories": _aggregate(complete_rows, stage_names),
        "four_stage_totals": _aggregate_four_stage_costs(complete_rows),
        "categories": {
            category: {
                "name": CATEGORY_NAMES[category],
                "question_count": category_question_counts[category],
            }
            for category in EVALUATED_CATEGORIES
        },
        "samples": samples,
        "incomplete_samples": incomplete,
    }


def _aggregate(
    rows: list[dict[str, Any]], stage_names: tuple[str, ...]
) -> dict[str, Any]:
    count = len(rows)
    if not rows:
        return {
            "question_count": 0,
            "sample_count": 0,
            "total_token_consumption": 0,
            "total_runtime_seconds": 0.0,
            "total_request_count": 0,
            "avg_token_consumption": 0.0,
            "avg_runtime_seconds": 0.0,
            "avg_request_count": 0.0,
            "stages": {
                name: {
                    "avg_token_consumption": 0.0,
                    "avg_runtime_seconds": 0.0,
                    "avg_request_count": 0.0,
                }
                for name in stage_names
            },
        }
    return {
        "question_count": sum(int(row["question_count"]) for row in rows),
        "sample_count": count,
        "total_token_consumption": sum(int(row["token_consumption"]) for row in rows),
        "total_runtime_seconds": _round_two(sum(float(row["runtime_seconds"]) for row in rows)),
        "total_request_count": sum(int(row["request_count"]) for row in rows),
        "avg_token_consumption": _round_two(
            sum(int(row["token_consumption"]) for row in rows) / count
        ),
        "avg_runtime_seconds": _round_two(
            sum(float(row["runtime_seconds"]) for row in rows) / count
        ),
        "avg_request_count": _round_two(
            sum(int(row["request_count"]) for row in rows) / count
        ),
        "stages": {
                name: {
                    "total_token_consumption": sum(int(row["stages"][name]["token_consumption"]) for row in rows),
                    "total_runtime_seconds": _round_two(sum(float(row["stages"][name]["runtime_seconds"]) for row in rows)),
                    "total_request_count": sum(int(row["stages"][name]["request_count"]) for row in rows),
                    "avg_token_consumption": _round_two(
                    sum(
                        int(row["stages"][name]["token_consumption"])
                        for row in rows
                    )
                    / count
                ),
                "avg_runtime_seconds": _round_two(
                    sum(
                        float(row["stages"][name]["runtime_seconds"])
                        for row in rows
                    )
                    / count
                ),
                "avg_request_count": _round_two(
                    sum(
                        int(row["stages"][name]["request_count"])
                        for row in rows
                    )
                    / count
                ),
            }
            for name in stage_names
        },
    }


def _sum_public_metrics(metrics: Iterable[Mapping[str, Any]]) -> dict[str, int | float]:
    values = list(metrics)
    return {
        "token_consumption": sum(int(metric["token_consumption"]) for metric in values),
        "runtime_seconds": _round_two(
            sum(float(metric["runtime_seconds"]) for metric in values)
        ),
        "request_count": sum(int(metric["request_count"]) for metric in values),
    }


def _four_stage_costs(stages: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, int | float]]:
    groups = {
        name: _sum_public_metrics(stages[stage] for stage in names)
        for name, names in benchmark.COST_GROUPS.items()
        if all(stage in stages for stage in names)
    }
    if "construction" in groups and "retrieval" in groups:
        groups["2_stage_total"] = _sum_public_metrics(
            (groups["construction"], groups["retrieval"])
        )
    if all(name in groups for name in ("construction", "retrieval", "answer_generation", "evaluation")):
        groups["4_stage_total"] = _sum_public_metrics(
            groups[name]
            for name in ("construction", "retrieval", "answer_generation", "evaluation")
        )
    return groups


def _aggregate_four_stage_costs(rows: list[dict[str, Any]]) -> dict[str, dict[str, int | float]]:
    if not rows:
        return {}
    names = tuple(rows[0].get("four_stage_costs", {}))
    return {
        name: _sum_public_metrics(
            row["four_stage_costs"][name]
            for row in rows
            if name in row.get("four_stage_costs", {})
        )
        for name in names
    }


def _field(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result < 1:
        raise ValueError(f"{name} must be at least 1")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if result < 0 or not math.isfinite(result):
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _round_two(value: int | float) -> float:
    return float(
        Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    )
