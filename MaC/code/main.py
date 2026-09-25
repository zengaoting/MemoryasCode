from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Callable

from openai import APITimeoutError
from tqdm import tqdm

from . import llm_client
from .benchmark import LM_CATEGORIES, LM_CATEGORY_NAMES
from .config import Config, load_config
from .cost_metrics import local_metric, merge_metrics, metric_is_complete, write_cost_reports
from .data_loader import load_dataset
from .evaluator import evaluate_predictions
from .llm_client import LLMClient, OpenAIJudgeClient, create_llm_client
from .memory_extractor import (
    extract_atomic_facts_for_sample,
    extractor_configuration_fingerprint,
)
from .memory_builder import build_canonical_fact_artifacts
from .semantic_compiler import (
    compile_semantic_specs,
    emit_compiled_memory,
    replay_cached_semantic_cost,
    semantic_compiler_fingerprint,
)
from .qa_engine import prediction_key, prediction_fingerprint, read_completed_prediction_fingerprints, read_stage_metric_for_cost, run_qa_question, stage_checkpoint_path, stage_fingerprint_from_prediction
from .utils import append_jsonl, ensure_dir, read_json, read_jsonl, write_json, write_jsonl


MEMORY_BUILD_DONE_FILE = "build.done.json"
MEMORY_BUILDER_VERSION = 13
DEFERRED_QUESTIONS_SCHEMA = "deferred_questions_v1"
QUESTION_EXECUTION_STAGES = (
    "candidate_retrieval",
    "program_execution",
    "answer_generation",
)


_logging_initialized = False


def setup_logging(log_file: str = "logs/run_our_v1.log") -> None:
    global _logging_initialized
    if _logging_initialized:
        return
    _logging_initialized = True

    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = Path(__file__).resolve().parents[1] / log_path
    ensure_dir(log_path.parent)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )

    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    pkg = logging.getLogger("code")
    pkg.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logging.getLogger().addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logging.getLogger().addHandler(sh)


def _thread_local_client_getter(
    api_key: str,
    base_url: str,
    model: str,
    *,
    provider: str,
    request_timeout: float | None = None,
    max_api_retries: int | None = None,
) -> Callable[[], LLMClient]:
    state = threading.local()

    def get_client() -> LLMClient:
        client = getattr(state, "client", None)
        if client is None:
            client = create_llm_client(
                api_key,
                base_url,
                model,
                provider=provider,
                request_timeout=request_timeout,
                max_api_retries=max_api_retries,
            )
            state.client = client
        return client

    return get_client


def _sample_specs(samples: list[dict], output_dir: str) -> list[tuple[dict, str, str]]:
    """Resolve IDs once and prevent two workers from sharing a sample directory."""
    specs: list[tuple[dict, str, str]] = []
    seen_ids: set[str] = set()
    for sample_index, sample in enumerate(samples):
        sample_id = str(sample.get("sample_id") or f"sample_{sample_index:03d}").strip()
        if not sample_id or not all(ch.isalnum() or ch in "_-" for ch in sample_id):
            raise ValueError(f"sample {sample_index} has unsafe sample_id {sample_id!r}")
        if sample_id in seen_ids:
            raise ValueError(
                f"Duplicate sample_id {sample_id!r}; parallel workers cannot share an output directory"
            )
        seen_ids.add(sample_id)
        specs.append((sample, sample_id, str(Path(output_dir) / sample_id)))
    return specs


def _selected_samples(config: Config) -> list[dict]:
    samples = load_dataset(config.data_path)
    if config.dataset == "locomo" or config.samples_per_category is None:
        return samples[: config.max_samples] if config.max_samples is not None else samples

    selected_ids: set[str] = set()
    counts = {category: 0 for category in LM_CATEGORIES}
    for sample in samples:
        metadata = sample.get("metadata") or {}
        category = str(metadata.get("question_type") or "")
        sample_id = str(sample.get("sample_id") or "").strip()
        if category not in counts or not sample_id or counts[category] >= config.samples_per_category:
            continue
        counts[category] += 1
        selected_ids.add(sample_id)
    missing = [category for category, count in counts.items() if count < config.samples_per_category]
    if missing:
        raise ValueError("dataset lacks enough samples for: " + ", ".join(missing))
    return [sample for sample in samples if str(sample.get("sample_id") or "") in selected_ids]


def _selected_qa_items(sample: dict, config: Config) -> list[dict]:
    qa_items = list(sample.get("qa", []))
    if config.max_questions_per_sample is not None:
        qa_items = qa_items[: config.max_questions_per_sample]
    return qa_items


def _evaluated_qa_items(sample: dict, config: Config) -> list[tuple[int, dict]]:
    return [
        (question_id, qa)
        for question_id, qa in enumerate(_selected_qa_items(sample, config))
        if str(qa.get("category") or "") in LM_CATEGORIES
    ]


def _round_robin_qa_tasks(tasks: list[tuple]) -> list[tuple]:
    """Interleave samples so every sample advances during a parallel QA run."""
    buckets: dict[str, deque] = {}
    sample_order: list[str] = []
    for task in tasks:
        sample_id = str(task[1])
        if sample_id not in buckets:
            buckets[sample_id] = deque()
            sample_order.append(sample_id)
        buckets[sample_id].append(task)

    interleaved: list[tuple] = []
    while True:
        added = False
        for sample_id in sample_order:
            if buckets[sample_id]:
                interleaved.append(buckets[sample_id].popleft())
                added = True
        if not added:
            return interleaved


def _collect_cost_records(config: Config, samples: list[dict]) -> list[dict]:
    records = []
    question_stages = (
        "candidate_retrieval",
        "program_execution",
        "answer_generation",
        "evaluation",
    )
    for sample, sample_id, sample_memory_dir in _sample_specs(samples, config.output_dir):
        qa_items = _evaluated_qa_items(sample, config)
        if not qa_items:
            continue
        category_question_counts = {category: 0 for category in LM_CATEGORIES}
        for _, qa in qa_items:
            category_question_counts[str(qa.get("category"))] += 1
        stage_costs: dict[str, dict | None] = {
            "fact_extraction": None,
            "memory_build": None,
        }
        build_path = Path(sample_memory_dir) / MEMORY_BUILD_DONE_FILE
        try:
            if _memory_build_complete(
                sample_memory_dir,
                sample_id,
                expected_input_sha256=_sample_build_fingerprint(sample, config.model),
            ):
                build_done = read_json(build_path)
                saved_stages = build_done.get("stage_costs") or {}
                stage_costs["fact_extraction"] = saved_stages.get("fact_extraction")
                stage_costs["memory_build"] = saved_stages.get("memory_build")
        except Exception:
            logging.warning("Unable to read construction cost checkpoint for %s", sample_id)

        per_question: dict[str, list[dict | None]] = {
            stage: [] for stage in question_stages
        }
        for question_id, qa in qa_items:
            try:
                prediction_sha256 = prediction_fingerprint(
                    sample,
                    qa,
                    sample_memory_dir,
                    config.model,
                )
                for stage in question_stages:
                    fingerprint = stage_fingerprint_from_prediction(
                        stage=stage,
                        prediction_sha256=prediction_sha256,
                        llm_model=config.model,
                        judge_model=(config.judge_model if stage == "evaluation" else ""),
                    )
                    per_question[stage].append(
                        read_stage_metric_for_cost(
                            sample_memory_dir,
                            sample_id,
                            question_id,
                            stage,
                            fingerprint,
                        )
                    )
            except Exception:
                for stage in question_stages:
                    per_question[stage].append(None)

        for stage, metrics in per_question.items():
            stage_costs[stage] = merge_metrics(*metrics)
        records.append(
            {
                "sample_id": sample_id,
                "question_count": len(qa_items),
                "category_question_counts": category_question_counts,
                "stages": stage_costs,
            }
        )
    return records


def _write_current_cost_reports(config: Config, samples: list[dict]) -> tuple[dict, dict]:
    return _write_cost_record_list(config, _collect_cost_records(config, samples))


def _write_cost_record_list(config: Config, records: list[dict]) -> tuple[dict, dict]:
    return write_cost_reports(
        records,
        model=config.model,
        output_dir=config.output_dir,
    )


def _prepare_memory_for_semantic_stage(
    sample: dict,
    sample_id: str,
    sample_memory_dir: str,
    llm: LLMClient,
    force: bool,
) -> int:
    """Stage A: extract and canonicalise facts without semantic LLM calls."""
    extract_atomic_facts_for_sample(
        sample,
        sample_id,
        llm,
        sample_memory_dir,
        force=force,
    )
    sample_dir = Path(sample_memory_dir)
    inflight_path = sample_dir / "memory_build.inflight.json"
    if force:
        inflight_path.unlink(missing_ok=True)
    write_json(
        {
            "schema": "memory_build_inflight_v2",
            "sample_id": sample_id,
            "input_sha256": _sample_build_fingerprint(sample, llm.model),
            "phase": "prepare",
        },
        inflight_path,
    )
    try:
        atomic_facts = read_jsonl(sample_dir / "atomic_facts.jsonl")
        _remove_legacy_memory_views(sample_dir)
        canonical_started = time.monotonic()
        facts = build_canonical_fact_artifacts(atomic_facts, sample_dir)
        canonical_runtime_seconds = time.monotonic() - canonical_started
    except BaseException:
        write_json(
            {
                "schema": "memory_build_inflight_v2",
                "sample_id": sample_id,
                "input_sha256": _sample_build_fingerprint(sample, llm.model),
                "phase": "prepare_failed",
            },
            inflight_path,
        )
        raise
    write_json(
        {
            "schema": "memory_build_inflight_v2",
            "sample_id": sample_id,
            "input_sha256": _sample_build_fingerprint(sample, llm.model),
            "phase": "semantic",
            "canonical_runtime_seconds": canonical_runtime_seconds,
        },
        inflight_path,
    )
    return len(facts)


def _compile_semantics_for_sample(
    sample_id: str,
    sample_memory_dir: str,
    llm: LLMClient,
    force: bool,
) -> dict:
    """Stage B: bounded remote semantic parsing and state linking."""
    return compile_semantic_specs(
        sample_id=sample_id,
        llm=llm,
        sample_memory_dir=sample_memory_dir,
        force=force,
    )


def _emit_memory_for_sample(
    sample: dict,
    sample_id: str,
    sample_memory_dir: str,
    model: str,
) -> int:
    """Stage C: local Python emission and final construction checkpoint."""
    sample_dir = Path(sample_memory_dir)
    runtime = emit_compiled_memory(sample_id=sample_id, sample_memory_dir=sample_dir)
    semantic_payload = read_json(sample_dir / "semantic_specs.json")
    semantic_metric = semantic_payload.get("cost_metrics") if isinstance(semantic_payload, dict) else None
    atomic_done = read_json(sample_dir / "atomic_facts.done.json")
    fact_extraction_cost = (atomic_done.get("stage_costs") or {}).get("fact_extraction")
    facts = read_json(sample_dir / "facts_by_id.json")
    fact_count = len(facts) if isinstance(facts, dict) else 0
    try:
        inflight = read_json(sample_dir / "memory_build.inflight.json")
    except Exception:
        inflight = {}
    canonical_runtime = float(inflight.get("canonical_runtime_seconds", 0.0)) if isinstance(inflight, dict) else 0.0
    metrics = [local_metric(canonical_runtime + float(runtime.get("local_runtime_seconds", 0.0)))]
    if isinstance(semantic_metric, dict):
        metrics.insert(0, semantic_metric)
    memory_build_cost = merge_metrics(*metrics)
    facts_path = sample_dir / "atomic_facts.jsonl"
    _write_memory_build_done(
        str(sample_dir),
        sample_id,
        fact_count,
        _sample_build_fingerprint(sample, model),
        _file_sha256(facts_path),
        runtime,
        {
            "fact_extraction": fact_extraction_cost,
            "memory_build": memory_build_cost,
        },
    )
    (sample_dir / "memory_build.inflight.json").unlink(missing_ok=True)
    return fact_count


def cmd_build_memory(config: Config, force: bool = False) -> None:
    config.output_dir = _scoped_output_path(config.output_dir)
    setup_logging()
    logging.info(
        "Starting build-memory | data=%s | model=%s | max_samples=%s | "
        "extraction_workers=%d | semantic_workers=%d",
        config.data_path,
        config.model,
        config.max_samples,
        config.build_max_workers,
        config.semantic_max_workers,
    )
    samples = _selected_samples(config)

    specs = _sample_specs(samples, config.output_dir)
    pending_specs = []
    semantic_ready_specs = []
    for sample, sample_id, sample_memory_dir in specs:
        input_sha256 = _sample_build_fingerprint(sample, config.model)
        if not force and _memory_build_complete(
            sample_memory_dir,
            sample_id,
            expected_input_sha256=input_sha256,
        ):
            logging.info("Memory build complete for %s; skipping", sample_id)
            continue
        if not force and _semantic_stage_prepared(
            sample_memory_dir,
            sample_id,
            expected_input_sha256=input_sha256,
        ):
            # Canonical facts already exist on disk.  A prior run was stopped
            # after Stage A, so resume directly at the first missing remote
            # semantic operation rather than repeating extraction/dedup.
            semantic_ready_specs.append((sample, sample_id, sample_memory_dir))
            logging.info("Canonical preparation complete for %s; resuming at semantic stage", sample_id)
            continue
        pending_specs.append((sample, sample_id, sample_memory_dir))

    if not pending_specs and not semantic_ready_specs:
        logging.info("All selected memory builds are already complete.")
        _write_current_cost_reports(config, samples)
        return

    get_thread_client = _thread_local_client_getter(
        config.api_key,
        config.base_url,
        config.model,
        provider=config.llm_provider,
    )
    get_semantic_client = _thread_local_client_getter(
        config.api_key,
        config.base_url,
        config.model,
        provider=config.llm_provider,
        request_timeout=config.semantic_request_timeout,
        max_api_retries=config.semantic_api_max_retries,
    )

    failures: list[tuple[str, Exception]] = []

    # The three pools form a streaming construction pipeline.  In particular,
    # a blocked semantic request never occupies an extraction worker, and an
    # emitted sample does not wait for every other sample to finish semantics.
    def prepare_one(spec: tuple[dict, str, str]) -> int:
        sample, sample_id, sample_memory_dir = spec
        return _prepare_memory_for_semantic_stage(
            sample,
            sample_id,
            sample_memory_dir,
            get_thread_client(),
            force,
        )

    def compile_one(spec: tuple[dict, str, str], semantic_force: bool) -> dict:
        _, sample_id, sample_memory_dir = spec
        return _compile_semantics_for_sample(
            sample_id,
            sample_memory_dir,
            get_semantic_client(),
            semantic_force,
        )

    total_pending = len(pending_specs) + len(semantic_ready_specs)
    extraction_progress = tqdm(total=len(pending_specs), desc="Extracting memories")
    semantic_progress = tqdm(total=total_pending, desc="Compiling semantics")
    emission_progress = tqdm(total=total_pending, desc="Emitting memory code")
    deferred_specs: list[tuple[dict, str, str]] = []
    retry_started = False
    try:
        with (
            ThreadPoolExecutor(
                max_workers=config.build_max_workers,
                thread_name_prefix="memory-extract",
            ) as extraction_executor,
            ThreadPoolExecutor(
                max_workers=config.semantic_max_workers,
                thread_name_prefix="memory-semantic",
            ) as semantic_executor,
            ThreadPoolExecutor(
                max_workers=config.build_max_workers,
                thread_name_prefix="memory-emit",
            ) as emission_executor,
        ):
            extraction_futures = {
                extraction_executor.submit(prepare_one, spec): spec
                for spec in pending_specs
            }
            semantic_futures: dict = {}
            emission_futures: dict = {}

            def submit_semantic(spec: tuple[dict, str, str], *, retry: bool) -> None:
                semantic_futures[
                    semantic_executor.submit(compile_one, spec, force and not retry)
                ] = (spec, retry)

            for spec in semantic_ready_specs:
                submit_semantic(spec, retry=False)

            while extraction_futures or semantic_futures or emission_futures or deferred_specs:
                if not (extraction_futures or semantic_futures or emission_futures):
                    # Normal semantic work has drained.  Retry only now, so a
                    # slow/failed sample cannot jump ahead of untried samples.
                    if retry_started:
                        break
                    retry_started = True
                    logging.info("Retrying %d deferred semantic build(s)", len(deferred_specs))
                    for spec in deferred_specs:
                        submit_semantic(spec, retry=True)
                    deferred_specs = []
                    continue

                active = list(extraction_futures) + list(semantic_futures) + list(emission_futures)
                completed, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in completed:
                    if future in extraction_futures:
                        spec = extraction_futures.pop(future)
                        _, sample_id, _ = spec
                        extraction_progress.update(1)
                        try:
                            fact_count = future.result()
                            logging.info(
                                "Extraction/canonicalisation complete for %s (%d facts)",
                                sample_id,
                                fact_count,
                            )
                            submit_semantic(spec, retry=False)
                        except Exception as exc:
                            logging.error(
                                "Memory extraction failed for %s; completed samples remain resumable.",
                                sample_id,
                                exc_info=(type(exc), exc, exc.__traceback__),
                            )
                            failures.append((sample_id, exc))
                    elif future in semantic_futures:
                        spec, retry = semantic_futures.pop(future)
                        sample, sample_id, sample_memory_dir = spec
                        try:
                            future.result()
                            semantic_progress.update(1)
                            logging.info("Semantic compilation complete for %s", sample_id)
                            emission_futures[
                                emission_executor.submit(
                                    _emit_memory_for_sample,
                                    sample,
                                    sample_id,
                                    sample_memory_dir,
                                    config.model,
                                )
                            ] = spec
                        except Exception as exc:
                            if retry:
                                failures.append((sample_id, exc))
                                logging.error(
                                    "Semantic compilation failed after deferred retry for %s",
                                    sample_id,
                                    exc_info=(type(exc), exc, exc.__traceback__),
                                )
                            else:
                                deferred_specs.append(spec)
                                logging.warning(
                                    "Deferring semantic compilation for %s until normal work drains: %s",
                                    sample_id,
                                    exc,
                                )
                    else:
                        spec = emission_futures.pop(future)
                        _, sample_id, _ = spec
                        emission_progress.update(1)
                        try:
                            fact_count = future.result()
                            logging.info("Memory build complete for %s (%d facts)", sample_id, fact_count)
                        except Exception as exc:
                            logging.error(
                                "Memory code emission failed for %s; semantic checkpoints remain resumable.",
                                sample_id,
                                exc_info=(type(exc), exc, exc.__traceback__),
                            )
                            failures.append((sample_id, exc))
    finally:
        extraction_progress.close()
        semantic_progress.close()
        emission_progress.close()

    if failures:
        _write_current_cost_reports(config, samples)
        first_sample_id, first_exc = failures[0]
        raise RuntimeError(
            f"{len(failures)} memory build(s) failed; rerun to resume. "
            f"First failed sample: {first_sample_id}; error: {first_exc}"
        ) from first_exc
    _write_current_cost_reports(config, samples)


def _backfill_memory_build_cost_for_sample(
    *, sample_id: str, sample_memory_dir: str, llm: LLMClient,
) -> bool:
    """Replay missing semantic-build accounting without changing memory outputs."""
    build_path = Path(sample_memory_dir) / MEMORY_BUILD_DONE_FILE
    payload = read_json(build_path)
    stage_costs = payload.get("stage_costs")
    if not isinstance(stage_costs, dict):
        stage_costs = {}
    existing_metric = stage_costs.get("memory_build")
    if metric_is_complete(existing_metric):
        return False

    replay_metric = replay_cached_semantic_cost(
        sample_id=sample_id,
        llm=llm,
        sample_memory_dir=sample_memory_dir,
    )
    stage_costs["memory_build"] = replay_metric
    payload["stage_costs"] = stage_costs
    payload["memory_build_cost_provenance"] = {
        "mode": "replayed_from_persisted_semantic_inputs_v1",
        "changes_memory_artifacts": False,
    }
    write_json(payload, build_path)
    return True


def cmd_backfill_construction_cost(config: Config) -> None:
    """Complete legacy semantic-build cost checkpoints, then regenerate reports."""
    config.output_dir = _scoped_output_path(config.output_dir)
    setup_logging()
    samples = _selected_samples(config)
    specs = _sample_specs(samples, config.output_dir)
    pending = []
    for _, sample_id, sample_memory_dir in specs:
        build_path = Path(sample_memory_dir) / MEMORY_BUILD_DONE_FILE
        if not build_path.exists():
            raise FileNotFoundError(
                f"cannot backfill construction cost before memory build: {sample_id}"
            )
        try:
            build_payload = read_json(build_path)
            metric = (build_payload.get("stage_costs") or {}).get("memory_build")
        except Exception as exc:
            raise RuntimeError(f"cannot read construction checkpoint for {sample_id}") from exc
        if not metric_is_complete(metric):
            pending.append((sample_id, sample_memory_dir))

    if not pending:
        logging.info("All construction cost checkpoints are already complete.")
        _write_current_cost_reports(config, samples)
        return

    logging.info(
        "Replaying semantic/state-link accounting for %d sample(s) without modifying memory artifacts.",
        len(pending),
    )
    get_thread_client = _thread_local_client_getter(
        config.api_key, config.base_url, config.model, provider=config.llm_provider
    )
    failures: list[tuple[str, BaseException]] = []
    with ThreadPoolExecutor(max_workers=config.build_max_workers) as executor:
        def backfill_one(sample_id: str, sample_memory_dir: str) -> bool:
            # The getter must run in the worker thread.  Calling it while
            # submitting futures would accidentally share one HTTP client
            # between workers and undermine the configured concurrency.
            return _backfill_memory_build_cost_for_sample(
                sample_id=sample_id,
                sample_memory_dir=sample_memory_dir,
                llm=get_thread_client(),
            )

        futures = {
            executor.submit(
                backfill_one,
                sample_id,
                sample_memory_dir,
            ): sample_id
            for sample_id, sample_memory_dir in pending
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                future.result()
                logging.info("Backfilled semantic construction cost for %s", sample_id)
            except BaseException as exc:
                failures.append((sample_id, exc))
                logging.exception("Construction-cost backfill failed for %s", sample_id)

    _write_current_cost_reports(config, samples)
    if failures:
        sample_id, exc = failures[0]
        raise RuntimeError(
            f"{len(failures)} construction cost backfill(s) failed; first: {sample_id}: {exc}"
        ) from exc


def _prediction_path(config: Config, pred_suffix: str = "") -> str:
    pred_dir = Path(config.output_dir) / "predictions"
    ensure_dir(pred_dir)
    return str(pred_dir / f"predictions{pred_suffix}.jsonl")


def _memory_build_complete(
    sample_memory_dir: str,
    sample_id: str,
    expected_input_sha256: str | None = None,
) -> bool:
    sample_dir = Path(sample_memory_dir)
    required = [
        sample_dir / "atomic_facts.jsonl",
        sample_dir / "atomic_facts.done.json",
        sample_dir / "facts_by_id.json",
        sample_dir / "fact_graph.json",
        sample_dir / "semantic_specs.json",
        sample_dir / "function_index.json",
        sample_dir / "view_index.json",
        sample_dir / "memory_code" / "facts" / "__init__.py",
        sample_dir / "memory_code" / "concept" / "__init__.py",
        sample_dir / "memory_code" / "sessions" / "__init__.py",
        sample_dir / "memory_code" / "relations.py",
        sample_dir / MEMORY_BUILD_DONE_FILE,
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        done = read_json(sample_dir / MEMORY_BUILD_DONE_FILE)
        facts_sha256 = _file_sha256(sample_dir / "atomic_facts.jsonl")
        facts_by_id_sha256 = _file_sha256(sample_dir / "facts_by_id.json")
        fact_graph_sha256 = _file_sha256(sample_dir / "fact_graph.json")
        function_index_sha256 = _file_sha256(sample_dir / "function_index.json")
        view_index_sha256 = _file_sha256(sample_dir / "view_index.json")
        memory_code_sha256 = _hash_memory_code(sample_dir / "memory_code")
    except Exception:
        return False
    fact_count = done.get("fact_count")
    stage_costs = done.get("stage_costs")
    return bool(
        done.get("sample_id") == sample_id
        and done.get("schema") == "memory_build_done_v13"
        and done.get("completed") is True
        and done.get("builder_version") == MEMORY_BUILDER_VERSION
        and (
            expected_input_sha256 is None
            or done.get("input_sha256") == expected_input_sha256
            # The opt-in resume mode deliberately ignores LLM/model/prompt
            # changes, while retaining every materialized-artifact hash check
            # below.  Therefore it cannot treat a changed fact store, graph,
            # index, or generated code as complete.
            or _resume_ignores_llm_configuration()
        )
        # ``atomic_facts.jsonl`` is an extraction-stage cache; it is not read
        # by retrieval.  In direct-resume mode it may have been refreshed by
        # an earlier interrupted run while the complete runtime artifacts
        # below are still mutually consistent.  Do not rebuild those runtime
        # artifacts merely for that cache difference.
        and (
            done.get("facts_sha256") == facts_sha256
            or _resume_ignores_llm_configuration()
        )
        and done.get("facts_by_id_sha256") == facts_by_id_sha256
        and done.get("fact_graph_sha256") == fact_graph_sha256
        and done.get("function_index_sha256") == function_index_sha256
        and done.get("view_index_sha256") == view_index_sha256
        and done.get("memory_code_sha256") == memory_code_sha256
        and isinstance(fact_count, int)
        and fact_count >= 0
        and isinstance(stage_costs, dict)
        and isinstance(stage_costs.get("fact_extraction"), dict)
        and isinstance(stage_costs.get("memory_build"), dict)
    )


def _semantic_stage_prepared(
    sample_memory_dir: str,
    sample_id: str,
    *,
    expected_input_sha256: str | None = None,
) -> bool:
    """Return whether Stage A completed and Stage B is the next safe stage.

    ``memory_build.inflight.json`` is written only after atomic extraction and
    canonical fact construction succeeded.  Treat it as a real resumable
    boundary: a restart must not repeat those completed operations merely
    because semantic compilation has not yet finished.
    """
    sample_dir = Path(sample_memory_dir)
    required = (
        sample_dir / "atomic_facts.jsonl",
        sample_dir / "atomic_facts.done.json",
        sample_dir / "facts_by_id.json",
        sample_dir / "fact_graph.json",
        sample_dir / "memory_build.inflight.json",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        payload = read_json(sample_dir / "memory_build.inflight.json")
    except Exception:
        return False
    return bool(
        isinstance(payload, dict)
        and payload.get("schema") == "memory_build_inflight_v2"
        and payload.get("sample_id") == sample_id
        and payload.get("phase") == "semantic"
        and (
            expected_input_sha256 is None
            or payload.get("input_sha256") == expected_input_sha256
            or _resume_ignores_llm_configuration()
        )
    )


def _resume_ignores_llm_configuration() -> bool:
    """Opt-in direct resume across reviewed LLM configuration changes."""
    return os.getenv("OUR_V1_RESUME_IGNORE_LLM_CONFIG", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _write_memory_build_done(
    sample_memory_dir: str,
    sample_id: str,
    fact_count: int,
    input_sha256: str,
    facts_sha256: str,
    runtime: dict,
    stage_costs: dict,
) -> None:
    write_json(
        {
            "schema": "memory_build_done_v13",
            "builder_version": MEMORY_BUILDER_VERSION,
            "sample_id": sample_id,
            "completed": True,
            "fact_count": fact_count,
            "input_sha256": input_sha256,
            "facts_sha256": facts_sha256,
            "facts_by_id_sha256": runtime.get("facts_by_id_sha256"),
            "fact_graph_sha256": runtime.get("fact_graph_sha256"),
            "function_index_sha256": runtime.get("function_index_sha256"),
            "view_index_sha256": runtime.get("view_index_sha256"),
            "memory_code_sha256": runtime.get("memory_code_sha256"),
            "function_count": runtime.get("function_count", 0),
            "stage_costs": stage_costs,
        },
        Path(sample_memory_dir) / MEMORY_BUILD_DONE_FILE,
    )


def _sample_build_fingerprint(sample: dict, model: str) -> str:
    payload = {
        "conversation": sample.get("conversation", {}),
        "extractor_sha256": extractor_configuration_fingerprint(model),
        "semantic_compiler_sha256": semantic_compiler_fingerprint(model),
        "builder_version": MEMORY_BUILDER_VERSION,
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _remove_legacy_memory_views(sample_dir: Path) -> None:
    """The new runtime never reads Markdown views; clear only exact old targets."""

    for name in ("concepts", "sessions"):
        target = sample_dir / name
        if target.is_dir():
            shutil.rmtree(target)
    for name in (
        "memory_index.json",
        "fact_view_map.json",
        "concept_graph.json",
        "fact_id_aliases.json",
        "fact_duplicates.json",
        "relation_index.json",
    ):
        (sample_dir / name).unlink(missing_ok=True)


def _scoped_output_path(output_dir: str) -> str:
    """Keep every generated artifact inside the user-approved ``MaC`` tree."""

    project_root = Path(__file__).resolve().parents[1]
    candidate = Path(output_dir)
    resolved = (candidate if candidate.is_absolute() else project_root / candidate).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"OUTPUT_DIR must be inside {project_root}, got {output_dir!r}"
        ) from exc
    if resolved == project_root:
        raise ValueError("OUTPUT_DIR must be a child directory of MaC, not MaC itself")
    return str(resolved)


def _hash_memory_code(path: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted(path.rglob("*.py")):
        digest.update(str(source.relative_to(path)).encode("utf-8"))
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _deferred_questions_path(config: Config, pred_suffix: str = "") -> Path:
    """Return the resumable queue of questions deferred after all API retries."""

    path = Path(config.output_dir) / "predictions" / f"deferred_questions{pred_suffix}.json"
    ensure_dir(path.parent)
    return path


def _deferred_question_key(sample_id: str, question_id: int) -> str:
    return f"{sample_id}:question_{question_id:03d}"


def _read_deferred_questions(path: Path) -> dict[str, dict]:
    """Read a queue defensively so an interrupted/manual edit never stops QA."""

    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        logging.warning("Ignoring unreadable deferred-question queue: %s", path)
        return {}
    if not (
        isinstance(payload, dict)
        and payload.get("schema") == DEFERRED_QUESTIONS_SCHEMA
        and isinstance(payload.get("questions"), list)
    ):
        logging.warning("Ignoring invalid deferred-question queue: %s", path)
        return {}

    queue: dict[str, dict] = {}
    for item in payload["questions"]:
        if not isinstance(item, dict):
            continue
        sample_id = item.get("sample_id")
        question_id = item.get("question_id")
        fingerprint = item.get("prediction_fingerprint")
        if not (
            isinstance(sample_id, str)
            and isinstance(question_id, int)
            and isinstance(fingerprint, str)
        ):
            continue
        queue[_deferred_question_key(sample_id, question_id)] = dict(item)
    return queue


def _write_deferred_questions(path: Path, queue: dict[str, dict]) -> None:
    """Atomically persist pending questions after every coordinator-side change."""

    questions = [queue[key] for key in sorted(queue)]
    write_json(
        {
            "schema": DEFERRED_QUESTIONS_SCHEMA,
            "pending_count": len(questions),
            "questions": questions,
        },
        path,
    )


def _configured_positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _next_unfinished_qa_stage(
    sample_memory_dir: str,
    sample_id: str,
    question_id: int,
) -> str:
    """Identify where a deferred question can resume from its checkpoints."""

    for stage in QUESTION_EXECUTION_STAGES:
        if not stage_checkpoint_path(
            sample_memory_dir, sample_id, question_id, stage
        ).exists():
            return stage
    return "prediction_write"


def _record_deferred_question(
    queue: dict[str, dict],
    *,
    sample_id: str,
    question_id: int,
    question: str,
    category: object,
    prediction_sha256: str,
    sample_memory_dir: str,
    exc: Exception,
) -> dict:
    """Upsert one exhausted-timeout question without losing its retry history."""

    key = _deferred_question_key(sample_id, question_id)
    previous = queue.get(key, {})
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entry = {
        "sample_id": sample_id,
        "question_id": question_id,
        "question": str(question),
        "category": category,
        "prediction_fingerprint": prediction_sha256,
        "status": "pending",
        "defer_rounds": int(previous.get("defer_rounds", 0)) + 1,
        "request_timeout_seconds": _configured_positive_int("MAC_REQUEST_TIMEOUT", 180),
        "llm_attempts_per_round": _configured_positive_int("MAC_API_MAX_RETRIES", 3),
        "resume_stage": _next_unfinished_qa_stage(
            sample_memory_dir, sample_id, question_id
        ),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "first_deferred_at": previous.get("first_deferred_at", now),
        "last_deferred_at": now,
    }
    queue[key] = entry
    return entry


def _is_legacy_program_abstain_prediction(row: dict) -> bool:
    """Recognise only the old hard-coded false-negative prediction shape."""

    program_trace = row.get("program_trace")
    answer_trace = row.get("answer_trace")
    return bool(
        isinstance(program_trace, dict)
        and program_trace.get("status") == "abstain"
        and isinstance(answer_trace, dict)
        and answer_trace.get("reason") == "program_generation_failed"
    )


def _purge_legacy_program_abstain_predictions(pred_path: str) -> int:
    """Remove old fake ``Not mentioned`` rows so their questions are resumed."""

    rows = read_jsonl(pred_path)
    retained = [row for row in rows if not _is_legacy_program_abstain_prediction(row)]
    removed = len(rows) - len(retained)
    if removed:
        write_jsonl(retained, pred_path)
    return removed


def cmd_run_qa(
    config: Config,
    pred_suffix: str = "",
    *,
    deferred_only: bool = False,
) -> int:
    config.output_dir = _scoped_output_path(config.output_dir)
    setup_logging()
    logging.info(
        "Starting run-qa | data=%s | model=%s | max_samples=%s | "
        "max_questions=%s | workers=%d | deferred_only=%s",
        config.data_path,
        config.model,
        config.max_samples,
        config.max_questions_per_sample,
        config.qa_max_workers,
        deferred_only,
    )
    samples = _selected_samples(config)

    specs = _sample_specs(samples, config.output_dir)
    pred_path = _prediction_path(config, pred_suffix)
    purged_abstentions = _purge_legacy_program_abstain_predictions(pred_path)
    if purged_abstentions:
        logging.info(
            "RETRY: removed %d legacy program-abstain prediction(s) for rerun",
            purged_abstentions,
        )
    deferred_path = _deferred_questions_path(config, pred_suffix)
    deferred_queue = _read_deferred_questions(deferred_path)

    # Resume state is read exactly once by the coordinator before workers start.
    completed_fingerprints = read_completed_prediction_fingerprints(pred_path)
    tasks: list[tuple[dict, str, str, int, dict, int, str]] = []
    selected_question_count = 0
    completed_count = 0
    deferred_skipped_count = 0
    queue_changed = False
    for sample, sample_id, sample_memory_dir in specs:
        selected_qa_items = _selected_qa_items(sample, config)
        qa_items = _evaluated_qa_items(sample, config)
        selected_question_count += len(qa_items)
        for question_id, qa in qa_items:
            key = prediction_key(sample_id, question_id)
            expected_fingerprint = prediction_fingerprint(
                sample,
                qa,
                sample_memory_dir,
                config.model,
            )
            if completed_fingerprints.get(key) == expected_fingerprint:
                if _deferred_question_key(sample_id, question_id) in deferred_queue:
                    deferred_queue.pop(_deferred_question_key(sample_id, question_id))
                    queue_changed = True
                completed_count += 1
                continue
            deferred_key = _deferred_question_key(sample_id, question_id)
            queued = deferred_queue.get(deferred_key)
            if queued and queued.get("prediction_fingerprint") != expected_fingerprint:
                # A changed question/memory representation should be eligible for a
                # fresh normal pass, rather than staying blocked by an old timeout.
                deferred_queue.pop(deferred_key)
                queued = None
                queue_changed = True
            if deferred_only:
                if queued is None:
                    continue
            elif queued is not None:
                deferred_skipped_count += 1
                continue
            tasks.append(
                (
                    sample,
                    sample_id,
                    sample_memory_dir,
                    question_id,
                    qa,
                    len(selected_qa_items),
                    expected_fingerprint,
                )
            )

    if queue_changed:
        _write_deferred_questions(deferred_path, deferred_queue)
    tasks = _round_robin_qa_tasks(tasks)
    if completed_count:
        logging.info("RESUME: skipping %d completed questions", completed_count)
    if deferred_skipped_count:
        logging.info(
            "DEFERRED: leaving %d exhausted-timeout question(s) for the final retry pass",
            deferred_skipped_count,
        )
    if not tasks:
        pending_count = len(deferred_queue)
        if pending_count:
            logging.info("No eligible QA tasks now; %d question(s) remain deferred.", pending_count)
        else:
            logging.info("All selected questions are already complete.")
        _write_current_cost_reports(config, samples)
        return pending_count

    get_thread_client = _thread_local_client_getter(
        config.api_key,
        config.base_url,
        config.model,
        provider=config.llm_provider,
    )

    def answer_one(task: tuple[dict, str, str, int, dict, int, str]) -> dict:
        sample, sample_id, sample_memory_dir, question_id, qa, total_questions, _ = task
        return run_qa_question(
            sample,
            sample_id,
            sample_memory_dir,
            question_id,
            qa,
            get_thread_client(),
            total_questions=total_questions,
        )

    failures: list[tuple[str, int, Exception]] = []
    with ThreadPoolExecutor(
        max_workers=config.qa_max_workers,
        thread_name_prefix="qa",
    ) as executor:
        future_to_task = {
            executor.submit(answer_one, task): task for task in tasks
        }
        for future in tqdm(
            as_completed(future_to_task),
            total=len(future_to_task),
            desc="Running QA",
            unit="q",
        ):
            task = future_to_task[future]
            sample, sample_id, sample_memory_dir, question_id, qa, _, expected_fingerprint = task
            try:
                prediction = future.result()
            except Exception as exc:
                if isinstance(exc, APITimeoutError):
                    entry = _record_deferred_question(
                        deferred_queue,
                        sample_id=sample_id,
                        question_id=question_id,
                        question=str(qa.get("question", "")),
                        category=qa.get("category"),
                        prediction_sha256=expected_fingerprint,
                        sample_memory_dir=sample_memory_dir,
                        exc=exc,
                    )
                    _write_deferred_questions(deferred_path, deferred_queue)
                    logging.warning(
                        "QA timed out after %d request attempt(s) for sample=%s question=%d; "
                        "deferred (round=%d, resume_stage=%s).",
                        entry["llm_attempts_per_round"],
                        sample_id,
                        question_id,
                        entry["defer_rounds"],
                        entry["resume_stage"],
                    )
                else:
                    logging.error(
                        "QA failed for sample=%s question=%d; completed rows remain resumable.",
                        sample_id,
                        question_id,
                        exc_info=(type(exc), exc, exc.__traceback__),
                    )
                    failures.append((sample_id, question_id, exc))
                continue

            # Only this coordinator thread writes the shared JSONL file.
            append_jsonl(prediction, pred_path)
            deferred_key = _deferred_question_key(sample_id, question_id)
            if deferred_key in deferred_queue:
                deferred_queue.pop(deferred_key)
                _write_deferred_questions(deferred_path, deferred_queue)

    _write_current_cost_reports(config, samples)
    if failures:
        first_sample_id, first_question_id, first_exc = failures[0]
        raise RuntimeError(
            f"{len(failures)} QA question(s) failed; rerun to resume. "
            f"First failed key: ({first_sample_id}, {first_question_id}); error: {first_exc}"
        ) from first_exc
    pending_count = len(deferred_queue)
    if pending_count:
        logging.info("QA pass complete with %d deferred question(s).", pending_count)
    return pending_count


def cmd_run_all(
    config: Config,
    force: bool = False,
    judge: bool = False,
    pred_suffix: str = "",
) -> None:
    cmd_build_memory(config, force=force)
    pending_count = cmd_run_qa(config, pred_suffix=pred_suffix)
    if pending_count:
        logging.info(
            "Starting automatic final retry for %d deferred question(s).", pending_count
        )
        pending_count = cmd_run_qa(
            config,
            pred_suffix=pred_suffix,
            deferred_only=True,
        )
    if pending_count:
        logging.warning(
            "Skipping evaluation: %d question(s) still timed out after the final retry. "
            "Their details remain in %s.",
            pending_count,
            _deferred_questions_path(config, pred_suffix),
        )
        return
    cmd_eval(config, judge=judge, pred_suffix=pred_suffix)


def cmd_eval(config: Config, judge: bool = False, pred_suffix: str = "") -> None:
    config.output_dir = _scoped_output_path(config.output_dir)
    setup_logging()
    pred_path = _prediction_path(config, pred_suffix)
    logging.info(
        "Starting eval | predictions=%s | judge=%s | judge_workers=%d",
        pred_path,
        judge,
        config.judge_max_workers,
    )

    llm = None
    judge_out = ""
    if judge:
        llm = OpenAIJudgeClient(
            config.judge_api_key,
            config.judge_base_url,
            config.judge_model,
        )
        judge_out = str(
            Path(config.output_dir)
            / "predictions"
            / f"result_judge{pred_suffix}.jsonl"
        )

    summary = evaluate_predictions(
        pred_path,
        llm=llm,
        judge_out_path=judge_out,
        judge_workers=config.judge_max_workers,
        generation_model=config.model,
    )
    summary_name = f"{config.dataset}_table_summary{pred_suffix}.json" if pred_suffix else f"{config.dataset}_table_summary.json"
    summary_path = Path(config.output_dir) / "predictions" / summary_name
    _, cost_all = _write_current_cost_reports(config, _selected_samples(config))
    summary["four_stage_costs"] = cost_all.get("four_stage_totals", {})
    summary["cost_report_path"] = str(Path(config.output_dir) / "predictions" / "cost_all.json")
    write_json(summary, summary_path)
    _write_result_summary(Path(config.output_dir) / f"result_summary{pred_suffix}.md", summary)
    logging.info("Wrote %s summary to %s", config.dataset, summary_path)


def _format_cost_value(metric: dict | None, key: str) -> str:
    if not metric or key not in metric:
        return "--"
    value = metric[key]
    return f"{float(value):,.2f}" if key == "runtime_seconds" else f"{int(value):,}"


def _write_result_summary(path: Path, summary: dict) -> None:
    lines = [
        "# Memory as Code Results", "", "## Accuracy", "",
        "| Category | Questions | F1 | Binary LLM Judge |",
        "| --- | ---: | ---: | ---: |",
    ]
    per_category = summary.get("per_category") or {}
    judge_by_category = ((summary.get("judge") or {}).get("per_category") or {})
    for category in LM_CATEGORIES:
        result = per_category.get(category, {})
        judge = judge_by_category.get(category, {})
        f1, accuracy = result.get("f1"), judge.get("accuracy")
        lines.append(
            f"| {LM_CATEGORY_NAMES[category]} | {result.get('count', 0)} | "
            f"{f'{float(f1) * 100:.2f}' if f1 is not None else '--'} | "
            f"{f'{float(accuracy) * 100:.2f}' if accuracy is not None else '--'} |"
        )
    judge = summary.get("judge") or {}
    overall_judge = judge.get("overall_accuracy")
    lines.extend([
        f"| Overall | {summary.get('total_questions', 0)} | {float(summary.get('overall_f1', 0.0)) * 100:.2f} | "
        f"{f'{float(overall_judge) * 100:.2f}' if overall_judge is not None else '--'} |", ""
    ])
    costs = summary.get("four_stage_costs") or {}
    order = (
        ("Construction", "construction"), ("Retrieval", "retrieval"),
        ("2-Stage Total", "2_stage_total"), ("Answer Generation", "answer_generation"),
        ("Evaluation", "evaluation"), ("4-Stage Total", "4_stage_total"),
    )
    for title, field in (("Token", "token_consumption"), ("Runtime (s)", "runtime_seconds"), ("Successful LLM Requests", "request_count")):
        lines.extend([
            f"## {title}", "", "| " + " | ".join(label for label, _ in order) + " |",
            "| " + " | ".join("---:" for _ in order) + " |",
            "| " + " | ".join(_format_cost_value(costs.get(name), field) for _, name in order) + " |", "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer >= 1") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _add_worker_argument(parser: argparse.ArgumentParser, flag: str, dest: str) -> None:
    parser.add_argument(flag, dest=dest, type=_positive_int, default=None)


def _write_run_manifest(config: Config) -> None:
    """Persist reproducibility metadata without ever writing API credentials."""
    write_json(
        {
            "schema": "mac_run_config_v1",
            "llm_provider": config.llm_provider,
            "model": config.model,
            "dataset": config.dataset,
            "data_path": config.data_path,
            "judge_model": config.judge_model,
            "candidate_reasoning": "disabled",
            "judge_reasoning": "disabled",
            "gpt_transport": "official_openai_responses_api" if config.llm_provider == "gpt" else None,
        },
        Path(config.output_dir) / "run_config.json",
    )


def _validate_parallel_config(parser: argparse.ArgumentParser, config: Config) -> None:
    for field_name in (
        "build_max_workers",
        "semantic_max_workers",
        "qa_max_workers",
        "judge_max_workers",
        "llm_max_concurrency",
    ):
        value = getattr(config, field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            parser.error(f"{field_name} must be an integer >= 1, got {value!r}")
    if config.max_samples is not None and config.samples_per_category is not None:
        parser.error("MAX_SAMPLES/--max-samples and SAMPLES_PER_CATEGORY/--samples-per-category are mutually exclusive")
    if config.dataset == "locomo" and config.samples_per_category is not None:
        parser.error("--samples-per-category is available only for longmemeval")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memory as Code: programmatic search over executable memory"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_build = subparsers.add_parser("build-memory")
    p_build.add_argument("--data", default=None)
    p_build.add_argument("--output", default=None)
    p_build.add_argument("--max-samples", type=_positive_int, default=None)
    p_build.add_argument("--samples-per-category", type=_positive_int, default=None)
    p_build.add_argument("--force", action="store_true")
    _add_worker_argument(p_build, "--build-workers", "build_max_workers")
    _add_worker_argument(p_build, "--semantic-workers", "semantic_max_workers")
    _add_worker_argument(p_build, "--llm-max-concurrency", "llm_max_concurrency")

    p_backfill = subparsers.add_parser("backfill-construction-cost")
    p_backfill.add_argument("--data", default=None)
    p_backfill.add_argument("--output", default=None)
    p_backfill.add_argument("--max-samples", type=_positive_int, default=None)
    p_backfill.add_argument("--samples-per-category", type=_positive_int, default=None)
    _add_worker_argument(p_backfill, "--build-workers", "build_max_workers")
    _add_worker_argument(p_backfill, "--llm-max-concurrency", "llm_max_concurrency")

    p_qa = subparsers.add_parser("run-qa")
    p_qa.add_argument("--data", default=None)
    p_qa.add_argument("--output", default=None)
    p_qa.add_argument("--max-samples", type=_positive_int, default=None)
    p_qa.add_argument("--samples-per-category", type=_positive_int, default=None)
    p_qa.add_argument("--max-questions-per-sample", type=_positive_int, default=None)
    p_qa.add_argument("--pred-suffix", default="")
    _add_worker_argument(p_qa, "--qa-workers", "qa_max_workers")
    _add_worker_argument(p_qa, "--llm-max-concurrency", "llm_max_concurrency")

    p_all = subparsers.add_parser("run-all")
    p_all.add_argument("--data", default=None)
    p_all.add_argument("--output", default=None)
    p_all.add_argument("--max-samples", type=_positive_int, default=None)
    p_all.add_argument("--samples-per-category", type=_positive_int, default=None)
    p_all.add_argument("--max-questions-per-sample", type=_positive_int, default=None)
    p_all.add_argument("--force", action="store_true")
    p_all.add_argument("--judge", action="store_true", help="Enable LLM-as-judge evaluation")
    p_all.add_argument("--pred-suffix", default="")
    _add_worker_argument(p_all, "--build-workers", "build_max_workers")
    _add_worker_argument(p_all, "--semantic-workers", "semantic_max_workers")
    _add_worker_argument(p_all, "--qa-workers", "qa_max_workers")
    _add_worker_argument(p_all, "--judge-workers", "judge_max_workers")
    _add_worker_argument(p_all, "--llm-max-concurrency", "llm_max_concurrency")

    p_eval = subparsers.add_parser("eval")
    p_eval.add_argument("--output", default=None)
    p_eval.add_argument("--judge", action="store_true", help="Enable LLM-as-judge evaluation")
    p_eval.add_argument("--pred-suffix", default="")
    _add_worker_argument(p_eval, "--judge-workers", "judge_max_workers")
    _add_worker_argument(p_eval, "--llm-max-concurrency", "llm_max_concurrency")

    for command_parser in (p_build, p_backfill, p_qa, p_all, p_eval):
        command_parser.add_argument("--llm", choices=("qwen", "deepseek", "gpt"), default="qwen", help="LLM API adapter")
        command_parser.add_argument("--dataset", choices=("locomo", "longmemeval"), default="longmemeval", help="benchmark dataset")
        command_parser.add_argument("--model", default=None, help="override the provider's default model name")

    args = parser.parse_args()
    config = load_config(provider=args.llm, dataset=args.dataset, model=args.model)

    if hasattr(args, "data") and args.data:
        config.data_path = args.data
    if hasattr(args, "output") and args.output:
        config.output_dir = str(Path(args.output))
    if hasattr(args, "max_samples") and args.max_samples is not None:
        config.max_samples = args.max_samples
    if hasattr(args, "samples_per_category") and args.samples_per_category is not None:
        config.samples_per_category = args.samples_per_category
    if (
        hasattr(args, "max_questions_per_sample")
        and args.max_questions_per_sample is not None
    ):
        config.max_questions_per_sample = args.max_questions_per_sample

    # Retrieval helpers load the original dialogue file through DATA_PATH.
    # Keep that lookup aligned with any CLI or environment override above.
    os.environ["DATA_PATH"] = config.data_path
    # DeepSeek's JSON output can truncate an 80-fact semantic batch. Keep the
    # original dataset defaults for Qwen/GPT while using a safe DeepSeek size.
    if config.llm_provider == "deepseek":
        os.environ.setdefault("MAC_SEMANTIC_BATCH_SIZE", "40")

    for field_name in (
        "build_max_workers",
        "semantic_max_workers",
        "qa_max_workers",
        "judge_max_workers",
        "llm_max_concurrency",
    ):
        override = getattr(args, field_name, None)
        if override is not None:
            setattr(config, field_name, override)

    _validate_parallel_config(parser, config)
    try:
        config.output_dir = _scoped_output_path(config.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    _write_run_manifest(config)
    llm_client.configure_max_concurrency(config.llm_max_concurrency)

    force = getattr(args, "force", False)
    judge = getattr(args, "judge", False)
    pred_suffix = getattr(args, "pred_suffix", "")

    if args.command == "build-memory":
        cmd_build_memory(config, force=force)
    elif args.command == "backfill-construction-cost":
        cmd_backfill_construction_cost(config)
    elif args.command == "run-qa":
        cmd_run_qa(config, pred_suffix=pred_suffix)
    elif args.command == "run-all":
        cmd_run_all(config, force=force, judge=judge, pred_suffix=pred_suffix)
    elif args.command == "eval":
        cmd_eval(config, judge=judge, pred_suffix=pred_suffix)


if __name__ == "__main__":
    main()
