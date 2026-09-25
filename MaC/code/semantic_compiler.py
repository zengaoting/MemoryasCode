"""Compile canonical facts into code-native executable memory packages."""

from __future__ import annotations

import hashlib
import json
import keyword
import logging
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .cost_metrics import merge_metrics, metric_is_complete, usage_metric
from .llm_client import LLMClient, capture_token_usage
from .utils import ensure_dir, read_json, write_json


logger = logging.getLogger(__name__)


SEMANTIC_SCHEMA_VERSION = 5
RUNTIME_SCHEMA_VERSION = "memory_as_code_v10"
SEMANTIC_BATCH_SIZE = 80
STATE_LINK_BATCH_FACTS = 96
STRUCTURAL_RELATIONS = {"temporal_before", "same_subject", "shared_topic"}
SEMANTIC_PARTIAL_DIR = "semantic_batches"
SEMANTIC_PARTIAL_SCHEMA = "semantic_batch_checkpoint_v1"


def semantic_batch_size() -> int:
    """Use the original dataset-specific semantic compilation batch size."""
    from .benchmark import DATASET_NAME

    configured = os.getenv("MAC_SEMANTIC_BATCH_SIZE", "").strip()
    if configured:
        value = int(configured)
        if value < 1:
            raise ValueError("MAC_SEMANTIC_BATCH_SIZE must be >= 1")
        return value
    return 40 if DATASET_NAME == "LoCoMo" else SEMANTIC_BATCH_SIZE


def _resume_ignores_llm_configuration() -> bool:
    """Reuse semantic outputs across reviewed model/prompt configuration changes."""
    return os.getenv("OUR_V1_RESUME_IGNORE_LLM_CONFIG", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }

SEMANTIC_COMPILER_SYSTEM = """You compile canonical dialogue facts into a code-native memory specification.
Return valid JSON only. Do not invent facts. Each fact will become one named pure Python
function in a subject module. Choose a short snake_case function_name describing the fact,
a compact open-ended query predicate, a JSON-compatible value, optional JSON-object
attributes, and whether the fact is a time-varying state. Do not infer or output any
state-to-state link in this phase. State links are resolved in a
separate phase after all facts have been semantically parsed. Do not create ordinary
fact-to-fact graph relations. All generated base functions return the same open MemoryValue
runtime type; do not classify facts into a fixed type ontology."""

SEMANTIC_COMPILER_USER = """Compile these facts.

Output exactly:
{
  "facts": [
    {
      "fact_id": "...",
      "function_name": "snake_case_name",
      "predicate": "...",
      "value": "string, number, or object",
      "attributes": {"optional_key": "optional JSON-compatible value"},
      "stateful": false
    }
  ]
}

Input facts:
{{facts}}
"""

STATE_LINK_SYSTEM = """You resolve explicit state-update links over already parsed dialogue facts.
Return valid JSON only. A link means the successor fact explicitly replaces the predecessor
as the same subject's same predicate state. Do not infer a link merely because one fact is
newer, related, or more specific. A successor may name at most one predecessor. Do not emit
ordinary fact-graph relations. Every ID in an output link must come from the same supplied
state candidate group."""

STATE_LINK_USER = """State candidates:
{{groups}}

Output exactly:
{
  "links": [
    {
      "successor_fact_id": "...",
      "predecessor_fact_id": "..."
    }
  ]
}
"""


def semantic_compiler_fingerprint(model: str) -> str:
    payload = {
        "version": SEMANTIC_SCHEMA_VERSION,
        "model": str(model),
        "batch_size": semantic_batch_size(),
        "state_link_batch_facts": STATE_LINK_BATCH_FACTS,
        "system": SEMANTIC_COMPILER_SYSTEM,
        "user": SEMANTIC_COMPILER_USER,
        "state_link_system": STATE_LINK_SYSTEM,
        "state_link_user": STATE_LINK_USER,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _append_metric(
    current: dict[str, Any] | None,
    additional: dict[str, Any],
) -> dict[str, Any]:
    """Accumulate request metrics without treating the first request as unknown."""
    return additional if current is None else merge_metrics(current, additional)


def compile_semantic_specs(
    *, sample_id: str, llm: LLMClient, sample_memory_dir: str | Path, force: bool = False,
) -> dict[str, Any]:
    """Persist semantic specs with per-request checkpoints.

    ``semantic_specs.json`` remains the complete, immutable hand-off consumed
    by code emission.  While it is being built, successful parser/state-link
    batches are written separately under ``semantic_batches/``.  A resumed
    construction consequently retries only the unfinished provider call.
    """

    sample_dir = Path(sample_memory_dir).resolve()
    project_root = Path(__file__).resolve().parents[1]
    try:
        sample_dir.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"sample memory directory must be inside {project_root}") from exc

    facts_path = sample_dir / "facts_by_id.json"
    facts = _read_canonical_facts(facts_path)
    facts_sha256 = _sha256_file(facts_path)
    semantic_path = sample_dir / "semantic_specs.json"
    fingerprint = semantic_compiler_fingerprint(str(getattr(llm, "model", "")))
    existing = _read_semantic_specs(semantic_path, sample_id, fingerprint, facts_sha256)
    checkpoint_dir = sample_dir / SEMANTIC_PARTIAL_DIR
    if force:
        semantic_path.unlink(missing_ok=True)
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        existing = None
    if force or existing is None:
        specs, semantic_metric = annotate_fact_semantics(
            facts,
            llm,
            checkpoint_dir=checkpoint_dir,
            sample_id=sample_id,
            compiler_fingerprint=fingerprint,
            facts_sha256=facts_sha256,
        )
        state_links, state_link_metric = link_explicit_state_updates(
            facts,
            specs,
            llm,
            checkpoint_dir=checkpoint_dir,
            sample_id=sample_id,
            compiler_fingerprint=fingerprint,
            facts_sha256=facts_sha256,
        )
        _apply_state_links(specs, state_links)
        cost_metric = (
            merge_metrics(semantic_metric, state_link_metric)
            if state_link_metric is not None
            else semantic_metric
        )
        write_json({
            "schema": "semantic_fact_specs_v5",
            "sample_id": sample_id,
            "compiler_fingerprint": fingerprint,
            "facts_by_id_sha256": facts_sha256,
            "facts": specs,
            "state_links": state_links,
            "cost_metrics": cost_metric,
        }, semantic_path)
    else:
        specs = list(existing["facts"])
        cost_metric = existing.get("cost_metrics")

    return {
        "facts_by_id_sha256": facts_sha256,
        "semantic_compilation": cost_metric,
        "fact_count": len(facts),
    }


def emit_compiled_memory(
    *, sample_id: str, sample_memory_dir: str | Path,
) -> dict[str, Any]:
    """Emit Python memory packages from already-persisted semantic specs."""

    sample_dir = Path(sample_memory_dir).resolve()
    project_root = Path(__file__).resolve().parents[1]
    try:
        sample_dir.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"sample memory directory must be inside {project_root}") from exc

    facts_path = sample_dir / "facts_by_id.json"
    facts = _read_canonical_facts(facts_path)
    facts_sha256 = _sha256_file(facts_path)
    # The model is part of the semantic checkpoint identity.  Read the saved
    # payload directly here: its own fingerprint was checked before writing.
    payload = read_json(sample_dir / "semantic_specs.json")
    if not isinstance(payload, dict) or payload.get("sample_id") != sample_id:
        raise ValueError(f"missing or invalid semantic specs for {sample_id}")
    if payload.get("facts_by_id_sha256") != facts_sha256:
        raise ValueError(f"semantic specs do not match canonical facts for {sample_id}")
    specs = payload.get("facts")
    if not isinstance(specs, list):
        raise ValueError(f"semantic specs are malformed for {sample_id}")

    emit_started = time.monotonic()
    runtime = emit_memory_code(facts=facts, specs=specs, sample_dir=sample_dir)
    runtime["local_runtime_seconds"] = time.monotonic() - emit_started
    runtime["facts_by_id_sha256"] = facts_sha256
    return runtime


def replay_cached_semantic_cost(
    *, sample_id: str, llm: LLMClient, sample_memory_dir: str | Path,
) -> dict[str, Any]:
    """Measure a legacy semantic build without changing its persisted memory.

    Older cached ``semantic_specs.json`` files can contain correct semantic
    outputs but lack provider usage.  The original usage cannot be recovered
    after the fact.  This function therefore replays the *same persisted
    semantic-parser and state-link inputs*, records the new request costs, and
    updates only the cost metadata.  It never replaces facts, semantic specs,
    links, or generated Python code.
    """
    sample_dir = Path(sample_memory_dir).resolve()
    project_root = Path(__file__).resolve().parents[1]
    try:
        sample_dir.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"sample memory directory must be inside {project_root}") from exc

    facts_path = sample_dir / "facts_by_id.json"
    facts = _read_canonical_facts(facts_path)
    facts_sha256 = _sha256_file(facts_path)
    fingerprint = semantic_compiler_fingerprint(str(getattr(llm, "model", "")))
    semantic_path = sample_dir / "semantic_specs.json"
    existing = _read_semantic_specs(semantic_path, sample_id, fingerprint, facts_sha256)
    if existing is None:
        raise ValueError(
            f"cannot replay semantic cost for {sample_id}: persisted semantic specs do not match current inputs"
        )

    combined_metric = None
    batch_size = semantic_batch_size()
    for start in range(0, len(facts), batch_size):
        batch = facts[start:start + batch_size]
        compact = [{
            "fact_id": str(fact.get("fact_id", "")),
            "speaker": fact.get("speaker", ""),
            "subject": fact.get("subject", ""),
            "fact_text": fact.get("fact_text", ""),
            "fact_type": fact.get("fact_type", "other"),
            "topics": fact.get("topics", []),
            "normalized_time": fact.get("normalized_time", ""),
            "sources": fact.get("sources", []),
        } for fact in batch]
        user = SEMANTIC_COMPILER_USER.replace(
            "{{facts}}", json.dumps(compact, ensure_ascii=False, separators=(",", ":")),
        )
        started = time.monotonic()
        with capture_token_usage() as usage:
            llm.chat_json(SEMANTIC_COMPILER_SYSTEM, user)
        combined_metric = _append_metric(
            combined_metric, usage_metric(usage.snapshot(), time.monotonic() - started),
        )

    # State-link prompts are reconstructed from the persisted specs rather
    # than a newly sampled parser output, so their inputs match the memory
    # representation used by the completed experiment.
    for batch in _state_link_batches(_state_candidate_groups(facts, list(existing["facts"]))):
        user = STATE_LINK_USER.replace(
            "{{groups}}", json.dumps(batch, ensure_ascii=False, separators=(",", ":")),
        )
        started = time.monotonic()
        with capture_token_usage() as usage:
            llm.chat_json(STATE_LINK_SYSTEM, user)
        combined_metric = _append_metric(
            combined_metric, usage_metric(usage.snapshot(), time.monotonic() - started),
        )

    if not metric_is_complete(combined_metric):
        raise RuntimeError(f"semantic cost replay for {sample_id} did not yield complete usage")

    payload = read_json(semantic_path)
    payload["cost_metrics"] = combined_metric
    payload["cost_metric_provenance"] = {
        "mode": "replayed_from_persisted_semantic_inputs_v1",
        "semantic_compiler_fingerprint": fingerprint,
    }
    write_json(payload, semantic_path)
    return combined_metric


def _batch_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _partial_batch_path(checkpoint_dir: Path, phase: str, index: int) -> Path:
    return checkpoint_dir / f"{phase}_{index:04d}.json"


def _read_partial_batch(
    *,
    checkpoint_dir: Path | None,
    phase: str,
    index: int,
    sample_id: str | None,
    compiler_fingerprint: str | None,
    facts_sha256: str | None,
    input_sha256: str,
) -> dict[str, Any] | None:
    if (
        checkpoint_dir is None
        or not sample_id
        or not compiler_fingerprint
        or not facts_sha256
    ):
        return None
    path = _partial_batch_path(checkpoint_dir, phase, index)
    try:
        payload = read_json(path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if not (
        payload.get("schema") == SEMANTIC_PARTIAL_SCHEMA
        and payload.get("phase") == phase
        and payload.get("index") == index
        and payload.get("sample_id") == sample_id
        and (
            payload.get("compiler_fingerprint") == compiler_fingerprint
            or _resume_ignores_llm_configuration()
        )
        and payload.get("facts_by_id_sha256") == facts_sha256
        and payload.get("input_sha256") == input_sha256
        and isinstance(payload.get("output"), list)
        and isinstance(payload.get("cost_metric"), dict)
    ):
        return None
    return payload


def _write_partial_batch(
    *,
    checkpoint_dir: Path | None,
    phase: str,
    index: int,
    sample_id: str | None,
    compiler_fingerprint: str | None,
    facts_sha256: str | None,
    input_sha256: str,
    output: list[dict[str, Any]],
    cost_metric: dict[str, Any],
) -> None:
    if (
        checkpoint_dir is None
        or not sample_id
        or not compiler_fingerprint
        or not facts_sha256
    ):
        return
    ensure_dir(checkpoint_dir)
    write_json(
        {
            "schema": SEMANTIC_PARTIAL_SCHEMA,
            "phase": phase,
            "index": index,
            "sample_id": sample_id,
            "compiler_fingerprint": compiler_fingerprint,
            "facts_by_id_sha256": facts_sha256,
            "input_sha256": input_sha256,
            "output": output,
            "cost_metric": cost_metric,
        },
        _partial_batch_path(checkpoint_dir, phase, index),
    )


def annotate_fact_semantics(
    facts: list[dict[str, Any]],
    llm: LLMClient,
    *,
    checkpoint_dir: Path | None = None,
    sample_id: str | None = None,
    compiler_fingerprint: str | None = None,
    facts_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    by_id: dict[str, dict[str, Any]] = {}
    combined_metric = None
    batch_size = semantic_batch_size()
    for batch_index, start in enumerate(range(0, len(facts), batch_size)):
        batch = facts[start:start + batch_size]
        compact = [{
            "fact_id": str(fact.get("fact_id", "")),
            "speaker": fact.get("speaker", ""),
            "subject": fact.get("subject", ""),
            "fact_text": fact.get("fact_text", ""),
            "fact_type": fact.get("fact_type", "other"),
            "topics": fact.get("topics", []),
            "normalized_time": fact.get("normalized_time", ""),
            "sources": fact.get("sources", []),
        } for fact in batch]
        input_sha256 = _batch_digest(compact)
        cached = _read_partial_batch(
            checkpoint_dir=checkpoint_dir,
            phase="semantic",
            index=batch_index,
            sample_id=sample_id,
            compiler_fingerprint=compiler_fingerprint,
            facts_sha256=facts_sha256,
            input_sha256=input_sha256,
        )
        if cached is not None:
            logger.info(
                "Reusing semantic checkpoint %s batch %d", sample_id or "<unscoped>", batch_index + 1,
            )
            batch_specs = [item for item in cached["output"] if isinstance(item, dict)]
            metric = cached["cost_metric"]
        else:
            logger.info(
                "Parsing semantic batch %s #%d (%d facts)",
                sample_id or "<unscoped>",
                batch_index + 1,
                len(batch),
            )
            user = SEMANTIC_COMPILER_USER.replace(
                "{{facts}}", json.dumps(compact, ensure_ascii=False, separators=(",", ":")),
            )
            started = time.monotonic()
            with capture_token_usage() as usage:
                result = llm.chat_json(SEMANTIC_COMPILER_SYSTEM, user)
            metric = usage_metric(usage.snapshot(), time.monotonic() - started)
            returned = {
                str(item.get("fact_id", "")): item
                for item in (result.get("facts", []) if isinstance(result, dict) else [])
                if isinstance(item, dict)
            }
            batch_specs = [
                _validated_spec(fact, returned.get(str(fact.get("fact_id", "")), {}))
                for fact in batch
            ]
            _write_partial_batch(
                checkpoint_dir=checkpoint_dir,
                phase="semantic",
                index=batch_index,
                sample_id=sample_id,
                compiler_fingerprint=compiler_fingerprint,
                facts_sha256=facts_sha256,
                input_sha256=input_sha256,
                output=batch_specs,
                cost_metric=metric,
            )
            logger.info(
                "Persisted semantic checkpoint %s batch %d",
                sample_id or "<unscoped>",
                batch_index + 1,
            )
        combined_metric = _append_metric(combined_metric, metric)
        for spec in batch_specs:
            fact_id = str(spec.get("fact_id", ""))
            if fact_id:
                by_id[fact_id] = spec
    return [by_id[str(fact.get("fact_id", ""))] for fact in facts], combined_metric


def _validated_spec(fact: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    predicate = _safe_identifier(raw.get("predicate") or fact.get("fact_type") or "other")
    function_name = _safe_function_name(raw.get("function_name")) or _fallback_function_name(fact, predicate)
    value = raw.get("value", fact.get("fact_text", ""))
    if not isinstance(value, (str, int, float, bool, list, dict)) and value is not None:
        value = str(value)
    attributes = raw.get("attributes", {})
    if not isinstance(attributes, dict):
        attributes = {}
    return {
        "fact_id": str(fact.get("fact_id", "")),
        "function_name": function_name,
        "predicate": predicate,
        "value": value,
        "attributes": attributes,
        "stateful": bool(raw.get("stateful", False)),
        "supersedes_fact_id": None,
    }


def link_explicit_state_updates(
    facts: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    llm: LLMClient,
    *,
    checkpoint_dir: Path | None = None,
    sample_id: str | None = None,
    compiler_fingerprint: str | None = None,
    facts_sha256: str | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    """Resolve update edges only after every fact has a stable semantic spec."""

    facts_by_id = {str(item.get("fact_id", "")): item for item in facts}
    specs_by_id = {str(item.get("fact_id", "")): item for item in specs}
    candidate_groups = _state_candidate_groups(facts, specs)

    raw_links: list[dict[str, Any]] = []
    combined_metric = None
    for batch_index, batch in enumerate(_state_link_batches(candidate_groups)):
        input_sha256 = _batch_digest(batch)
        cached = _read_partial_batch(
            checkpoint_dir=checkpoint_dir,
            phase="state_link",
            index=batch_index,
            sample_id=sample_id,
            compiler_fingerprint=compiler_fingerprint,
            facts_sha256=facts_sha256,
            input_sha256=input_sha256,
        )
        if cached is not None:
            logger.info(
                "Reusing state-link checkpoint %s batch %d", sample_id or "<unscoped>", batch_index + 1,
            )
            batch_links = [item for item in cached["output"] if isinstance(item, dict)]
            metric = cached["cost_metric"]
        else:
            logger.info(
                "Linking state batch %s #%d (%d candidate groups)",
                sample_id or "<unscoped>",
                batch_index + 1,
                len(batch),
            )
            user = STATE_LINK_USER.replace(
                "{{groups}}", json.dumps(batch, ensure_ascii=False, separators=(",", ":")),
            )
            started = time.monotonic()
            with capture_token_usage() as usage:
                result = llm.chat_json(STATE_LINK_SYSTEM, user)
            metric = usage_metric(usage.snapshot(), time.monotonic() - started)
            batch_links = [
                item for item in (result.get("links", []) if isinstance(result, dict) else [])
                if isinstance(item, dict)
            ]
            _write_partial_batch(
                checkpoint_dir=checkpoint_dir,
                phase="state_link",
                index=batch_index,
                sample_id=sample_id,
                compiler_fingerprint=compiler_fingerprint,
                facts_sha256=facts_sha256,
                input_sha256=input_sha256,
                output=batch_links,
                cost_metric=metric,
            )
            logger.info(
                "Persisted state-link checkpoint %s batch %d",
                sample_id or "<unscoped>",
                batch_index + 1,
            )
        combined_metric = _append_metric(combined_metric, metric)
        raw_links.extend(batch_links)

    return _validated_state_links(facts_by_id, specs_by_id, raw_links), combined_metric


def _state_candidate_groups(
    facts: list[dict[str, Any]], specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build deterministic state-link inputs from persisted semantic specs."""
    specs_by_id = {str(item.get("fact_id", "")): item for item in specs}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        fact_id = str(fact.get("fact_id", ""))
        spec = specs_by_id.get(fact_id, {})
        if not spec.get("stateful"):
            continue
        key = (_normalise(fact.get("subject")), str(spec.get("predicate", "")))
        if key[0] and key[1]:
            groups[key].append(fact)

    candidate_groups: list[dict[str, Any]] = []
    for (subject, predicate), items in sorted(groups.items()):
        if len(items) < 2:
            continue
        ordered = sorted(items, key=lambda item: _state_order_key(item, facts))
        candidate_groups.append({
            "subject": subject,
            "predicate": predicate,
            "facts": [
                {
                    "fact_id": str(item.get("fact_id", "")),
                    "time": item.get("normalized_time", ""),
                    "fact_text": item.get("fact_text", ""),
                    "value": specs_by_id[str(item.get("fact_id", ""))].get("value"),
                    "attributes": specs_by_id[str(item.get("fact_id", ""))].get("attributes", {}),
                }
                for item in ordered
            ],
        })

    return candidate_groups


def _state_link_batches(groups: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_size = 0
    for group in groups:
        group_size = len(group.get("facts", []))
        if current and current_size + group_size > STATE_LINK_BATCH_FACTS:
            batches.append(current)
            current, current_size = [], 0
        current.append(group)
        current_size += group_size
    if current:
        batches.append(current)
    return batches


def _validated_state_links(
    facts_by_id: dict[str, dict[str, Any]],
    specs_by_id: dict[str, dict[str, Any]],
    raw_links: list[dict[str, Any]],
) -> list[dict[str, str]]:
    order = {fact_id: _state_order_key(fact, list(facts_by_id.values())) for fact_id, fact in facts_by_id.items()}
    selected: dict[str, str] = {}
    for raw in raw_links:
        successor = str(raw.get("successor_fact_id", "")).strip()
        predecessor = str(raw.get("predecessor_fact_id", "")).strip()
        if not successor or not predecessor or successor == predecessor:
            continue
        fact, old_fact = facts_by_id.get(successor), facts_by_id.get(predecessor)
        spec, old_spec = specs_by_id.get(successor), specs_by_id.get(predecessor)
        if not (fact and old_fact and spec and old_spec):
            continue
        if not (spec.get("stateful") and old_spec.get("stateful")):
            continue
        if (
            _normalise(fact.get("subject")) != _normalise(old_fact.get("subject"))
            or str(spec.get("predicate", "")) != str(old_spec.get("predicate", ""))
            or order[predecessor] >= order[successor]
        ):
            continue
        # A fact cannot replace two different predecessors.  Conflicting LLM
        # outputs are rejected rather than arbitrarily choosing one.
        if successor in selected and selected[successor] != predecessor:
            selected[successor] = ""
            continue
        if successor not in selected:
            selected[successor] = predecessor
    return [
        {"successor_fact_id": successor, "predecessor_fact_id": predecessor}
        for successor, predecessor in sorted(selected.items())
        if predecessor
    ]


def _apply_state_links(specs: list[dict[str, Any]], links: list[dict[str, str]]) -> None:
    specs_by_id = {str(item.get("fact_id", "")): item for item in specs}
    for spec in specs:
        spec["supersedes_fact_id"] = None
    for link in links:
        successor = specs_by_id.get(link["successor_fact_id"])
        predecessor = specs_by_id.get(link["predecessor_fact_id"])
        if successor is not None and predecessor is not None:
            successor["supersedes_fact_id"] = link["predecessor_fact_id"]
            successor["stateful"] = True
            predecessor["stateful"] = True


def _state_order_key(fact: dict[str, Any], facts: list[dict[str, Any]]) -> tuple:
    fact_id = str(fact.get("fact_id", ""))
    try:
        original_index = next(index for index, item in enumerate(facts) if str(item.get("fact_id", "")) == fact_id)
    except StopIteration:
        original_index = len(facts)
    return (_time_order_key(fact.get("normalized_time")), original_index, fact_id)


def _time_order_key(value: Any) -> tuple[int, int, int, str]:
    match = re.search(r"(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?", str(value or ""))
    if not match:
        return (0, 0, 0, str(value or ""))
    return (int(match.group(1)), int(match.group(2) or 0), int(match.group(3) or 0), str(value or ""))


def emit_memory_code(*, facts: list[dict[str, Any]], specs: list[dict[str, Any]], sample_dir: Path) -> dict[str, Any]:
    """Render packages; graph and indexes are derived from generated code metadata."""

    spec_by_id = {str(item.get("fact_id", "")): item for item in specs}
    records = [
        _runtime_fact(fact, spec_by_id.get(str(fact.get("fact_id", "")), {}), ordinal)
        for ordinal, fact in enumerate(facts, start=1)
    ]
    _assign_unique_function_names(records)
    # Every atomic fact is a standalone importable module.  The function name
    # is globally unique, so it is also a stable code-file name and lookup key.
    for record in records:
        record["module"] = f"facts.{record['function_name']}"
    links = _read_structural_links(sample_dir / "fact_graph.json", records)
    derived_groups = _derived_groups(records)
    code_dir = sample_dir / "memory_code"
    _write_runtime_package(code_dir, records, links, derived_groups)
    # ``fact_graph.json`` was created once from the canonical facts before
    # semantic compilation.  ``relations.py`` is its equivalent inspectable
    # code-native representation; the active graph retriever reads the JSON
    # index, so do not round-trip code back into JSON or overwrite that index.

    function_index = {
        "schema": "executable_function_index_v2",
        "functions": _function_index(records, derived_groups),
    }
    write_json(function_index, sample_dir / "function_index.json")
    # This deliberately contains only module membership and function IDs.  It
    # is the reverse navigation layer for code-native view expansion: after a
    # function-index hit, the runtime can find the concept/session modules
    # that import or define that function without storing fact text again.
    view_index = {
        "schema": "executable_view_index_v1",
        "views": _view_index(records, derived_groups),
    }
    write_json(view_index, sample_dir / "view_index.json")
    # These represented the former JSON-first retrieval design.  Canonical
    # facts now live only in facts_by_id.json and code discovery in
    # function_index.json, so leave no competing persisted index behind.
    for obsolete in (sample_dir / "fact_index.json", sample_dir / "function_catalog.json"):
        if obsolete.exists():
            obsolete.unlink()
    return {
        "schema": RUNTIME_SCHEMA_VERSION,
        "fact_count": len(records),
        "function_count": len(function_index["functions"]),
        "function_index_sha256": _sha256_file(sample_dir / "function_index.json"),
        "view_index_sha256": _sha256_file(sample_dir / "view_index.json"),
        "fact_graph_sha256": _sha256_file(sample_dir / "fact_graph.json"),
        "memory_code_sha256": _hash_code_dir(code_dir),
    }


def _runtime_fact(fact: dict[str, Any], spec: dict[str, Any], ordinal: int) -> dict[str, Any]:
    fact_id = str(fact.get("fact_id") or f"fact_{ordinal}")
    predicate = _safe_identifier(spec.get("predicate") or fact.get("fact_type") or "other")
    sources = _fact_sources(fact)
    return {
        "fact_id": fact_id,
        "module": "",
        "function_name": _safe_function_name(spec.get("function_name")) or _fallback_function_name(fact, predicate),
        "fact_text": str(fact.get("fact_text") or ""),
        "speaker": str(fact.get("speaker") or ""),
        "subject": str(fact.get("subject") or "unknown"),
        "predicate": predicate,
        "fact_type": str(fact.get("fact_type") or "other"),
        "value": spec.get("value", fact.get("fact_text", "")),
        "attributes": spec.get("attributes", {}),
        "topics": [str(value) for value in fact.get("topics", []) if str(value).strip()],
        "normalized_time": str(fact.get("normalized_time") or ""),
        "sources": sources,
        "dialogue_ids": [item["dialogue_id"] for item in sources],
        "session_id": sources[0]["session_id"] if sources else "unknown",
        "session_ids": sorted({item["session_id"] for item in sources}),
        "stateful": bool(spec.get("stateful", False)),
        "supersedes_fact_id": spec.get("supersedes_fact_id") or None,
    }


def _fact_sources(fact: dict[str, Any]) -> list[dict[str, str]]:
    pairs: set[tuple[str, str]] = set()
    raw_sources = fact.get("sources", [])
    if isinstance(raw_sources, list):
        for raw in raw_sources:
            if not isinstance(raw, dict):
                continue
            session = str(raw.get("session_id", raw.get("session", "unknown"))).strip() or "unknown"
            dialogue = str(raw.get("dialogue_id", raw.get("dialog", ""))).strip()
            if dialogue:
                pairs.add((session, dialogue))
    if not pairs:
        session = str(fact.get("session_id") or "unknown").strip() or "unknown"
        for dialogue in fact.get("dialogue_ids") or [fact.get("dia_id", "")]:
            text = str(dialogue).strip()
            if text:
                pairs.add((session, text))
    return [
        {"session_id": session, "dialogue_id": dialogue}
        for session, dialogue in sorted(pairs, key=lambda item: (item[0], _dialogue_order_key(item[1])))
    ]


def _dialogue_order_key(value: str) -> tuple[int, ...]:
    parts = [int(part) for part in re.findall(r"\d+", str(value))]
    return tuple(parts) if parts else (10**9,)


def _assign_unique_function_names(records: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["function_name"]].append(record)
    for name, items in groups.items():
        if len(items) > 1:
            for item in sorted(items, key=lambda value: value["fact_id"]):
                item["function_name"] = f"{name}__{_safe_identifier(item['fact_id'])}"


def _read_structural_links(path: Path, records: list[dict[str, Any]]) -> list[dict[str, str]]:
    payload = read_json(path)
    known = {record["fact_id"] for record in records}
    links, seen = [], set()
    for edge in payload.get("edges", []) if isinstance(payload, dict) else []:
        if not isinstance(edge, dict):
            continue
        source, target, relation = str(edge.get("source", "")), str(edge.get("target", "")), str(edge.get("relation", ""))
        key = (source, target, relation)
        if source in known and target in known and source != target and relation in STRUCTURAL_RELATIONS and key not in seen:
            seen.add(key)
            links.append({"source": source, "target": target, "relation": relation})
    return links


def _derived_groups(records: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    by_id = {record["fact_id"]: record for record in records}
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        predecessor = record.get("supersedes_fact_id")
        if predecessor and predecessor in by_id:
            groups[(record["subject"], record["predicate"])].update({record["fact_id"], str(predecessor)})
    order = {record["fact_id"]: index for index, record in enumerate(records)}
    return {
        key: sorted((by_id[fact_id] for fact_id in ids), key=lambda item: order[item["fact_id"]])
        for key, ids in groups.items()
    }


def _write_runtime_package(
    code_dir: Path,
    records: list[dict[str, Any]],
    links: list[dict[str, str]],
    derived_groups: dict[tuple[str, str], list[dict[str, Any]]],
) -> None:
    if code_dir.exists():
        shutil.rmtree(code_dir)
    ensure_dir(code_dir)
    (code_dir / "__init__.py").write_text("# Generated code-native executable memory package.\n", encoding="utf-8")
    for package in ("facts", "concept", "sessions"):
        directory = code_dir / package
        ensure_dir(directory)
        (directory / "__init__.py").write_text(f"# Generated {package} modules.\n", encoding="utf-8")

    by_id = {record["fact_id"]: record for record in records}
    for record in records:
        _write_fact_module(code_dir, record, by_id)
    _write_relation_module(code_dir, records, links)
    _write_concept_modules(code_dir, records, derived_groups)
    _write_topic_concept_modules(code_dir, records)
    _write_view_modules(code_dir, "sessions", records, lambda item: item["session_ids"])


def _write_fact_module(code_dir: Path, record: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> None:
    """Write exactly one atomic function into its own facts/<function>.py file."""

    path = code_dir / Path(*record["module"].split(".")).with_suffix(".py")
    lines = ["from code.memory_runtime import MemoryValue, fact, source, topics", ""]
    predecessor = str(record.get("supersedes_fact_id") or "")
    if predecessor in by_id:
        previous = by_id[predecessor]
        lines.extend([
            f"from .{previous['function_name']} import {previous['function_name']}",
            "",
        ])
    supersedes = by_id[predecessor]["function_name"] if predecessor in by_id else "None"
    lines.append(
        "@fact("
        f"fact_id={record['fact_id']!r}, predicate={record['predicate']!r}, "
        f"fact_type={record['fact_type']!r}, "
        f"stateful={record['stateful']!r}, supersedes={supersedes})"
    )
    for item in reversed(record["sources"]):
        lines.append(f"@source(session={item['session_id']!r}, dialog={item['dialogue_id']!r})")
    if record["topics"]:
        lines.append("@topics(" + ", ".join(repr(topic) for topic in record["topics"]) + ")")
    lines.extend([
        f"def {record['function_name']}() -> MemoryValue:",
        f"    return {_value_constructor(record)}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _value_constructor(record: dict[str, Any]) -> str:
    value = record["value"]
    attributes = record.get("attributes", {})
    if not isinstance(attributes, dict):
        attributes = {}
    arguments = [
        f"subject={record['subject']!r}",
        f"time={record['normalized_time']!r}",
        f"description={record['fact_text']!r}",
        f"value={value!r}",
        f"attributes={attributes!r}",
    ]
    return "MemoryValue(" + ", ".join(arguments) + ")"


def _write_relation_module(code_dir: Path, records: list[dict[str, Any]], links: list[dict[str, str]]) -> None:
    by_id = {record["fact_id"]: record for record in records}
    required: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for link in links:
        for fact_id in (link["source"], link["target"]):
            record = by_id[fact_id]
            if record not in required[record["module"]]:
                required[record["module"]].append(record)
    aliases: dict[str, str] = {}
    lines = ["from code.memory_runtime import related_to", ""]
    for module in sorted(required):
        imports = []
        for record in sorted(required[module], key=lambda item: item["fact_id"]):
            alias = f"f_{_safe_identifier(record['fact_id'])}"
            aliases[record["fact_id"]] = alias
            imports.append(f"{record['function_name']} as {alias}")
        lines.append(f"from .{module} import " + ", ".join(imports))
    if len(lines) > 2:
        lines.append("")
    for link in links:
        lines.append(
            f"{aliases[link['source']]} = related_to({aliases[link['target']]}, "
            f"relation={link['relation']!r})({aliases[link['source']]})"
        )
    (code_dir / "relations.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_concept_modules(
    code_dir: Path, records: list[dict[str, Any]], derived_groups: dict[tuple[str, str], list[dict[str, Any]]],
) -> None:
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_subject[record["subject"]].append(record)
    for subject, items in by_subject.items():
        lines = ["from code.memory_runtime import history, latest", ""]
        _append_fact_imports(lines, items)
        lines.extend(["", "MEMORIES = ("])
        lines.extend(f"    {record['function_name']}," for record in items)
        lines.extend([")", ""])
        for (group_subject, predicate), chain in sorted(derived_groups.items()):
            if group_subject != subject:
                continue
            names = ", ".join(record["function_name"] for record in chain)
            suffix = _safe_identifier(predicate)
            lines.extend([
                f"def current_{suffix}(as_of=None):",
                f"    return latest({names}, as_of=as_of)",
                "",
                f"def history_{suffix}(as_of=None):",
                f"    return history({names}, as_of=as_of)",
                "",
            ])
        (code_dir / "concept" / f"person_{_safe_identifier(subject)}.py").write_text("\n".join(lines), encoding="utf-8")


def _write_topic_concept_modules(code_dir: Path, records: list[dict[str, Any]]) -> None:
    """Render topic views in the same concept namespace as person views."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for topic in record["topics"]:
            grouped[_safe_identifier(topic)].append(record)
    for topic, items in grouped.items():
        lines: list[str] = []
        _append_fact_imports(lines, items)
        lines.extend(["", "MEMORIES = ("])
        lines.extend(f"    {record['function_name']}," for record in items)
        lines.append(")")
        (code_dir / "concept" / f"topic_{topic}.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_view_modules(
    code_dir: Path, package: str, records: list[dict[str, Any]], keys,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for key in keys(record):
            grouped[_safe_identifier(key)].append(record)
    for key, items in grouped.items():
        lines: list[str] = []
        _append_fact_imports(lines, items)
        lines.extend(["", "MEMORIES = ("])
        lines.extend(f"    {record['function_name']}," for record in items)
        lines.append(")")
        (code_dir / package / f"{key}.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_fact_imports(lines: list[str], records: list[dict[str, Any]]) -> None:
    by_module: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_module[record["module"]].append(record["function_name"])
    for module, names in sorted(by_module.items()):
        lines.append(f"from ..{module} import " + ", ".join(names))


def _function_index(
    records: list[dict[str, Any]], derived_groups: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Write only the non-derivable fields needed to find executable code.

    ``entrypoint`` encodes both module/code location and function name.  The
    runtime derives module, file path, kind, base capabilities, and canonical
    base-fact ID from it/``id`` instead of serialising duplicate views.
    """

    out = []
    for record in records:
        search = " ".join([
            record["subject"], record["function_name"].replace("_", " "),
            record["predicate"], *record["topics"],
        ])
        out.append({
            "id": f"fact.{record['fact_id']}",
            "entrypoint": f"{record['module']}:{record['function_name']}",
            "subject": record["subject"],
            "predicate": record["predicate"],
            "stateful": record["stateful"],
            "search": search,
        })
    for (subject, predicate), chain in sorted(derived_groups.items()):
        module, suffix = f"concept.person_{_safe_identifier(subject)}", _safe_identifier(predicate)
        for operation in ("current", "history"):
            function_name = f"{operation}_{suffix}"
            out.append({
                "id": f"concept.person_{_safe_identifier(subject)}.{function_name}",
                "entrypoint": f"{module}:{function_name}",
                "subject": subject,
                "predicate": predicate,
                "operation": operation,
                "args": ["as_of"],
                "search": f"{subject} {operation} {predicate} active state temporal",
            })
    return out


def _view_index(
    records: list[dict[str, Any]], derived_groups: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return compact concept/session membership for executable view expansion.

    The generated ``concept`` and ``sessions`` modules are the source-level
    views.  This index is their small runtime navigation companion: an
    ``entrypoint`` is enough to derive the code path, while ``members`` lists
    only callable IDs already described by ``function_index.json``.  In
    particular it never repeats a canonical fact's natural-language body.
    """

    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_subject[record["subject"]].append(record)
        for topic in record["topics"]:
            by_topic[_safe_identifier(topic)].append(record)
        for session_id in record["session_ids"]:
            by_session[_safe_identifier(session_id)].append(record)

    derived_by_subject: dict[str, list[str]] = defaultdict(list)
    for (subject, predicate) in sorted(derived_groups):
        subject_id, predicate_id = _safe_identifier(subject), _safe_identifier(predicate)
        for operation in ("current", "history"):
            derived_by_subject[subject].append(
                f"concept.person_{subject_id}.{operation}_{predicate_id}"
            )

    views: list[dict[str, Any]] = []

    def append_view(entrypoint: str, items: list[dict[str, Any]], extra_members: list[str] | None = None) -> None:
        members = [f"fact.{item['fact_id']}" for item in items]
        members.extend(extra_members or [])
        # A merged fact can be present through multiple sources; each module
        # should still expose its function only once and preserve code order.
        unique_members = list(dict.fromkeys(members))
        if unique_members:
            views.append({"entrypoint": entrypoint, "members": unique_members})

    for subject, items in sorted(by_subject.items(), key=lambda item: _safe_identifier(item[0])):
        append_view(
            f"concept.person_{_safe_identifier(subject)}",
            items,
            derived_by_subject.get(subject, []),
        )
    for topic, items in sorted(by_topic.items()):
        append_view(f"concept.topic_{topic}", items)
    for session, items in sorted(by_session.items()):
        append_view(f"sessions.{session}", items)

    return sorted(views, key=lambda item: str(item["entrypoint"]))


def _read_semantic_specs(path: Path, sample_id: str, fingerprint: str, facts_sha256: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    if (
        payload.get("schema") == "semantic_fact_specs_v5"
        and payload.get("sample_id") == sample_id
        and (
            payload.get("compiler_fingerprint") == fingerprint
            or _resume_ignores_llm_configuration()
        )
        and payload.get("facts_by_id_sha256") == facts_sha256
        and isinstance(payload.get("facts"), list)
    ):
        return payload
    return None


def _fallback_function_name(fact: dict[str, Any], predicate: str) -> str:
    subject_tokens = set(_normalise(fact.get("subject", "")).split())
    tokens = [token for token in _normalise(fact.get("fact_text", "")).split() if token not in subject_tokens]
    return _safe_function_name("_".join(tokens[:6])) or f"fact_{_safe_identifier(fact.get('fact_id') or predicate)}"


def _safe_function_name(value: Any) -> str:
    result = _safe_identifier(value)
    return "" if result == "other" or keyword.iskeyword(result) else result


def _safe_identifier(value: Any) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "other").lower()).strip("_")
    if not cleaned:
        cleaned = "other"
    return f"v_{cleaned}" if cleaned[0].isdigit() else cleaned


def _normalise(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def _read_canonical_facts(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"facts_by_id.json must contain an object: {path}")
    facts = []
    for fact_id, raw in payload.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Canonical fact {fact_id!r} must be an object")
        fact = dict(raw)
        fact["fact_id"] = str(fact_id)
        facts.append(fact)
    return facts


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_code_dir(path: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted(path.rglob("*.py")):
        digest.update(str(source.relative_to(path)).encode("utf-8"))
        digest.update(source.read_bytes())
    return digest.hexdigest()
