"""Parallel code/keyword retrieval for programmatic Memory-as-Code QA.

The expensive LLM term extraction and Cross-Encoder calls run in the parent
process. The resulting bundle is immutable input to the sandboxed search
program, which materialises selected executable memories through required SDK
calls placed by the planner.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from .fact_reranker import rank_facts_with_cross_encoder
from .llm_client import LLMClient
from .prompts import (
    RETRIEVAL_TERM_EXTRACTOR_SYSTEM,
    RETRIEVAL_TERM_EXTRACTOR_USER,
    fill_template,
)
from .utils import estimate_text_tokens, format_fact_line, read_json


CODE_RECALL_LIMIT = 24
CODE_RECALL_PREFILTER_LIMIT = 320
HIGH_COVERAGE_CODE_PREFILTER_LIMIT = 768
VIEW_EXPAND_LIMIT = 12
VIEW_EXPAND_PREFILTER_LIMIT = 96
KEYWORD_RECALL_LIMIT = 32
SOURCE_FACT_LIMIT = 12
SUPPLEMENTAL_SOURCE_FACT_LIMIT = 16
HIGH_COVERAGE_SUPPLEMENTAL_SOURCE_FACT_LIMIT = 24
SOURCE_WINDOW = 2


def prepare_retrieval_bundle(
    sample_memory_dir: str | Path,
    sample: dict[str, Any],
    question: str,
    llm: LLMClient,
    *,
    category: str = "",
    question_date: str = "",
) -> dict[str, Any]:
    """Build the base evidence routes for one question.

    Code recall and LLM keyword-term extraction deliberately begin in parallel.
    Scope expansion starts from selected code functions. Graph traversal is
    deferred to ``memory.expand_relation()`` when a generated program needs it.
    """

    sample_dir = Path(sample_memory_dir)
    facts_by_id = _canonical_facts(sample_dir)
    function_index = _function_index(sample_dir)
    view_index = _view_index(sample_dir)
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="code-recall") as pool:
        code_future = pool.submit(_code_recall, question, function_index, category)
        terms = extract_retrieval_terms(question, llm)
        code_candidates, code_metadata = code_future.result()
    keyword_matches = keyword_recall_facts(facts_by_id, terms, limit=KEYWORD_RECALL_LIMIT)

    code_fact_ids, code_execution = _materialise_code_candidates(
        sample_dir, sample, code_candidates, facts_by_id
    )
    view_candidates, view_metadata = _view_expand_candidates(
        question,
        code_candidates,
        function_index,
        view_index,
        facts_by_id,
    )
    view_fact_ids, view_execution = _materialise_code_candidates(
        sample_dir, sample, view_candidates, facts_by_id
    )
    keyword_fact_ids = _dedupe_ids(item["fact_id"] for item in keyword_matches)
    merged_fact_ids = _dedupe_ids(code_fact_ids + view_fact_ids + keyword_fact_ids)
    base_facts = [facts_by_id[fact_id] for fact_id in merged_fact_ids if fact_id in facts_by_id]
    context = build_source_context(
        sample, base_facts, question=question, terms=terms,
        category=category, question_date=question_date,
    )

    origins = {}
    for fact_id in merged_fact_ids:
        origins[fact_id] = {
            "code_recall": fact_id in set(code_fact_ids),
            "view_expand": fact_id in set(view_fact_ids),
            "keyword_recall": fact_id in set(keyword_fact_ids),
        }
    return {
        "schema": "query_bundle_v4",
        "question": question,
        "question_context": {"category": category, "question_date": question_date},
        "resolve": {
            "candidate_functions": code_candidates,
            "function_ids": [str(item.get("id", "")) for item in code_candidates],
            "fact_ids": code_fact_ids,
            "metadata": code_metadata,
            "execution": code_execution,
        },
        "expand_scope": {
            "seed_function_ids": [str(item.get("id", "")) for item in code_candidates],
            "view_entrypoints": list(view_metadata.get("matched_view_entrypoints", [])),
            "candidate_function_ids": list(view_metadata.get("eligible_function_ids", [])),
            "candidate_functions": view_candidates,
            "function_ids": [str(item.get("id", "")) for item in view_candidates],
            "fact_ids": view_fact_ids,
            "metadata": view_metadata,
            "execution": view_execution,
        },
        "match": {
            "terms": terms,
            "matches": keyword_matches,
            "fact_ids": keyword_fact_ids,
        },
        "base": {"fact_ids": merged_fact_ids, "origins": origins},
        "final": {
            "fact_ids": merged_fact_ids,
            "facts": base_facts,
            "ranking": {"strategy": "base_code_scope_keyword_only"},
        },
        "context": context,
    }


def extract_retrieval_terms(question: str, llm: LLMClient) -> dict[str, Any]:
    if not str(question or "").strip():
        return _normalise_terms({})
    payload = llm.chat_json(
        RETRIEVAL_TERM_EXTRACTOR_SYSTEM,
        fill_template(RETRIEVAL_TERM_EXTRACTOR_USER, question=question),
    )
    return _augment_terms_for_question(question, _normalise_terms(payload))


def keyword_recall_facts(
    facts_by_id: dict[str, dict[str, Any]], terms: dict[str, Any], *, limit: int,
) -> list[dict[str, Any]]:
    """Use the exact term groups and weights of the established MAC retriever."""

    normalised = _normalise_terms(terms)
    scored = []
    for fact_id, fact in facts_by_id.items():
        score, matched_terms = _score_fact_for_terms(fact, normalised)
        if score > 0:
            scored.append({
                "fact_id": fact_id,
                "score": round(score, 4),
                "matched_terms": matched_terms,
            })
    scored.sort(key=lambda item: (-float(item["score"]), str(item["fact_id"])))
    return scored[: max(0, int(limit))]


def build_answer_context(
    bundle: dict[str, Any],
    program_result: dict[str, Any] | list[dict[str, Any]] | None = None,
    *,
    program_trace: dict[str, Any] | None = None,
    canonical_facts_by_id: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Render answer context from the program-hydrated evidence when present."""

    final = bundle.get("final", {}) if isinstance(bundle, dict) else {}
    facts = list(final.get("facts", [])) if isinstance(final, dict) else []
    context = bundle.get("context", {}) if isinstance(bundle, dict) else {}
    source_lines = list(context.get("source_lines", [])) if isinstance(context, dict) else []
    runtime_context = (
        program_result.get("value", {})
        if isinstance(program_result, dict) and isinstance(program_result.get("value"), dict)
        else {}
    )
    program_source_lines = list(
        runtime_context.get("source_lines", runtime_context.get("program_source_lines", []))
    )
    program_facts = list(
        runtime_context.get("facts", runtime_context.get("program_facts", []))
    )
    if not program_facts and isinstance(canonical_facts_by_id, dict):
        for fact_id in _program_result_provenance(program_result).get("fact_ids", []):
            fact = canonical_facts_by_id.get(str(fact_id))
            if isinstance(fact, dict):
                program_facts.append(dict(fact))
    if program_facts:
        facts = program_facts
    if program_source_lines:
        source_lines = program_source_lines
    temporal_computations = list(context.get("temporal_computations", [])) if isinstance(context, dict) else []
    lines: list[str] = []
    question_context = bundle.get("question_context", {}) if isinstance(bundle, dict) else {}
    if isinstance(question_context, dict) and any(question_context.values()):
        lines.extend([
            "# Question Context", "",
            f"- category: {question_context.get('category', '')}",
            f"- question_date: {question_context.get('question_date', '')}",
            "- question_date is the temporal cutoff for current/latest-state questions.",
            "",
        ])
    if source_lines:
        lines.extend([
            "# Original Dialogue Evidence Near Top-Ranked Facts", "",
            "Use original lines when their person, date, event, and requested attribute match the question.",
            "",
            *source_lines,
        ])
    lines.extend(["", "# Deduplicated Raw Facts", ""])
    lines.extend(format_fact_line(fact) for fact in facts)
    if temporal_computations:
        lines.extend(["", "# Deterministic Temporal Calculations", "", "These values are computed from the cited session timestamps; use them only when the question requests that operation.", json.dumps(temporal_computations, ensure_ascii=False, indent=2)])
    trace = _answer_safe_program_trace(program_result, program_trace)
    if trace:
        lines.extend([
            "",
            "# Executed Search Trace (Non-authoritative)",
            "",
            "This trace records how evidence was located. It is not an additional fact and must not override the original dialogue or raw facts above.",
            json.dumps(trace, ensure_ascii=False, indent=2),
        ])
    return "\n".join(lines)


def _answer_safe_program_trace(
    program_result: dict[str, Any] | list[dict[str, Any]] | None,
    program_trace: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return provenance only; never expose an executable result's value/text."""

    trace = dict(program_trace) if isinstance(program_trace, dict) else {}
    operations = [
        str(value) for value in trace.get("operations", [])
        if str(value).strip()
    ]
    provenance = trace.get("result_provenance")
    if not isinstance(provenance, dict):
        provenance = _program_result_provenance(program_result)
    clean = {
        field: list(dict.fromkeys(str(value) for value in provenance.get(field, []) if str(value).strip()))
        for field in ("fact_ids", "dialogue_ids", "rules")
    }
    if not operations and not any(clean.values()):
        return None
    return {
        "schema": "answer_safe_program_trace_v1",
        "operations": list(dict.fromkeys(operations)),
        "result_provenance": clean,
    }


def _program_result_provenance(value: Any) -> dict[str, list[str]]:
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


def _canonical_facts(sample_dir: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(sample_dir / "facts_by_id.json")
    if not isinstance(payload, dict):
        raise ValueError("facts_by_id.json must be an object keyed by fact_id")
    return {str(key): dict(value) for key, value in payload.items() if isinstance(value, dict)}


def _function_index(sample_dir: Path) -> list[dict[str, Any]]:
    payload = read_json(sample_dir / "function_index.json")
    if payload.get("schema") != "executable_function_index_v2":
        raise ValueError("Unsupported function_index.json schema")
    values = payload.get("functions", [])
    return [dict(item) for item in values if isinstance(item, dict)]


def _view_index(sample_dir: Path) -> list[dict[str, Any]]:
    """Read compact code-view memberships without loading fact text."""

    payload = read_json(sample_dir / "view_index.json")
    if payload.get("schema") != "executable_view_index_v1":
        raise ValueError("Unsupported view_index.json schema")
    views = []
    for raw in payload.get("views", []):
        if not isinstance(raw, dict):
            continue
        entrypoint = str(raw.get("entrypoint", "")).strip()
        members = [str(value) for value in raw.get("members", []) if str(value).strip()]
        if entrypoint and members:
            views.append({"entrypoint": entrypoint, "members": list(dict.fromkeys(members))})
    return views


def _code_recall(question: str, functions: list[dict[str, Any]], category: str = "") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    retrieval_question = _expanded_retrieval_query(question)
    prefilter_limit = HIGH_COVERAGE_CODE_PREFILTER_LIMIT if _needs_high_coverage(question, category) else CODE_RECALL_PREFILTER_LIMIT
    lexical = _rank_records(retrieval_question, functions, limit=prefilter_limit)
    projected = []
    for item in lexical:
        projected.append({
            "fact_id": str(item.get("id", "")),
            "fact_text": str(item.get("search") or ""),
            "speaker": str(item.get("subject", "")),
            "subject": str(item.get("subject", "")),
            "topics": [str(item.get("predicate") or ""), str(item.get("operation") or "")],
            "normalized_time": "",
            "dialogue_ids": [],
        })
    reranked, metadata = rank_facts_with_cross_encoder(retrieval_question, projected)
    lookup = {str(item.get("id", "")): item for item in lexical}
    selected = []
    for scored in reranked[:CODE_RECALL_LIMIT]:
        item = dict(lookup.get(str(scored.get("fact_id", "")), {}))
        if item:
            item["_cross_encoder_score"] = scored.get("_cross_encoder_score")
            selected.append(item)
    metadata["strategy"] = "function_index_lexical_cross_encoder_v1"
    metadata["lexical_prefilter_count"] = len(lexical)
    metadata["query_expansions"] = _deterministic_query_expansions(question)
    return selected, metadata


def _view_expand_candidates(
    question: str,
    code_candidates: list[dict[str, Any]],
    functions: list[dict[str, Any]],
    views: list[dict[str, Any]],
    facts_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expand selected code through the concept/session modules that contain it.

    The compiler derived ``views`` from the exact ``MEMORIES`` imports and
    person-module derived functions.  Therefore this is code-view expansion,
    not a second graph relation.  The canonical fact body is read only to rank
    executable members after their containing modules have been identified.
    """

    seed_ids = [str(item.get("id", "")) for item in code_candidates if item.get("id")]
    seed_set = set(seed_ids)
    matched_views = [
        view for view in views
        if seed_set.intersection(str(member) for member in view.get("members", []))
    ]
    eligible_ids: list[str] = []
    for view in matched_views:
        for function_id in view.get("members", []):
            function_id = str(function_id)
            if function_id and function_id not in seed_set and function_id not in eligible_ids:
                eligible_ids.append(function_id)

    by_id = {str(item.get("id", "")): item for item in functions}
    eligible = [dict(by_id[function_id]) for function_id in eligible_ids if function_id in by_id]
    lexical = _rank_view_records(
        question,
        eligible,
        facts_by_id,
        limit=VIEW_EXPAND_PREFILTER_LIMIT,
    )
    projected = [_project_view_record(item, facts_by_id) for item in lexical]
    reranked, metadata = rank_facts_with_cross_encoder(question, projected)
    lookup = {str(item.get("id", "")): item for item in lexical}
    selected: list[dict[str, Any]] = []
    for scored in reranked[:VIEW_EXPAND_LIMIT]:
        function_id = str(scored.get("fact_id", ""))
        record = lookup.get(function_id)
        if record:
            selected_record = dict(record)
            selected_record["_cross_encoder_score"] = scored.get("_cross_encoder_score")
            selected.append(selected_record)

    metadata["strategy"] = "code_view_membership_lexical_cross_encoder_v1"
    metadata["seed_function_ids"] = seed_ids
    metadata["matched_view_entrypoints"] = [str(view["entrypoint"]) for view in matched_views]
    metadata["eligible_function_ids"] = eligible_ids
    metadata["lexical_prefilter_function_ids"] = [str(item.get("id", "")) for item in lexical]
    metadata["selected_function_count"] = len(selected)
    return selected, metadata


def _project_view_record(record: dict[str, Any], facts_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Adapt an executable member for the existing fact Cross-Encoder API."""

    function_id = str(record.get("id", ""))
    canonical_id = function_id.removeprefix("fact.") if function_id.startswith("fact.") else ""
    fact = facts_by_id.get(canonical_id, {})
    text = str(fact.get("fact_text") or record.get("search") or "")
    return {
        "fact_id": function_id,
        "fact_text": text,
        "speaker": str(fact.get("speaker") or record.get("subject") or ""),
        "subject": str(fact.get("subject") or record.get("subject") or ""),
        "topics": list(fact.get("topics") or [str(record.get("predicate") or ""), str(record.get("operation") or "")]),
        "normalized_time": str(fact.get("normalized_time") or ""),
        "dialogue_ids": list(fact.get("dialogue_ids") or []),
    }


def _rank_view_records(
    question: str,
    records: list[dict[str, Any]],
    facts_by_id: dict[str, dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    question_tokens = _tokens(question)
    scored = []
    for index, record in enumerate(records):
        function_id = str(record.get("id", ""))
        canonical_id = function_id.removeprefix("fact.") if function_id.startswith("fact.") else ""
        fact = facts_by_id.get(canonical_id, {})
        text = " ".join((str(record.get("search") or ""), str(fact.get("fact_text") or "")))
        overlap = len(question_tokens & _tokens(text))
        if overlap:
            scored.append((overlap, index, record))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored:
        return [dict(record) for record in records[: max(0, int(limit))]]
    return [dict(record) for _, _, record in scored[: max(0, int(limit))]]


def _materialise_code_candidates(
    sample_dir: Path,
    sample: dict[str, Any],
    candidates: list[dict[str, Any]],
    facts_by_id: dict[str, dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Actually execute selected code before rehydrating canonical facts."""

    # Local import avoids a module-import cycle: the SDK itself has no
    # dependency on this parent-process retrieval pipeline.
    from .executable_memory import MemorySDK

    sdk = MemorySDK(sample_dir, source_dialogue=sample)
    fact_ids: list[str] = []
    execution: list[dict[str, Any]] = []
    for candidate in candidates:
        function_id = str(candidate.get("id", ""))
        try:
            evidence = sdk.execute(function_id)
            returned = [fact_id for fact_id in evidence.fact_ids if fact_id in facts_by_id]
            fact_ids.extend(returned)
            execution.append({"function_id": function_id, "status": "ok", "returned_fact_ids": returned})
        except Exception as exc:
            execution.append({"function_id": function_id, "status": "error", "error": f"{exc.__class__.__name__}: {exc}"})
    return _dedupe_ids(fact_ids), execution


def _rank_records(question: str, records: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    question_tokens = _tokens(question)
    scored = []
    for index, record in enumerate(records):
        score = len(question_tokens & _tokens(record.get("search") or ""))
        if score:
            scored.append((score, index, record))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored:
        return [dict(record) for record in records[:limit]]
    return [dict(record) for _, _, record in scored[:limit]]


def build_source_context(
    sample: dict[str, Any], ranked_facts: list[dict[str, Any]], *, question: str = "", terms: dict[str, Any] | None = None, category: str = "", question_date: str = "",
) -> dict[str, Any]:
    source_index = _source_dialogue_index(sample)
    anchor_facts = _select_source_anchors(
        ranked_facts, question, _normalise_terms(terms or {}), source_index,
        category=category,
    )
    lines, dialogue_ids = [], []
    seen: set[str] = set()
    rank_by_id = {str(fact.get("fact_id", "")): rank for rank, fact in enumerate(ranked_facts, start=1)}
    for fact in anchor_facts:
        fact_rank = rank_by_id.get(str(fact.get("fact_id", "")), 0)
        for raw_id in fact.get("dialogue_ids") or [fact.get("dia_id", "")]:
            parsed = _parse_dialogue_id(raw_id)
            if parsed is None:
                continue
            session, turn = parsed
            for neighbor in range(max(1, turn - SOURCE_WINDOW), turn + SOURCE_WINDOW + 1):
                source = source_index.get((session, neighbor))
                if not source:
                    continue
                dia_id = str(source.get("dia_id") or f"D{session}:{neighbor}")
                if dia_id in seen:
                    continue
                seen.add(dia_id)
                dialogue_ids.append(dia_id)
                text = re.sub(r"\s+", " ", str(source.get("text") or "")).strip()
                details = text or "(non-text message)"
                session_date = str(source.get("session_datetime") or "").strip()
                date_note = f" | session_datetime={session_date}" if session_date else ""
                visual = "; ".join(value for value in (str(source.get("blip_caption") or "").strip(), str(source.get("query") or "").strip()) if value)
                visual_note = f" | visual={visual}" if visual else ""
                lines.append(
                    f"- [{dia_id} | rank={fact_rank} | neighbor={neighbor - turn:+d}{date_note}{visual_note} | "
                    f"{source.get('speaker') or 'unknown'}] {details}"
                )
    return {
        "source_lines": lines,
        "dialogue_ids": dialogue_ids,
        "anchor_fact_ids": [str(item.get("fact_id", "")) for item in anchor_facts],
        "window": SOURCE_WINDOW,
        "estimated_tokens": estimate_text_tokens("\n".join(lines)),
        "question_date": question_date,
        "temporal_computations": _temporal_computations(
            question, anchor_facts, source_index, question_date=question_date
        ),
    }


def _temporal_computations(
    question: str,
    anchor_facts: list[dict[str, Any]],
    source_index: dict[tuple[int, int], dict[str, Any]],
    *,
    question_date: str = "",
) -> list[dict[str, Any]]:
    computations = _relative_time_from_question_date(
        question, anchor_facts, source_index, question_date
    )
    computations.extend(_elapsed_days_between_events(question, anchor_facts, source_index))
    computations.extend(_ordered_events(question, anchor_facts, source_index))
    computations.extend(_direct_recent_onset_time(question, anchor_facts, source_index))
    if not re.search(r"\bafter how many weeks\b", question, re.I):
        return [*computations, *_plan_followup_time_intersection(question, anchor_facts, source_index)]
    points = []
    seen_dates: set[datetime] = set()
    for fact in anchor_facts:
        for raw_id in fact.get("dialogue_ids") or [fact.get("dia_id", "")]:
            parsed = _parse_dialogue_id(raw_id)
            source = source_index.get(parsed) if parsed else None
            date = _parse_session_datetime(source.get("session_datetime")) if source else None
            if date and date not in seen_dates:
                points.append((date, str(raw_id), str(fact.get("fact_id", ""))))
                seen_dates.add(date)
                break
        if len(points) >= 2:
            break
    if len(points) < 2:
        return [*computations, *_plan_followup_time_intersection(question, anchor_facts, source_index)]
    start, end = sorted(points, key=lambda item: item[0])
    elapsed_days = (end[0] - start[0]).days
    return [*computations, {"operation": "elapsed_whole_weeks_between_reporting_sessions", "from": start[0].date().isoformat(), "to": end[0].date().isoformat(), "elapsed_days": elapsed_days, "whole_weeks": elapsed_days // 7, "answer_text": f"{elapsed_days // 7} weeks", "support_dialogue_ids": [start[1], end[1]], "support_fact_ids": [start[2], end[2]]}, *_plan_followup_time_intersection(question, anchor_facts, source_index)]


_EVENT_MATCH_STOPWORDS = {
    "a", "an", "and", "at", "between", "day", "did", "do", "from", "happened",
    "how", "i", "in", "last", "many", "me", "my", "of", "on", "passed", "the",
    "then", "to", "was", "weeks", "week", "days", "ago", "which", "first",
}


def _fact_source_date(fact, source_index):
    for raw_id in fact.get("dialogue_ids") or [fact.get("dia_id", "")]:
        parsed = _parse_dialogue_id(raw_id)
        source = source_index.get(parsed) if parsed else None
        date = _parse_session_datetime(source.get("session_datetime")) if source else None
        if date:
            return date, str(raw_id)
    return None


def _event_match_score(query, fact):
    query_tokens = _tokens(query) - _EVENT_MATCH_STOPWORDS
    if not query_tokens:
        return 0.0
    text = " ".join([str(fact.get("fact_text") or ""), str(fact.get("subject") or ""), " ".join(str(topic) for topic in fact.get("topics", []))])
    matched = query_tokens & _tokens(text)
    return sum(2.0 if len(token) >= 6 else 1.0 for token in matched) / max(1, len(query_tokens))


def _best_dated_fact(query, facts, source_index, *, excluded_fact_ids=None):
    excluded = excluded_fact_ids or set()
    ranked = []
    for index, fact in enumerate(facts):
        fact_id = str(fact.get("fact_id", ""))
        dated = None if fact_id in excluded else _fact_source_date(fact, source_index)
        score = _event_match_score(query, fact)
        if dated and score > 0:
            ranked.append((score, -index, fact, dated[0], dated[1]))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], -item[1]))
    best = ranked[0]
    return best[2], best[3], best[4]


def _relative_time_from_question_date(question, anchor_facts, source_index, question_date):
    match = re.search(r"\bhow many (days?|weeks?|months?) ago\b", question, re.I)
    query_date = _parse_session_datetime(question_date)
    selected = _best_dated_fact(question, anchor_facts, source_index)
    if not match or not query_date or not selected:
        return []
    fact, event_date, dialogue_id = selected
    elapsed_days = (query_date.date() - event_date.date()).days
    if elapsed_days < 0:
        return []
    unit = match.group(1).lower()
    if unit.startswith("day"):
        value, label = elapsed_days, "days"
    elif unit.startswith("week"):
        value, label = elapsed_days // 7, "weeks"
    else:
        value = (query_date.year - event_date.year) * 12 + query_date.month - event_date.month
        if query_date.day < event_date.day:
            value -= 1
        value, label = max(0, value), "months"
    return [{"operation": "elapsed_relative_to_question_date", "from": event_date.date().isoformat(), "to": query_date.date().isoformat(), "elapsed_days": elapsed_days, "answer_text": f"{value} {label}", "support_dialogue_ids": [dialogue_id], "support_fact_ids": [str(fact.get("fact_id", ""))]}]


def _elapsed_days_between_events(question, anchor_facts, source_index):
    match = re.search(r"\bhow many days (?:passed )?between (?:the day )?(.*?) and (?:the day )?(.*?)(?:\?|$)", question, re.I)
    if not match:
        return []
    first = _best_dated_fact(match.group(1), anchor_facts, source_index)
    if not first:
        return []
    first_id = str(first[0].get("fact_id", ""))
    second = _best_dated_fact(match.group(2), anchor_facts, source_index, excluded_fact_ids={first_id})
    if not second:
        return []
    elapsed = abs((second[1].date() - first[1].date()).days)
    return [{"operation": "elapsed_days_between_matched_events", "from": first[1].date().isoformat(), "to": second[1].date().isoformat(), "elapsed_days": elapsed, "inclusive_days": elapsed + 1, "answer_text": f"{elapsed} days", "inclusive_answer_text": f"{elapsed + 1} days (including the last day)", "support_dialogue_ids": [first[2], second[2]], "support_fact_ids": [first_id, str(second[0].get("fact_id", ""))]}]


def _ordered_events(question, anchor_facts, source_index):
    if not re.search(r"\border from first to last\b", question, re.I) or ":" not in question:
        return []
    body = question.split(":", 1)[1].rstrip(" ?")
    clauses = [part.strip(" ,") for part in re.split(r",\s*(?:and\s+)?|\s+and\s+", body) if part.strip(" ,")]
    if len(clauses) < 2:
        return []
    selected, excluded = [], set()
    for clause in clauses:
        item = _best_dated_fact(clause, anchor_facts, source_index, excluded_fact_ids=excluded)
        if not item:
            return []
        fact_id = str(item[0].get("fact_id", ""))
        excluded.add(fact_id)
        selected.append((item[1], clause, fact_id, item[2]))
    selected.sort(key=lambda item: item[0])
    return [{"operation": "order_matched_events_by_session_datetime", "ordered_events": [item[1] for item in selected], "answer_text": "; then ".join(item[1] for item in selected), "support_fact_ids": [item[2] for item in selected], "support_dialogue_ids": [item[3] for item in selected]}]


def _direct_recent_onset_time(question: str, anchor_facts: list[dict[str, Any]], source_index: dict[tuple[int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve a directly reported recent career onset at session granularity."""
    normalized_question = _normalise(question)
    if not (re.search(r"\bwhen\b", normalized_question) and re.search(r"\b(start|started|begin|began)\b", normalized_question) and re.search(r"\b(professional|professionally|career)\b", normalized_question)):
        return []
    for fact in anchor_facts:
        subject = _normalise(fact.get("subject") or fact.get("speaker") or "")
        fact_text = _normalise(fact.get("fact_text") or "")
        if subject and subject not in normalized_question:
            continue
        if not (re.search(r"\b(signed|joined|drafted|started|began)\b", fact_text) and re.search(r"\b(team|club|professional|career)\b", fact_text)):
            continue
        for raw_id in fact.get("dialogue_ids") or [fact.get("dia_id", "")]:
            parsed = _parse_dialogue_id(raw_id)
            source = source_index.get(parsed) if parsed else None
            source_text = _normalise(source.get("text") or "") if source else ""
            if not re.search(r"\b(?:just|recently)\b.{0,35}\b(?:signed|joined|started|began)\b|\b(?:signed|joined|started|began)\b.{0,35}\b(?:just|recently)\b", source_text):
                continue
            date = _parse_session_datetime(source.get("session_datetime"))
            if not date:
                continue
            return [{"operation": "direct_recent_onset_from_reporting_session", "answer_text": date.strftime("%B %Y"), "support_dialogue_ids": [str(raw_id)], "support_fact_ids": [str(fact.get("fact_id", ""))], "derivation": "a directly reported recent signing/joining event uses its reporting-session month"}]
    return []


def _plan_followup_time_intersection(question: str, anchor_facts: list[dict[str, Any]], source_index: dict[tuple[int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    if not re.search(r"\bwhen\b", question, re.I):
        return []
    stop = {"when", "did", "was", "were", "is", "in", "for", "the", "a", "an", "to", "of"}
    question_tokens = _tokens(question) - stop
    sources = []
    seen = set()
    for fact in anchor_facts:
        for raw_id in fact.get("dialogue_ids") or [fact.get("dia_id", "")]:
            parsed = _parse_dialogue_id(raw_id)
            if not parsed:
                continue
            for neighbor in range(max(1, parsed[1] - SOURCE_WINDOW), parsed[1] + SOURCE_WINDOW + 1):
                source = source_index.get((parsed[0], neighbor))
                dialogue_id = str(source.get("dia_id", "")) if source else ""
                date = _parse_session_datetime(source.get("session_datetime")) if source else None
                if source and date and dialogue_id not in seen:
                    sources.append((source, date)); seen.add(dialogue_id)
    for planned, plan_date in sources:
        plan_text = str(planned.get("text") or "")
        if not re.search(r"\bnext month\b", plan_text, re.I) or not (question_tokens & _tokens(plan_text)):
            continue
        target_year = plan_date.year + (1 if plan_date.month == 12 else 0)
        target_month = 1 if plan_date.month == 12 else plan_date.month + 1
        for report, report_date in sources:
            report_text = str(report.get("text") or "")
            if not re.search(r"\blast week\b", report_text, re.I) or str(report.get("speaker", "")) != str(planned.get("speaker", "")) or not (question_tokens & _tokens(report_text)):
                continue
            if (report_date.year, report_date.month) != (target_year, target_month):
                continue
            granularity = "early" if report_date.day <= 14 else "mid" if report_date.day <= 21 else "late"
            return [{"operation": "intersect_plan_and_followup_relative_time", "answer_text": f"{granularity} {report_date.strftime('%B %Y')}", "support_dialogue_ids": [str(planned.get("dia_id", "")), str(report.get("dia_id", ""))], "derivation": "next month plan intersected with last week follow-up"}]
    return []


def _parse_session_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    iso_like = re.search(r"\b(\d{4})/(\d{1,2})/(\d{1,2})(?:\s+\([A-Za-z]{3}\))?(?:\s+\d{1,2}:\d{2})?", text)
    if iso_like:
        try:
            return datetime(int(iso_like.group(1)), int(iso_like.group(2)), int(iso_like.group(3)))
        except ValueError:
            return None
    dashed = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if dashed:
        try:
            return datetime(int(dashed.group(1)), int(dashed.group(2)), int(dashed.group(3)))
        except ValueError:
            return None
    match = re.search(r"\bon\s+(\d{1,2}\s+[A-Za-z]+,\s+\d{4})\s*$", text, re.I)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%d %B, %Y")
    except ValueError:
        return None


def _needs_high_coverage(question: str, category: str = "") -> bool:
    text = str(question or "")
    hard_form = bool(re.search(r"\b(?:when|how long|how many|what date|what year|what month|which|what (?:are|were|kinds|types|places|items|books|games|hobbies|skills|ways))\b", text, re.I))
    explicit_period = bool(re.search(r"\b(?:(?:19|20)\d{2}|january|february|march|april|may|june|july|august|september|october|november|december)\b", text, re.I))
    return str(category) in {"multi-session", "temporal-reasoning", "knowledge-update"} or hard_form or explicit_period


def _select_source_anchors(
    ranked_facts: list[dict[str, Any]], question: str, terms: dict[str, Any], source_index: dict[tuple[int, int], dict[str, Any]], *, category: str = "",
) -> list[dict[str, Any]]:
    base = list(ranked_facts[:SOURCE_FACT_LIMIT])
    if not _needs_high_coverage(question, category):
        return base
    explicit_years = set(re.findall(r"\b(?:19|20)\d{2}\b", question))
    month_names = "january|february|march|april|may|june|july|august|september|october|november|december"
    explicit_months = set(re.findall(rf"\b(?:{month_names})\b", question.lower()))
    temporal = bool(re.search(r"\b(?:when|how long|week|month|year|date|before|after|during|since|as of)\b", question, re.I))
    exhaustive = bool(re.search(r"\b(?:how many|which|what (?:are|were|kinds|types|places|items|books|games|hobbies|skills|ways))\b", question, re.I))
    scored = []
    for rank, fact in enumerate(ranked_facts[SOURCE_FACT_LIMIT:], start=SOURCE_FACT_LIMIT + 1):
        score, _ = _score_fact_for_terms(fact, terms)
        source_times = []
        for raw_id in fact.get("dialogue_ids") or [fact.get("dia_id", "")]:
            parsed = _parse_dialogue_id(raw_id)
            if parsed and parsed in source_index:
                source_times.append(str(source_index[parsed].get("session_datetime") or ""))
        combined_time = " ".join([str(fact.get("normalized_time") or ""), *source_times]).lower()
        if temporal:
            score += 1.0 if combined_time.strip() else 0.0
            if explicit_years:
                score += 8.0 if any(year in combined_time for year in explicit_years) else -4.0
            if explicit_months:
                score += 8.0 if any(month in combined_time for month in explicit_months) else -4.0
        if exhaustive:
            score += max(0.0, 2.0 - rank / 100.0)
        if score > 0:
            scored.append((score, rank, fact))
    scored.sort(key=lambda item: (-item[0], item[1]))
    seen = {str(fact.get("fact_id", "")) for fact in base}
    for _, _, fact in scored:
        fact_id = str(fact.get("fact_id", ""))
        if fact_id not in seen:
            base.append(fact)
            seen.add(fact_id)
        supplemental_limit = HIGH_COVERAGE_SUPPLEMENTAL_SOURCE_FACT_LIMIT if _needs_high_coverage(question, category) else SUPPLEMENTAL_SOURCE_FACT_LIMIT
        if len(base) >= SOURCE_FACT_LIMIT + supplemental_limit:
            break
    return base


def _source_dialogue_index(sample: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    conversation = sample.get("conversation", {}) if isinstance(sample, dict) else {}
    index = {}
    for session_name, messages in conversation.items():
        match = re.fullmatch(r"session_(\d+)", str(session_name))
        if not match or not isinstance(messages, list):
            continue
        session = int(match.group(1))
        session_datetime = str(conversation.get(f"session_{session}_date_time") or "")
        for message in messages:
            parsed = _parse_dialogue_id(message.get("dia_id") if isinstance(message, dict) else "")
            if parsed is not None and isinstance(message, dict):
                index[parsed] = {**message, "session_datetime": session_datetime}
    return index


def _parse_dialogue_id(value: Any) -> tuple[int, int] | None:
    match = re.fullmatch(r"D(\d+):(\d+)", str(value or "").strip())
    return (int(match.group(1)), int(match.group(2))) if match else None


def _deterministic_query_expansions(question: str) -> list[str]:
    """Add only high-precision event paraphrases needed before semantic recall.

    These are retrieval aliases, not answer facts.  They bridge common question
    wording to how a directly observed event is typically recorded in memory.
    """

    normalized = _normalise(question)
    expansions: list[str] = []
    if (
        re.search(r"\b(start|started|begin|began)\b", normalized)
        and re.search(r"\b(professional|professionally|career)\b", normalized)
    ):
        expansions.extend([
            "signed with a new team",
            "joined a professional team",
            "drafted by a professional team",
            "professional debut",
        ])
    return expansions


def _expanded_retrieval_query(question: str) -> str:
    return " ".join([str(question or "").strip(), *_deterministic_query_expansions(question)]).strip()


def _augment_terms_for_question(question: str, terms: dict[str, Any]) -> dict[str, Any]:
    augmented = {key: list(value) if isinstance(value, list) else value for key, value in terms.items()}
    augmented["expanded_terms"] = _clean_terms([
        *augmented.get("expanded_terms", []),
        *_deterministic_query_expansions(question),
    ])
    return augmented


def _normalise_terms(raw: Any) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    return {
        "question_type": str(raw.get("question_type") or "other").strip().lower() or "other",
        "entities": _clean_terms(raw.get("entities", [])),
        "phrases": _clean_terms(raw.get("phrases", [])),
        "keywords": _clean_terms(raw.get("keywords", [])),
        "time_terms": _clean_terms(raw.get("time_terms", [])),
        "expanded_terms": _clean_terms(raw.get("expanded_terms", [])),
        "avoid_terms": _clean_terms(raw.get("avoid_terms", [])),
    }


def _clean_terms(value: Any) -> list[str]:
    values = [value] if isinstance(value, str) else list(value or []) if isinstance(value, (list, tuple, set)) else []
    out, seen = [], set()
    for item in values:
        text = re.sub(r"\s+", " ", str(item or "").strip())
        key = _normalise(text)
        if key and key not in seen:
            out.append(text)
            seen.add(key)
    return out


def _score_fact_for_terms(fact: dict[str, Any], terms: dict[str, Any]) -> tuple[float, list[str]]:
    text = " ".join([
        str(fact.get("fact_text") or ""), str(fact.get("subject") or ""), str(fact.get("speaker") or ""),
        " ".join(str(value) for value in fact.get("topics", []) if value), str(fact.get("time_text") or ""),
        str(fact.get("normalized_time") or ""), str(fact.get("semantic_key") or ""),
    ])
    normalized_text, token_set = _normalise(text), _tokens(text)
    matched: list[str] = []
    score = 0.0
    score += _score_group(terms["phrases"], normalized_text, token_set, 4.0, matched)
    score += _score_group(terms["entities"], normalized_text, token_set, 3.0, matched)
    score += _score_group(terms["keywords"], normalized_text, token_set, 1.5, matched)
    time_weight = 3.0 if terms["question_type"] in {"when", "how"} else 2.0
    score += _score_group(terms["time_terms"], normalized_text, token_set, time_weight, matched)
    score += _score_group(terms["expanded_terms"], normalized_text, token_set, 0.75, matched)
    avoid = _score_group(terms["avoid_terms"], normalized_text, token_set, 1.0, [])
    if avoid:
        score -= avoid
    if matched:
        score += {"high": 0.4, "medium": 0.2}.get(str(fact.get("importance", "")).lower(), 0)
    return score, matched


def _score_group(terms: list[str], text: str, token_set: set[str], weight: float, matched: list[str]) -> float:
    score = 0.0
    for term in terms:
        normalized = _normalise(term)
        if not normalized:
            continue
        pieces = normalized.split()
        hit = normalized in text if len(pieces) > 1 else any(variant in token_set for variant in _token_variants(pieces[0]))
        if hit:
            score += weight
            if term not in matched:
                matched.append(term)
    return score


def _token_variants(token: str) -> set[str]:
    variants = {token}
    if len(token) > 4 and token.endswith("ies"):
        variants.add(token[:-3] + "y")
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        variants.add(token[:-1])
    return variants


def _normalise(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _tokens(value: Any) -> set[str]:
    return set(_normalise(value).split())


def _dedupe_ids(values: Any) -> list[str]:
    out, seen = [], set()
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out
