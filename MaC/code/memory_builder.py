from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path


from .utils import ensure_dir, write_json

logger = logging.getLogger(__name__)

SEMANTIC_DUPLICATE_THRESHOLD = 0.72
MAX_GLOBAL_SAME_SUBJECT_EDGES_PER_FACT = 4
MAX_GLOBAL_SHARED_TOPIC_EDGES_PER_FACT = 4
WEAK_GRAPH_TOPICS = {"", "general", "other", "unknown", "misc", "miscellaneous"}

# ``facts_by_id.json`` is the canonical hand-off between extraction and the
# executable-memory compiler.  Keep it deliberately small: the omitted
# fields (pre-dedupe IDs, semantic keys, duplicate counters, etc.) are useful
# while constructing the canonical set, but are not part of retrieval or
# answer-time evidence.


def build_canonical_fact_artifacts(
    facts: list[dict], sample_memory_dir: str | Path
) -> list[dict]:
    """Write the compact canonical fact store and its legacy-equivalent graph.

    Atomic facts are intentionally *not* an executable-memory input.  They
    first pass through the established normalisation and semantic-deduplication
    routine. After semantic deduplication, the frozen canonical set receives
    readable final IDs (`F1`, `F2`, ...) in a deterministic source order.
    """

    canonical_facts, _, _ = _prepare_fact_ids(facts)
    compact_facts = [_compact_canonical_fact(fact) for fact in canonical_facts]
    output_dir = Path(sample_memory_dir)
    ensure_dir(output_dir)
    write_json(
        {fact["fact_id"]: fact for fact in compact_facts},
        output_dir / "facts_by_id.json",
    )
    # This deliberately preserves fact_graph_v1 and its three structural edge
    # types.  The executable runtime reads this file directly.
    write_json(_build_global_fact_graph(compact_facts), output_dir / "fact_graph.json")
    return compact_facts


def _compact_canonical_fact(fact: dict) -> dict:
    """Project a construction-time fact onto the runtime canonical schema."""

    return {
        "fact_id": str(fact.get("fact_id") or ""),
        "fact_text": str(fact.get("fact_text") or ""),
        "speaker": str(fact.get("speaker") or ""),
        "subject": str(fact.get("subject") or "unknown"),
        "fact_type": str(fact.get("fact_type") or "other"),
        "topics": [str(topic) for topic in fact.get("topics", []) if str(topic).strip()],
        "normalized_time": str(fact.get("normalized_time") or ""),
        "dia_id": str(fact.get("dia_id") or ""),
        "session_id": str(fact.get("session_id") or "unknown"),
        "dialogue_ids": [
            str(dialogue_id)
            for dialogue_id in fact.get("dialogue_ids", [])
            if str(dialogue_id).strip()
        ],
        "sources": _normalise_sources(fact),
    }


def _normalise_sources(fact: dict) -> list[dict[str, str]]:
    """Return ordered, exact ``(session, dialogue)`` source pairs.

    ``sources`` is canonical.  The legacy scalar session/dialogue fields and
    dialogue-id list are accepted only so older intermediate artifacts can be
    rebuilt into the new representation.
    """

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
        dialogue_ids = fact.get("dialogue_ids") or [fact.get("dia_id", "")]
        for dialogue in dialogue_ids:
            text = str(dialogue).strip()
            if text:
                pairs.add((session, text))
    return [
        {"session_id": session, "dialogue_id": dialogue}
        for session, dialogue in sorted(pairs, key=lambda item: (_session_sort_key(item[0]), _dialogue_sort_key(item[1])))
    ]


def _normalize_fact(fact: dict, idx: int) -> dict:
    item = dict(fact)
    source_fact_id = str(item.get("source_fact_id") or item.get("fact_id") or "").strip()
    item["source_fact_id"] = source_fact_id
    item["fact_id"] = str(item.get("fact_id") or item.get("dia_id") or f"fact_{idx:04d}")
    item["speaker"] = str(item.get("speaker") or "").strip()
    item["subject"] = str(item.get("subject") or "unknown").strip() or "unknown"
    item["fact_text"] = str(item.get("fact_text") or "").strip()
    item["topics"] = [str(t).strip() for t in item.get("topics", []) if str(t).strip()]
    if not item["topics"]:
        item["topics"] = ["general"]
    item.setdefault("fact_type", "other")
    item.setdefault("session_id", "unknown")
    item.setdefault("session_datetime", "")
    item.setdefault("dia_id", item["fact_id"])
    item["importance"] = _normalize_or_infer_importance(item)
    item["semantic_key"] = _semantic_key(item)
    return item


def _prepare_fact_ids(facts: list[dict]) -> tuple[list[dict], dict, list[dict]]:
    """Normalize fact IDs and merge semantic duplicates.

    Returns (deduplicated_canonical_facts, aliases, duplicate_groups).
    """
    # Step 1: normalize all facts
    normalized: list[dict] = []
    for idx, fact in enumerate(facts):
        item = _normalize_fact(fact, idx)
        item["pre_dedupe_fact_id"] = f"raw_{idx + 1}"
        item["fact_id"] = item["pre_dedupe_fact_id"]
        item["source_fact_ids"] = [item["source_fact_id"]] if item.get("source_fact_id") else []
        item["pre_dedupe_fact_ids"] = [item["pre_dedupe_fact_id"]]
        item["dialogue_ids"] = [item.get("dia_id", item["fact_id"])]
        item["session_ids"] = [item.get("session_id", "unknown")]
        item["sources"] = _normalise_sources(item)
        item["duplicate_of"] = ""
        item["semantic_duplicate_count"] = 1
        normalized.append(item)

    # Deduplication picks a deterministic representative even if the extractor
    # happens to return its facts in a different order on a later rebuild.
    normalized.sort(key=_canonical_input_sort_key)

    # Step 2: detect and merge semantic duplicates
    canonical_facts: list[dict] = []
    duplicate_groups: list[dict] = []
    for item in normalized:
        dup_idx = _find_semantic_duplicate(item, canonical_facts)
        if dup_idx is not None:
            representative = canonical_facts[dup_idx]
            _merge_duplicate_fact(representative, item)
            item["duplicate_of"] = representative["fact_id"]
            duplicate_groups.append({
                "duplicate_fact_id": item["pre_dedupe_fact_id"],
                "canonical_fact_id": representative["fact_id"],
                "source_fact_ids": item.get("source_fact_ids", []),
            })
        else:
            canonical_facts.append(item)

    # Step 3: assign the final, human-readable IDs only after the complete
    # canonical set has been deduplicated.  A completed memory version is
    # immutable during QA, so `F1`, `F2`, ... are the permanent IDs for that
    # version.  The ordering is deterministic for a given frozen fact set.
    canonical_facts.sort(key=_canonical_input_sort_key)
    aliases = {
        "schema": "fact_id_aliases_v2",
        "description": "Maps pre-dedupe raw ids to final canonical fact ids.",
        "rewrote_fact_ids": True,
        "by_source_fact_id": {},
        "by_pre_dedupe_fact_id": {},
        "duplicate_count": len(duplicate_groups),
        "total_normalized": len(normalized),
        "total_canonical": len(canonical_facts),
    }

    for ordinal, fact in enumerate(canonical_facts, start=1):
        canonical_id = f"F{ordinal}"
        for old_id in fact["pre_dedupe_fact_ids"]:
            aliases["by_pre_dedupe_fact_id"][old_id] = canonical_id
        for source_id in fact.get("source_fact_ids", []):
            if source_id:
                aliases["by_source_fact_id"].setdefault(source_id, []).append(canonical_id)
        fact["fact_id"] = canonical_id

    for key, values in list(aliases["by_source_fact_id"].items()):
        aliases["by_source_fact_id"][key] = sorted(set(values))
    for group in duplicate_groups:
        original = str(group.get("canonical_fact_id", ""))
        group["canonical_fact_id"] = aliases["by_pre_dedupe_fact_id"].get(original, original)

    logger.info(
        "  Fact dedup: %d normalized -> %d canonical (merged %d semantic duplicates)",
        len(normalized), len(canonical_facts), len(duplicate_groups),
    )

    return canonical_facts, aliases, duplicate_groups


def _canonical_input_sort_key(fact: dict) -> tuple:
    source = _normalise_sources(fact)
    primary = source[0] if source else {"session_id": "unknown", "dialogue_id": ""}
    return (
        _session_sort_key(primary["session_id"]),
        _dialogue_sort_key(primary["dialogue_id"]),
        _semantic_key(fact),
        _normalize_text(fact.get("fact_text", "")),
    )


def _session_sort_key(value: object) -> tuple[int, int, str]:
    """Order ``session_2`` before ``session_10`` for final F-numbering."""

    text = str(value or "")
    match = re.fullmatch(r"session_(\d+)", text)
    return (0, int(match.group(1)), text) if match else (1, 10**9, text)


def _semantic_key(fact: dict) -> str:
    parts = [
        fact.get("speaker", ""),
        fact.get("subject", ""),
        fact.get("fact_type", ""),
        fact.get("normalized_time", ""),
        fact.get("fact_text", ""),
    ]
    return _normalize_text(" ".join(str(p) for p in parts))


def _semantic_tokens(fact: dict) -> set[str]:
    text = " ".join([
        str(fact.get("speaker", "")),
        str(fact.get("subject", "")),
        str(fact.get("fact_type", "")),
        str(fact.get("normalized_time", "")),
        str(fact.get("fact_text", "")),
        " ".join(str(t) for t in fact.get("topics", [])),
    ])
    stopwords = {
        "a", "an", "the", "and", "or", "to", "of", "in", "on", "at", "for",
        "with", "her", "his", "their", "she", "he", "they", "is", "was",
        "are", "were", "be", "been", "being", "has", "have", "had",
    }
    return {tok for tok in _normalize_text(text).split() if tok not in stopwords}


def _find_semantic_duplicate(fact: dict, existing: list[dict]) -> int | None:
    fact_key = fact.get("semantic_key", "")
    fact_tokens = _semantic_tokens(fact)
    for idx, candidate in enumerate(existing):
        if fact_key and fact_key == candidate.get("semantic_key", ""):
            return idx
        if not _same_duplicate_scope(fact, candidate):
            continue
        candidate_tokens = _semantic_tokens(candidate)
        union = fact_tokens | candidate_tokens
        if not union:
            continue
        jaccard = len(fact_tokens & candidate_tokens) / len(union)
        if jaccard >= SEMANTIC_DUPLICATE_THRESHOLD:
            return idx
    return None


def _same_duplicate_scope(left: dict, right: dict) -> bool:
    if _normalize_text(left.get("speaker", "")) != _normalize_text(right.get("speaker", "")):
        return False
    if _normalize_text(left.get("subject", "")) != _normalize_text(right.get("subject", "")):
        return False
    if str(left.get("fact_type", "")) != str(right.get("fact_type", "")):
        return False
    left_time = str(left.get("normalized_time", "")).strip().lower()
    right_time = str(right.get("normalized_time", "")).strip().lower()
    if left_time and right_time and left_time != right_time:
        return False
    return bool(set(left.get("topics", [])) & set(right.get("topics", [])))


def _merge_duplicate_fact(representative: dict, duplicate: dict) -> None:
    representative["semantic_duplicate_count"] = representative.get("semantic_duplicate_count", 1) + 1
    representative["pre_dedupe_fact_ids"] = sorted(set(
        representative.get("pre_dedupe_fact_ids", []) + [duplicate["pre_dedupe_fact_id"]]
    ))
    if duplicate.get("source_fact_id"):
        representative["source_fact_ids"] = sorted(set(
            representative.get("source_fact_ids", []) + [duplicate["source_fact_id"]]
        ))
    representative["dialogue_ids"] = sorted(set(
        representative.get("dialogue_ids", []) + [duplicate.get("dia_id", duplicate["fact_id"])]
    ), key=_dialogue_sort_key)
    representative["session_ids"] = sorted(set(
        representative.get("session_ids", []) + [duplicate.get("session_id", "unknown")]
    ))
    representative["sources"] = _normalise_sources({
        "sources": [
            *representative.get("sources", []),
            *duplicate.get("sources", []),
        ],
    })
    representative["topics"] = sorted(set(representative.get("topics", []) + duplicate.get("topics", [])))
    representative["merged_fact_texts"] = sorted(set(
        representative.get("merged_fact_texts", []) + [duplicate.get("fact_text", "")]
    ))
    representative["importance"] = _max_importance(representative.get("importance"), duplicate.get("importance"))


def _max_importance(left: str, right: str) -> str:
    rank = {"low": 0, "medium": 1, "high": 2}
    return left if rank.get(left, 1) >= rank.get(right, 1) else right


def _normalize_or_infer_importance(fact: dict) -> str:
    fact_type = str(fact.get("fact_type", "other")).strip().lower()
    topics = {str(t).strip().lower() for t in fact.get("topics", [])}
    time_value = str(fact.get("normalized_time") or fact.get("time_text") or "").strip().lower()

    if fact_type in {"profile", "relationship", "plan", "possession", "place", "temporal"}:
        return "high"
    if fact_type in {"event", "motivation"}:
        return "high" if time_value and time_value not in {"not specified", "unspecified"} else "medium"
    if topics & {"career", "family", "children", "adoption", "counseling", "mental health", "lgbtq"}:
        return "high"
    if fact_type in {"preference", "emotion", "activity"}:
        return "medium"
    return "low"


def _normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _build_global_fact_graph(facts: list[dict]) -> dict:
    sorted_facts = _sort_facts(_dedupe_facts_by_id(facts))
    edges: list[dict] = []
    edge_keys: set[tuple[str, str, str]] = set()

    def add_edge(source: str, target: str, relation: str) -> None:
        if not source or not target or source == target:
            return
        key = (source, target, relation)
        if key in edge_keys:
            return
        edge_keys.add(key)
        edges.append({"source": source, "target": target, "relation": relation})

    facts_by_session: dict[str, list[tuple[tuple, dict]]] = defaultdict(list)
    for fact in sorted_facts:
        for source in _normalise_sources(fact):
            facts_by_session[source["session_id"]].append(
                (_dialogue_sort_key(source["dialogue_id"]), fact)
            )
    for session_items in facts_by_session.values():
        session_facts = [
            fact for _, fact in sorted(session_items, key=lambda item: (item[0], item[1]["fact_id"]))
        ]
        for prev, cur in zip(session_facts, session_facts[1:]):
            add_edge(prev["fact_id"], cur["fact_id"], "temporal_before")

    fact_by_id = {fact["fact_id"]: fact for fact in sorted_facts}
    for fact in sorted_facts:
        for neighbor in _global_same_subject_neighbors(fact, sorted_facts):
            add_edge(fact["fact_id"], neighbor["fact_id"], "same_subject")
        for neighbor in _global_shared_topic_neighbors(fact, sorted_facts, fact_by_id):
            add_edge(fact["fact_id"], neighbor["fact_id"], "shared_topic")

    return {
        "schema": "fact_graph_v1",
        "nodes": [fact["fact_id"] for fact in sorted_facts],
        "edges": edges,
    }


def _global_same_subject_neighbors(fact: dict, facts: list[dict]) -> list[dict]:
    subject = _normalize_text(fact.get("subject", ""))
    if not subject:
        return []
    candidates = [
        candidate for candidate in facts
        if candidate["fact_id"] != fact["fact_id"]
        and _normalize_text(candidate.get("subject", "")) == subject
    ]
    candidates.sort(key=lambda candidate: _neighbor_sort_key(fact, candidate))
    return candidates[:MAX_GLOBAL_SAME_SUBJECT_EDGES_PER_FACT]


def _global_shared_topic_neighbors(fact: dict, facts: list[dict], fact_by_id: dict[str, dict]) -> list[dict]:
    del fact_by_id  # Kept for symmetry with future graph builders.
    fact_topics = _graph_topics(fact)
    if not fact_topics:
        return []
    candidates = [
        candidate for candidate in facts
        if candidate["fact_id"] != fact["fact_id"]
        and fact_topics & _graph_topics(candidate)
    ]
    candidates.sort(key=lambda candidate: _neighbor_sort_key(fact, candidate))
    return candidates[:MAX_GLOBAL_SHARED_TOPIC_EDGES_PER_FACT]


def _neighbor_sort_key(left: dict, right: dict) -> tuple:
    same_session = left.get("session_id") == right.get("session_id")
    shared_topics = len(_graph_topics(left) & _graph_topics(right))
    left_key = _dialogue_sort_key(left.get("dia_id", ""))
    right_key = _dialogue_sort_key(right.get("dia_id", ""))
    dialogue_gap = _dialogue_distance(left_key, right_key)
    return (
        0 if same_session else 1,
        -shared_topics,
        dialogue_gap,
        str(right.get("session_id", "")),
        right_key,
        right.get("fact_id", ""),
    )


def _dialogue_distance(left_key: tuple, right_key: tuple) -> int:
    max_len = max(len(left_key), len(right_key))
    total = 0
    for idx in range(max_len):
        left_value = left_key[idx] if idx < len(left_key) else 0
        right_value = right_key[idx] if idx < len(right_key) else 0
        total += abs(left_value - right_value)
    return total


def _graph_topics(fact: dict) -> set[str]:
    return {
        _normalize_text(topic)
        for topic in fact.get("topics", [])
        if _normalize_text(topic) not in WEAK_GRAPH_TOPICS
    }


def _dedupe_facts_by_id(facts: list[dict]) -> list[dict]:
    """Remove duplicate facts with the same fact_id, keeping first occurrence."""
    seen: set[str] = set()
    result: list[dict] = []
    for f in facts:
        fid = f["fact_id"]
        if fid not in seen:
            seen.add(fid)
            result.append(f)
    return result


def _sort_facts(facts: list[dict]) -> list[dict]:
    return sorted(facts, key=lambda f: (_session_sort_key(f.get("session_id", "")), _dialogue_sort_key(f.get("dia_id", "")), f.get("fact_id", "")))


def _dialogue_sort_key(value: str) -> tuple:
    nums = [int(n) for n in re.findall(r"\d+", str(value))]
    return tuple(nums) if nums else (10**9,)
