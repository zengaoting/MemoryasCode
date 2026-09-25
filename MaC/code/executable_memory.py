"""Executable-memory SDK exposed to validated LLM search programs."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .memory_runtime import Evidence, evidence_from_function, merge_evidence


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def _normalise(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _tokens(value: Any) -> set[str]:
    return set(_normalise(value).split())


def _canonical_fact_id(value: Any) -> str:
    """Accept either canonical IDs (``F12``) or index IDs (``fact.F12``)."""

    text = str(value or "").strip()
    return text.removeprefix("fact.") if text.startswith("fact.") else text


def _program_selected(item: Evidence) -> Evidence:
    selected = list(item.metadata.get("program_selected_fact_ids", []))
    for fact_id in item.fact_ids:
        if fact_id and fact_id not in selected:
            selected.append(fact_id)
    return replace(item, metadata={**item.metadata, "program_selected_fact_ids": selected})


def _as_date(value: Any) -> tuple[int, int, int, str]:
    """Order ISO dates and common LoCoMo natural-language dates safely."""

    text = str(value or "").strip()
    iso = re.search(r"(\d{4})-(\d{1,2})(?:-(\d{1,2}))?", text)
    if iso:
        return (int(iso.group(1)), int(iso.group(2)), int(iso.group(3) or 0), text)
    months = {
        name: index for index, name in enumerate(
            ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"),
            start=1,
        )
    }
    natural = re.search(r"(\d{1,2})\s+([a-zA-Z]+),?\s+(\d{4})", text)
    if natural and natural.group(2).lower() in months:
        return (int(natural.group(3)), months[natural.group(2).lower()], int(natural.group(1)), text)
    year = re.search(r"(\d{4})", text)
    return (int(year.group(1)), 0, 0, text) if year else (0, 0, 0, text)


def _flatten(values: Evidence | list[Evidence] | tuple[Evidence, ...]) -> list[Evidence]:
    return list(values) if isinstance(values, (list, tuple)) else [values]


def _dedupe_evidence(values: list[Evidence], *, limit: int | None = None) -> list[Evidence]:
    """Keep one graph candidate per canonical fact-ID tuple, in input order."""

    output: list[Evidence] = []
    seen: set[tuple[str, ...]] = set()
    for value in values:
        key = tuple(value.fact_ids)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
        if limit is not None and len(output) >= limit:
            break
    return output


def _annotate_evidence(item: Evidence, **metadata: Any) -> Evidence:
    """Add runtime-only provenance without changing fact/source evidence."""

    return replace(item, metadata={**item.metadata, **metadata})


def _is_base_fact(record: dict[str, Any]) -> bool:
    return str(record.get("id", "")).startswith("fact.")


def _base_fact_id(record: dict[str, Any]) -> str:
    """Derive the canonical fact ID from the compact base-function ID."""

    function_id = str(record.get("id", ""))
    return function_id.removeprefix("fact.") if function_id.startswith("fact.") else ""


def _capabilities(record: dict[str, Any]) -> list[str]:
    """Derive runtime capability labels instead of persisting them in the index."""

    if _is_base_fact(record):
        return ["fact_lookup", f"predicate:{record.get('predicate', '')}"]
    operation = str(record.get("operation", ""))
    if operation == "current":
        return ["current_state", "temporal_reasoning", "state_update", "explicit_supersedes"]
    if operation == "history":
        return ["complete_retrieval", "temporal_reasoning", "explicit_supersedes"]
    return []


class MemorySDK:
    """Safe runtime facade for code-native facts and prepared retrieval bundles."""

    def __init__(
        self,
        sample_memory_dir: str | Path,
        source_dialogue: dict[str, Any] | None = None,
        retrieval_bundle: dict[str, Any] | None = None,
    ):
        self.sample_dir = Path(sample_memory_dir).resolve()
        self.code_dir = self.sample_dir / "memory_code"
        self._facts_by_id = _load_json(self.sample_dir / "facts_by_id.json")
        function_payload = _load_json(self.sample_dir / "function_index.json")
        if function_payload.get("schema") != "executable_function_index_v2":
            raise ValueError(f"Unsupported function index in {self.sample_dir}")
        self.function_index = [dict(item) for item in function_payload.get("functions", []) if isinstance(item, dict)]
        self._functions_by_id = {str(item.get("id", "")): item for item in self.function_index}
        self._base_by_fact_id = {
            _base_fact_id(item): item
            for item in self.function_index
            if _is_base_fact(item)
        }
        view_payload = _load_json(self.sample_dir / "view_index.json")
        if view_payload.get("schema") != "executable_view_index_v1":
            raise ValueError(f"Unsupported view index in {self.sample_dir}")
        self.view_index = []
        self._view_members_by_entrypoint: dict[str, tuple[str, ...]] = {}
        self._view_entrypoints_by_member: dict[str, list[str]] = {}
        for raw_view in view_payload.get("views", []):
            if not isinstance(raw_view, dict):
                continue
            entrypoint = str(raw_view.get("entrypoint", "")).strip()
            members = tuple(
                dict.fromkeys(
                    str(member)
                    for member in raw_view.get("members", [])
                    if str(member) in self._functions_by_id
                )
            )
            if not entrypoint or not members:
                continue
            view = {"entrypoint": entrypoint, "members": list(members)}
            self.view_index.append(view)
            self._view_members_by_entrypoint[entrypoint] = members
            for function_id in members:
                self._view_entrypoints_by_member.setdefault(function_id, []).append(entrypoint)
        graph = _load_json(self.sample_dir / "fact_graph.json")
        if graph.get("schema") != "fact_graph_v1":
            raise ValueError(f"Unsupported fact graph in {self.sample_dir}")
        self.fact_graph_edges = [dict(edge) for edge in graph.get("edges", []) if isinstance(edge, dict)]
        self.source_dialogue = source_dialogue or {}
        self.retrieval_bundle = retrieval_bundle or {}
        self._module_cache: dict[str, Any] = {}
        self._package_name = "mac_runtime_" + hashlib.sha256(str(self.code_dir).encode("utf-8")).hexdigest()[:16]

    # ----- Query operators exposed to generated programs -----

    def resolve(self) -> list[Evidence]:
        """Execute pre-ranked code functions, then expose their evidence."""

        payload = self.retrieval_bundle.get("resolve", self.retrieval_bundle.get("code_recall", {}))
        function_ids = payload.get("function_ids", []) if isinstance(payload, dict) else []
        output = []
        for function_id in function_ids:
            try:
                function_id = str(function_id)
                record = self._functions_by_id[function_id]
                output.append(
                    _annotate_evidence(
                        self._call_record(record, derived=not _is_base_fact(record)),
                        resolved_function_id=function_id,
                    )
                )
            except (ImportError, AttributeError, KeyError, TypeError, ValueError):
                continue
        return output

    def expand_scope(self, evidence: Evidence | list[Evidence], scopes: list[str] | None = None) -> list[Evidence]:
        """Execute other functions from concept/session modules containing code hits.

        ``resolve`` annotates its Evidence with the actual selected
        function ID.  The compact view index then maps that code hit to all
        generated person/topic/session modules which contain it.  The parent
        process has already relevance-ranked the expanded functions; this
        runtime call verifies the same module membership and materialises the
        selected callable code for the per-line execution trace.
        """

        seed_ids: list[str] = []
        for item in _flatten(evidence):
            raw = item.metadata.get("resolved_function_id")
            values = raw if isinstance(raw, (list, tuple)) else [raw]
            for value in values:
                function_id = str(value or "")
                if function_id and function_id not in seed_ids:
                    seed_ids.append(function_id)

        allowed_scopes = {str(scope).strip().lower() for scope in (scopes or ["concept", "session"])}
        matched_views: list[str] = []
        eligible_ids: list[str] = []
        seed_set = set(seed_ids)
        for seed_id in seed_ids:
            for entrypoint in self._view_entrypoints_by_member.get(seed_id, []):
                scope = entrypoint.split(".", 1)[0].lower()
                if scope not in allowed_scopes:
                    continue
                if entrypoint not in matched_views:
                    matched_views.append(entrypoint)
                for function_id in self._view_members_by_entrypoint[entrypoint]:
                    if function_id not in seed_set and function_id not in eligible_ids:
                        eligible_ids.append(function_id)

        payload = self.retrieval_bundle.get("expand_scope", self.retrieval_bundle.get("view_expand", {}))
        bundle_ids = payload.get("function_ids", []) if isinstance(payload, dict) else []
        if bundle_ids:
            selected_ids = [str(item) for item in bundle_ids if str(item) in set(eligible_ids)]
        else:
            # Useful for direct SDK inspection/tests; normal program execution
            # always receives the parent-ranked bundle.
            selected_ids = eligible_ids[:32]

        output = []
        for function_id in selected_ids:
            try:
                output.append(
                    _annotate_evidence(
                        self._call_record(
                            self._functions_by_id[function_id],
                            derived=not _is_base_fact(self._functions_by_id[function_id]),
                        ),
                        scope_function_id=function_id,
                        scope_entrypoints=[
                            entrypoint
                            for entrypoint in self._view_entrypoints_by_member.get(function_id, [])
                            if entrypoint in matched_views
                        ],
                    )
                )
            except (ImportError, AttributeError, KeyError, TypeError, ValueError):
                continue
        return output

    def match(self) -> list[Evidence]:
        payload = self.retrieval_bundle.get("match", self.retrieval_bundle.get("keyword_recall", {}))
        fact_ids = payload.get("fact_ids", []) if isinstance(payload, dict) else []
        return self._call_facts([str(fact_id) for fact_id in fact_ids])

    def union(self, left: Evidence | list[Evidence], right: Evidence | list[Evidence]) -> Evidence:
        return merge_evidence(_flatten(left) + _flatten(right), rule="union_evidence")

    def rank(self, evidence: Evidence | list[Evidence], limit: int = 36) -> list[Evidence]:
        values = _dedupe_evidence(_flatten(evidence))
        question = str(self.retrieval_bundle.get("question", ""))
        scores = []
        for index, item in enumerate(values):
            text = " ".join(str(self._facts_by_id.get(fid, {}).get("fact_text", "")) for fid in item.fact_ids)
            scores.append((len(_tokens(question) & _tokens(text)), index, item))
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [item for _, _, item in scores[: max(0, int(limit))]]

    def hydrate(self, evidence: Evidence | list[Evidence]) -> Evidence:
        values = _flatten(evidence)
        fact_ids = list(dict.fromkeys(
            fact_id for item in values for fact_id in item.fact_ids if fact_id in self._facts_by_id
        ))
        context = {
            "fact_ids": fact_ids,
            "facts": [self._facts_by_id[fact_id] for fact_id in fact_ids],
            "source_lines": self._source_lines(fact_ids),
        }
        return merge_evidence(values, value=context, rule="hydrate_evidence")

    # ----- Generic programmatic-search APIs -----

    def _call_facts(self, fact_ids: list[str] | tuple[str, ...]) -> list[Evidence]:
        output = []
        for fact_id in fact_ids:
            record = self._base_by_fact_id.get(_canonical_fact_id(fact_id))
            if record:
                output.append(self._call_record(record, derived=False))
        return output

    def execute(self, function_id: str, **kwargs: Any) -> Evidence:
        record = self._functions_by_id.get(str(function_id))
        if not record:
            raise KeyError(f"Unknown memory function: {function_id}")
        return _program_selected(
            self._call_record(record, derived=not _is_base_fact(record), kwargs=kwargs)
        )

    def search_functions(
        self,
        query: str = "",
        subject: str | None = None,
        predicate: str | None = None,
        capability: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        records = []
        for record in self.function_index:
            if subject and _normalise(record.get("subject")) != _normalise(subject):
                continue
            if predicate and _normalise(record.get("predicate")) != _normalise(predicate):
                continue
            capabilities = {_normalise(value) for value in _capabilities(record)}
            if capability and _normalise(capability) not in capabilities:
                continue
            records.append(record)
        return self._rank_records(query, records, limit) if query else records[: max(0, int(limit))]

    def search(self, query: str, limit: int = 24) -> list[Evidence]:
        records = self._rank_records(query, list(self._facts_by_id.values()), limit)
        return self._call_facts([str(item.get("fact_id", "")) for item in records])

    def select(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        topics: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        complete: bool = False,
        limit: int = 24,
    ) -> list[Evidence]:
        wanted_topics = {_normalise(value) for value in (topics or []) if _normalise(value)}
        candidates = []
        predicate_matches = []
        for fact in self._facts_by_id.values():
            if subject and _normalise(fact.get("subject")) != _normalise(subject):
                continue
            base = self._base_by_fact_id.get(str(fact.get("fact_id", "")), {})
            predicate_exact = not predicate or _normalise(base.get("predicate")) == _normalise(predicate)
            fact_topics = {_normalise(value) for value in fact.get("topics", [])}
            if wanted_topics and not wanted_topics.intersection(fact_topics):
                continue
            current = _as_date(fact.get("normalized_time"))
            if start and current < _as_date(start):
                continue
            if end and current > _as_date(end):
                continue
            if predicate_exact:
                candidates.append(fact)
            elif predicate:
                query_tokens = _tokens(predicate) - {"count", "all", "list", "history"}
                searchable = " ".join((str(base.get("predicate", "")), str(fact.get("fact_text", ""))))
                matched_tokens = query_tokens & _tokens(searchable)
                overlap = len(matched_tokens)
                if overlap:
                    predicate_matches.append((overlap, fact, matched_tokens))
        if predicate and not candidates:
            token_frequency = {
                token: sum(token in matched for _, _, matched in predicate_matches)
                for token in (_tokens(predicate) - {"count", "all", "list", "history"})
            }
            weighted = [
                (sum(1.0 / token_frequency[token] for token in matched), fact)
                for _, fact, matched in predicate_matches
            ]
            best = max((score for score, _ in weighted), default=0.0)
            weighted = [item for item in weighted if item[0] >= best * 0.5]
            weighted.sort(key=lambda item: (-item[0], _as_date(item[1].get("normalized_time"))))
            candidates = [fact for _, fact in weighted]
        candidates.sort(key=lambda item: (_as_date(item.get("normalized_time")), str(item.get("fact_id", ""))))
        if not complete:
            candidates = candidates[: max(0, int(limit))]
        return self._call_facts([str(item.get("fact_id", "")) for item in candidates])

    def _source_lines(self, fact_ids: list[str], window: int = 2) -> list[str]:
        wanted: list[tuple[int, int]] = []
        for fact_id in fact_ids:
            fact = self._facts_by_id.get(fact_id, {})
            for dialogue_id in fact.get("dialogue_ids") or [fact.get("dia_id", "")]:
                match = re.fullmatch(r"D(\d+):(\d+)", str(dialogue_id))
                if match:
                    wanted.append((int(match.group(1)), int(match.group(2))))
        index: dict[tuple[int, int], tuple[dict[str, Any], str]] = {}
        conversation = self.source_dialogue.get("conversation", {})
        for session_name, turns in conversation.items():
            match = re.fullmatch(r"session_(\d+)", str(session_name))
            if not match or not isinstance(turns, list):
                continue
            session = int(match.group(1))
            session_time = str(conversation.get(f"session_{session}_date_time") or "")
            for turn in turns:
                dialogue = re.fullmatch(r"D(\d+):(\d+)", str(turn.get("dia_id", ""))) if isinstance(turn, dict) else None
                if dialogue:
                    index[(int(dialogue.group(1)), int(dialogue.group(2)))] = (turn, session_time)
        lines, seen = [], set()
        for session, turn in wanted:
            for neighbor in range(max(1, turn - window), turn + window + 1):
                item = index.get((session, neighbor))
                if not item:
                    continue
                source, session_time = item
                dialogue_id = str(source.get("dia_id", ""))
                if dialogue_id in seen:
                    continue
                seen.add(dialogue_id)
                details = re.sub(r"\s+", " ", str(source.get("text") or "")).strip() or "(non-text message)"
                visual = "; ".join(
                    value for value in (
                        str(source.get("blip_caption") or "").strip(),
                        str(source.get("query") or "").strip(),
                    ) if value
                )
                visual_note = f" | visual={visual}" if visual else ""
                lines.append(
                    f"- [{dialogue_id} | program_neighbor={neighbor - turn:+d} | "
                    f"session_datetime={session_time}{visual_note} | {source.get('speaker') or 'unknown'}] {details}"
                )
        return lines

    def where(self, evidence: Evidence | list[Evidence], start: str | None = None, end: str | None = None) -> list[Evidence]:
        output = []
        for item in _flatten(evidence):
            current = _as_date(item.metadata.get("normalized_time"))
            if start and current < _as_date(start):
                continue
            if end and current > _as_date(end):
                continue
            output.append(item)
        return output

    def expand_relation(self, evidence: Evidence | list[Evidence], relations: list[str] | None = None) -> list[Evidence]:
        seeds = {fact_id for item in _flatten(evidence) for fact_id in item.fact_ids}
        allowed = set(relations or [])
        structural = {"temporal_before", "same_subject", "shared_topic"}
        targets = []
        for edge in self.fact_graph_edges:
            relation = str(edge.get("relation", ""))
            if relation not in structural or (allowed and relation not in allowed):
                continue
            source, target = str(edge.get("source", "")), str(edge.get("target", ""))
            neighbor = target if source in seeds else source if target in seeds else ""
            if neighbor and neighbor not in seeds and neighbor not in targets:
                targets.append(neighbor)
        ranked = self._rank_records(str(self.retrieval_bundle.get("question", "")), [self._facts_by_id[item] for item in targets if item in self._facts_by_id], 96)
        return self._call_facts([str(item.get("fact_id", "")) for item in ranked])

    def aggregate(self, evidence: Evidence | list[Evidence], operation: str = "count") -> Evidence:
        values = _flatten(evidence)
        if operation != "count":
            raise ValueError(f"Unsupported aggregate operation: {operation}")
        return merge_evidence(values, value=len(values), rule="aggregate_count")

    def validate(self, question: str, evidence: Evidence | list[Evidence]) -> dict[str, Any]:
        values = _flatten(evidence)
        text = " ".join(str(item.value) + " " + str(item.metadata.get("fact_text", "")) for item in values)
        coverage = len(_tokens(question) & _tokens(text))
        unresolved = any("ambiguous_active_state" in item.rules or "no_active_state" in item.rules for item in values)
        temporal = bool(re.search(r"\b(when|date|time|year|month|day|last|next|current|now)\b", question, re.I))
        multi_hop = bool(re.search(r"\b(why|because|and|then|after|before)\b", question, re.I))
        return {
            "sufficient": bool(values) and coverage > 0 and not unresolved,
            "needs_source": temporal or multi_hop or coverage == 0 or len(values) > 1 or unresolved,
            "coverage": coverage,
            "fact_ids": [fact_id for item in values for fact_id in item.fact_ids],
            "unresolved_state": unresolved,
        }

    # ----- Code loading and indexing internals -----

    def _call_record(self, record: dict[str, Any], *, derived: bool, kwargs: dict[str, Any] | None = None) -> Evidence:
        module_name, separator, function_name = str(record.get("entrypoint", "")).partition(":")
        if not separator:
            raise ValueError(f"Invalid catalog entrypoint: {record.get('entrypoint')!r}")
        function = getattr(self._load_generated_module(module_name), function_name, None)
        if not callable(function):
            raise ValueError(f"Entrypoint does not resolve to callable: {record.get('entrypoint')}")
        if not derived and kwargs:
            raise ValueError("Base fact functions do not accept arguments")
        result = function(**(kwargs or {})) if derived else evidence_from_function(function)
        if not isinstance(result, Evidence):
            raise TypeError(f"Memory function {record.get('entrypoint')} did not return Evidence")
        return result

    def _load_generated_module(self, module_name: str):
        if module_name in self._module_cache:
            return self._module_cache[module_name]
        if not re.fullmatch(r"[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*", module_name):
            raise ValueError(f"Invalid generated module name: {module_name!r}")
        if self._package_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                self._package_name,
                self.code_dir / "__init__.py",
                submodule_search_locations=[str(self.code_dir)],
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load generated package: {self.code_dir}")
            package = importlib.util.module_from_spec(spec)
            sys.modules[self._package_name] = package
            spec.loader.exec_module(package)
        module = importlib.import_module(f"{self._package_name}.{module_name}")
        self._module_cache[module_name] = module
        return module

    def _rank_records(self, query: str, records: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        values = list(records)
        question_tokens = _tokens(query)
        scored = []
        for index, record in enumerate(values):
            text = " ".join(str(record.get(key, "")) for key in ("fact_text", "search", "subject", "predicate"))
            overlap = len(question_tokens & _tokens(text))
            if overlap:
                scored.append((overlap, index, record))
        scored.sort(key=lambda item: (-item[0], item[1]))
        if not scored:
            return [dict(item) for item in values[: max(0, int(limit))]]
        return [dict(item) for _, _, item in scored[: max(0, int(limit))]]
