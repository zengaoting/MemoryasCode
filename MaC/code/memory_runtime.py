"""Code-native runtime primitives for generated executable memory packages.

Generated ``facts/*`` modules define plain functions returning ``MemoryValue``.
Decorators attach provenance and dependency metadata to those exact function
objects; ``concept/*`` and ``sessions/*`` modules import the same
objects rather than copying fact text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class MemoryValue:
    subject: str
    time: str
    description: str
    value: Any = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = self.__class__.__name__
        return payload


@dataclass
class SourceRef:
    session: str
    dialog: str


@dataclass
class RelationRef:
    target: Callable[..., MemoryValue]
    relation: str


@dataclass
class FactMetadata:
    fact_id: str = ""
    predicate: str = "other"
    fact_type: str = "other"
    stateful: bool = False
    sources: list[SourceRef] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    supersedes: Callable[..., MemoryValue] | None = None
    relations: list[RelationRef] = field(default_factory=list)


def _metadata(function: Callable[..., MemoryValue]) -> FactMetadata:
    metadata = getattr(function, "__memory_metadata__", None)
    if not isinstance(metadata, FactMetadata):
        metadata = FactMetadata()
        setattr(function, "__memory_metadata__", metadata)
    return metadata


def fact(
    *,
    fact_id: str,
    predicate: str,
    fact_type: str,
    stateful: bool = False,
    supersedes: Callable[..., MemoryValue] | None = None,
):
    """Attach canonical identity and state dependency to a fact function."""

    def decorate(function: Callable[..., MemoryValue]) -> Callable[..., MemoryValue]:
        metadata = _metadata(function)
        metadata.fact_id = str(fact_id)
        metadata.predicate = str(predicate)
        metadata.fact_type = str(fact_type)
        metadata.stateful = bool(stateful)
        metadata.supersedes = supersedes
        return function

    return decorate


def source(*, session: str, dialog: str):
    """Attach one source turn. The decorator is intentionally repeatable."""

    def decorate(function: Callable[..., MemoryValue]) -> Callable[..., MemoryValue]:
        metadata = _metadata(function)
        item = SourceRef(session=str(session), dialog=str(dialog))
        if item not in metadata.sources:
            metadata.sources.append(item)
        return function

    return decorate


def topics(*values: str):
    """Attach topic labels without copying the underlying fact."""

    def decorate(function: Callable[..., MemoryValue]) -> Callable[..., MemoryValue]:
        metadata = _metadata(function)
        for value in values:
            topic = str(value).strip()
            if topic and topic not in metadata.topics:
                metadata.topics.append(topic)
        return function

    return decorate


def related_to(target: Callable[..., MemoryValue], *, relation: str):
    """Attach a structural relation using the actual target function object."""

    if relation not in {"temporal_before", "same_subject", "shared_topic"}:
        raise ValueError(f"Unsupported structural relation: {relation}")

    def decorate(function: Callable[..., MemoryValue]) -> Callable[..., MemoryValue]:
        metadata = _metadata(function)
        if not any(item.target is target and item.relation == relation for item in metadata.relations):
            metadata.relations.append(RelationRef(target=target, relation=relation))
        return function

    return decorate


def get_fact_metadata(function: Callable[..., MemoryValue]) -> FactMetadata:
    """Return metadata attached to a generated fact function."""

    return _metadata(function)


def _jsonable(value: Any) -> Any:
    if isinstance(value, MemoryValue):
        return value.to_dict()
    if isinstance(value, Evidence):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class Evidence:
    """A value together with canonical facts and source turns that support it."""

    value: Any
    fact_ids: tuple[str, ...] = ()
    dialogue_ids: tuple[str, ...] = ()
    rules: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": _jsonable(self.value),
            "fact_ids": list(self.fact_ids),
            "dialogue_ids": list(self.dialogue_ids),
            "rules": list(self.rules),
            "sources": list(self.sources),
            "metadata": _jsonable(self.metadata),
        }


def evidence_from_function(function: Callable[..., MemoryValue]) -> Evidence:
    """Call one fact function and materialise its decorator provenance."""

    value = function()
    if not isinstance(value, MemoryValue):
        raise TypeError(f"Fact function {function.__name__} must return MemoryValue")
    metadata = get_fact_metadata(function)
    dialogue_ids = tuple(item.dialog for item in metadata.sources if item.dialog)
    source_refs = [
        {"session_id": item.session, "dialogue_id": item.dialog}
        for item in metadata.sources
        if item.dialog
    ]
    return Evidence(
        value=value,
        fact_ids=(metadata.fact_id,) if metadata.fact_id else (),
        dialogue_ids=dialogue_ids,
        sources=dialogue_ids,
        metadata={
            "fact_text": value.description,
            "speaker": value.subject,
            "subject": value.subject,
            "predicate": metadata.predicate,
            "topics": list(metadata.topics),
            "normalized_time": value.time,
            "fact_type": metadata.fact_type,
            "source_refs": source_refs,
        },
    )


def latest(*functions: Callable[..., MemoryValue], as_of: str | None = None) -> Evidence:
    """Resolve one explicit active state, rejecting zero or multiple actives."""

    pairs = [(function, evidence_from_function(function)) for function in functions]
    if as_of:
        pairs = [
            (function, item)
            for function, item in pairs
            if str(item.metadata.get("normalized_time", "")) <= str(as_of)
        ]
    values = [item for _, item in pairs]
    if not values:
        return Evidence(
            value=None,
            rules=("no_active_state", "state_cardinality_0"),
            metadata={"active_fact_ids": [], "active_state_count": 0},
        )
    superseded_ids = {
        get_fact_metadata(get_fact_metadata(function).supersedes).fact_id
        for function, _ in pairs
        if get_fact_metadata(function).supersedes is not None
    }
    active = [item for item in values if not set(item.fact_ids).intersection(superseded_ids)]
    active_ids = [fact_id for item in active for fact_id in item.fact_ids]
    if len(active) != 1:
        rule = "no_active_state" if not active else "ambiguous_active_state"
        merged = merge_evidence(values, value=None, rule=rule)
        return Evidence(
            value=None,
            fact_ids=merged.fact_ids,
            dialogue_ids=merged.dialogue_ids,
            sources=merged.sources,
            rules=merged.rules + (f"state_cardinality_{len(active)}",),
            metadata={
                "active_fact_ids": active_ids,
                "active_state_count": len(active),
                "as_of": as_of,
            },
        )
    selected = active[0]
    merged = merge_evidence(values, value=selected.value, rule="resolve_explicit_supersedes")
    return Evidence(
        value=merged.value,
        fact_ids=merged.fact_ids,
        dialogue_ids=merged.dialogue_ids,
        sources=merged.sources,
        rules=merged.rules + ("state_cardinality_1",),
        metadata={
            **selected.metadata,
            "active_fact_ids": active_ids,
            "active_state_count": 1,
            "as_of": as_of,
        },
    )


def history(*functions: Callable[..., MemoryValue], as_of: str | None = None) -> Evidence:
    values = [evidence_from_function(function) for function in functions]
    if as_of:
        values = [item for item in values if str(item.metadata.get("normalized_time", "")) <= str(as_of)]
    return merge_evidence(values, rule="explicit_state_history")


def merge_evidence(
    values: Iterable[Evidence],
    *,
    value: Any = None,
    rule: str | None = None,
) -> Evidence:
    """Combine evidence without losing order-stable provenance."""

    fact_ids: list[str] = []
    dialogue_ids: list[str] = []
    sources: list[str] = []
    rules: list[str] = []
    materialised = list(values)
    program_selected_fact_ids: list[str] = []
    for item in materialised:
        for collection, target in (
            (item.fact_ids, fact_ids),
            (item.dialogue_ids, dialogue_ids),
            (item.sources, sources),
            (item.rules, rules),
        ):
            for entry in collection:
                if entry and entry not in target:
                    target.append(entry)
        for fact_id in item.metadata.get("program_selected_fact_ids", []):
            fact_id = str(fact_id).strip()
            if fact_id and fact_id not in program_selected_fact_ids:
                program_selected_fact_ids.append(fact_id)
    if rule and rule not in rules:
        rules.append(rule)
    metadata = {}
    if program_selected_fact_ids:
        metadata["program_selected_fact_ids"] = program_selected_fact_ids
    return Evidence(
        value=value if value is not None else [item.value for item in materialised],
        fact_ids=tuple(fact_ids),
        dialogue_ids=tuple(dialogue_ids),
        sources=tuple(sources),
        rules=tuple(rules),
        metadata=metadata,
    )
