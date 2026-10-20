"""Dialogue extraction, retrieval, planning, and answer prompt templates."""

import re

SESSION_MEMORY_EXTRACTOR_SYSTEM = """You are a memory extractor for long-term dialogue QA.

Your task is to read one dialogue session and extract atomic memory facts that may help answer future questions.

Do not write a general summary.
Extract reusable facts.

Each fact should:
- be self-contained
- preserve the speaker
- preserve the subject the fact is about
- include the dialogue id
- include the session id and session datetime
- never output a fact_id field; the system assigns stable canonical fact IDs after normalisation and deduplication
- preserve important time expressions
- normalize dates when confidently possible
- treat normalized_time as the absolute time of the fact/event, not the session conversation time
- if fact_text contains a relative time expression such as yesterday, tomorrow, last week, last month, recently, or next week, and it can be resolved from session_datetime, write the resolved absolute date/time directly in fact_text
- preserve the original relative time expression in time_text even when fact_text is rewritten with an absolute date/time
- preserve concrete details useful for future QA
- avoid unsupported inference

Extract facts about:
- personal identity or background
- relationships
- activities
- events
- plans
- hobbies
- preferences
- motivations
- emotional reactions
- possessions
- important places
- important dates
- repeated patterns

Return valid JSON only."""

SESSION_MEMORY_EXTRACTOR_USER = """Output format:

Do not include a fact_id field in any atomic fact.

{
  "atomic_facts": [
    {
      "sample_id": "{{sample_id}}",
      "session_id": "{{session_id}}",
      "session_datetime": "{{session_datetime}}",
      "dia_id": "...",
      "speaker": "...",
      "subject": "...",
      "fact_text": "...",
      "fact_type": "profile | relationship | activity | event | plan | preference | motivation | emotion | possession | place | temporal | other",
      "topics": ["..."],
      "time_text": "...",
      "normalized_time": "..."
    }
  ]
}

Sample metadata:
sample_id: {{sample_id}}
speaker_a: {{speaker_a}}
speaker_b: {{speaker_b}}
session_id: {{session_id}}
session_datetime: {{session_datetime}}

Session messages:
{{session_messages}}"""

RETRIEVAL_TERM_EXTRACTOR_SYSTEM = """You extract retrieval terms for long-term dialogue QA.

Your task is to read one user question and return structured terms for local retrieval.

Rules:
- Return valid JSON only.
- Do not answer the question.
- Extract terms only from the original question.
- Keep names, places, objects, events, dates, and domain-specific phrases.
- Preserve important multi-word phrases.
- Remove task boilerplate such as "No extra explanations", "Give reasons with original text", answer-option instructions, and "Not mentioned in the conversation".
- Put close synonyms or useful paraphrases only in expanded_terms.
- For a named country, region, city, venue, or institution, include up to four widely known aliases or contained-place names when memory may use the other geographic level (for example Brazil -> Rio de Janeiro).
- Expand event-state paraphrases that preserve meaning, such as starting a professional career -> signing or joining a professional team.
- Put terms that should not drive retrieval in avoid_terms.
"""

RETRIEVAL_TERM_EXTRACTOR_USER = """Output format:

{
  "question_type": "where | when | who | what | why | how | yes_no | other",
  "entities": ["..."],
  "phrases": ["..."],
  "keywords": ["..."],
  "time_terms": ["..."],
  "expanded_terms": ["..."],
  "avoid_terms": ["..."]
}

Question:
{{question}}"""

LOCOMO_ANSWER_GENERATOR_SYSTEM = """
   You must answer the question with queried contents.
      Rules:
      -  Original Dialogue Evidence and Deduplicated Raw Facts are authoritative. The Executed Search Trace only records retrieval provenance; it is not a fact and cannot override, narrow, aggregate, or replace the original evidence.
      -  Match the full requested entity, event, relation, and time constraints before answering. Do not substitute a merely similar event or add related facts that were not asked for.
      -  For yes/no or binary questions, output 'Yes', 'No', 'Likely yes', 'Likely no'.
      -  For time/date questions, first identify the exact event. Original dialogue wording and its session_datetime outrank a normalized fact time when their granularity conflicts. Preserve coarse expressions such as next month, last week, a few days ago, or about four months; do not invent a day-of-month.
      -  Image caption/query text shown as `visual=` is source evidence for what an image depicts and may disambiguate an otherwise generic utterance.
      -  If `Deterministic Temporal Calculations` contains an operation matching the question and cites the matched source turns, use its `answer_text` exactly.
      -  For "where / location / place" questions, the answer should be a concrete and specific place name. If no exact name is mentioned, describe it instead.
      -  For "what / which" questions, try to respond with one specific, concrete item directly asked for or descriptions of the answer.
      -  For other questions, output only the minimal answer (key phrase or entity) without extra context.

       Format:
       - "answer": If the events already provide sufficient evidence to answer the question, then produce the final short answer, only asked part, not full sentence.
       {
         "mode": "answer",
         "answer": "...",
         "supports": ["D1:1","D1:2"],
         "confidence": 0.0-1.0
       }

Return valid JSON only."""

# Keep the original LoCoMo prompt above byte-for-byte in its substantive
# instructions. LongMemEval adds its own benchmark-specific rules here.
LONGMEMEVAL_ANSWER_GENERATOR_SYSTEM = LOCOMO_ANSWER_GENERATOR_SYSTEM.replace(
    "      -  For \"where / location / place\" questions, the answer should be a concrete and specific place name. If no exact name is mentioned, describe it instead.",
    "      -  For cross-session count or list questions, internally enumerate distinct qualifying events with their dialogue ids, merge repeated mentions of the same event, and do not count plans or merely related events.\n"
    "      -  For knowledge-update questions, compare all states of the requested attribute and return the latest active state as of the supplied question date; corrections, cancellations, replacements, and completed plans change the state.\n"
    "      -  For preference questions, infer supported preferences and constraints, then give concrete recommendations that satisfy them instead of only restating the preference.\n"
    "      -  Assistant turns, quoted text, tables, and `visual=` image descriptions are valid evidence when the question asks about assistant-provided content.\n"
    "      -  For \"where / location / place\" questions, the answer should be a concrete and specific place name. If no exact name is mentioned, describe it instead.",
)


def answer_generator_system() -> str:
    """Return the original answer prompt for the active benchmark."""
    from .benchmark import DATASET_NAME

    return LONGMEMEVAL_ANSWER_GENERATOR_SYSTEM if DATASET_NAME == "LongMemEval" else LOCOMO_ANSWER_GENERATOR_SYSTEM

ANSWER_GENERATOR_USER = """Output format:

{
  "mode": "answer",
  "answer": "...",
  "supports": ["D..."],
  "confidence": 0.0
}

Question:
{{question}}

Queried contents:
{{loaded_memory}}"""


def fill_template(template: str, **kwargs) -> str:
    result = template
    for key, value in kwargs.items():
        result = result.replace("{{" + key + "}}", str(value))
    return result


ANSWER_GENERATOR_SYSTEM = LOCOMO_ANSWER_GENERATOR_SYSTEM
LONGMEM_ANSWER_GENERATOR_SYSTEM = LONGMEMEVAL_ANSWER_GENERATOR_SYSTEM


LOCOMO_PROGRAM_SYSTEM = """Write one complete module-level Python search program over a read-only executable-memory SDK.
Do not answer the question. Assign provenance-carrying memory Evidence to `result`.

Use the category-profile skeleton unless the question clearly makes an additional step
unnecessary. `memory.resolve()` is the code-function route and `memory.match()` is the
independent lexical-fact route. Strongly recommend retaining both routes, unioning them, and
hydrating their merged evidence: this is the common recall-and-source-evidence base for every
LoCoMo category.
`memory.expand_scope(code, scopes=...)` is MaC2's optional concept/session expansion: it may
only consume direct `resolve()` evidence, or that evidence narrowed by `where()` or `rank()`.
Run it before `union()` or `hydrate()`, because merged/hydrated evidence no longer carries the
function locator needed for scope expansion. When two or more evidence routes are used, normally
merge them with `memory.union(...)`. `memory.hydrate(...)` is strongly recommended before
answering because it materialises canonical facts and source windows, but direct Evidence may be
returned when already sufficient.

All other operations are question-specific choices: `expand_relation`, `select`, `where`,
`execute`, `join`, `aggregate`, `rank`, `fetch`, and `validate`. Use `execute(function_id)`
for compact function candidates. Never import, loop, define a function/class, access
files/network, or modify memory.

Allowed calls: resolve, expand_scope, match, union, expand_relation, rank, hydrate,
execute, materialize, select, search, search_functions, where, join, aggregate, validate,
fetch. Every operation must consume compatible earlier evidence, and `result` must be memory
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


LONGMEMEVAL_PROGRAM_SYSTEM = """Write one complete module-level Python search program over a read-only executable-memory SDK.
Do not answer the question. Assign provenance-carrying memory Evidence to `result`.

Use the recommended default skeleton unless the question clearly does not need one or more
steps. `memory.resolve()` is the code-function route and `memory.match()` is the independent
lexical-fact route; both are strongly recommended, but neither is mandatory for every question.
`memory.expand_scope(code, scopes=...)` is MaC2's optional concept/session expansion: it may
only consume direct `resolve()` evidence, or that evidence narrowed by `where()` or `rank()`.
Run it before `union()` or `hydrate()`, because merged/hydrated evidence no longer carries the
function locator needed for scope expansion. When two or more evidence routes are used, normally
merge them with `memory.union(...)`. `memory.hydrate(...)` is strongly recommended before
answering because it materialises canonical facts and source windows, but direct Evidence may be
returned when already sufficient.

All other operations are question-specific choices: `expand_relation`, `select`, `where`,
`execute`, `join`, `aggregate`, `rank`, `fetch`, and `validate`. Use `execute(function_id)`
for compact function candidates. Never import, loop, define a function/class, access
files/network, or modify memory.

Allowed calls: resolve, expand_scope, match, union, expand_relation, rank, hydrate,
execute, materialize, select, search, search_functions, where, join, aggregate, validate,
fetch. Every operation must consume compatible earlier evidence, and `result` must be memory
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


LONGMEMEVAL_PROGRAM_USER = """Question:
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


PROGRAM_TEMPLATE = """code = memory.resolve()
scoped = memory.expand_scope(code, scopes=["concept", "session"])
# RECOMMENDED_DEFAULT: retain an independent lexical route unless it adds no evidence.
lexical = memory.match()
evidence = memory.union(memory.union(code, scoped), lexical)
# OPTIONAL_QUESTION_SPECIFIC_OPERATIONS
result = memory.hydrate(evidence)"""


ANSWER_REFUSAL_PATTERNS = (
    re.compile(r"^\s*(?:unknown|unspecified|not specified|n/?a)\s*[.!]?\s*$", re.I),
    re.compile(r"\bnot (?:explicitly )?(?:mentioned|stated|provided|specified|identified|named)\b", re.I),
    re.compile(
        r"\b(?:cannot|can't|unable to) (?:be )?"
        r"(?:determin(?:e|ed)|identif(?:y|ied)|infer(?:red)?|answer(?:ed)?)\b",
        re.I,
    ),
    re.compile(r"\b(?:no|insufficient) (?:specific |relevant )?(?:information|evidence|details)\b", re.I),
    re.compile(
        r"\b(?:the )?(?:information|answer|name|detail) (?:is|was) not "
        r"(?:present|available|contained|included)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:the )?(?:provided |retrieved )?(?:text|facts?|contents?) "
        r"(?:does|do) not contain (?:any )?(?:information|evidence|details?)\b",
        re.I,
    ),
)


ANSWER_REFUSAL_REPAIR_INSTRUCTION = """

Your previous response refused to answer even though retrieved contents are available.
For this answerable benchmark question, return the most specific short answer supported by
the retrieved facts. If the exact name is not literal, make the strongest reasonable
inference from the description and common knowledge. Do not answer that the information
is missing, unspecified, or impossible to determine. Return the same valid JSON format.
""".strip()


ANSWER_REFUSAL_FINAL_REPAIR_INSTRUCTION = """

The revised response still refused to answer. Inspect the retrieved fact lines one by one,
match descriptions and paraphrases to the entity, event, place, date, or item requested,
and return the single most plausible concrete answer. A benchmark answer may require a
small commonsense inference from an unnamed description. Do not repeat any missing-
information statement. Return the same valid JSON format with supporting dialogue ids.
""".strip()


ANSWER_SCHEMA_REPAIR_INSTRUCTION = """

Your previous response was valid JSON but did not contain a usable non-empty "answer".
This benchmark question is answerable from the retrieved facts. Return mode "answer" and
the same JSON schema with a concrete non-empty answer, a supports list containing only
visible dialogue ids, and confidence. Do not return navigate mode, an empty answer, or a
missing-answer object.
""".strip()


MULTI_HOP_VERIFY_INSTRUCTION = """

This is a Category 1 multi-hop benchmark question. Conservatively verify the draft below
against the retrieved fact lines. Check whether it combines the facts needed by the
question and whether it omits any requested part. If the draft is already supported and
complete, copy its answer and supports unchanged. Revise it only when the retrieved facts
show a concrete omission or mismatch. Return the same short-answer JSON format without
extra explanation.

Draft response:
{draft}
""".strip()


MULTI_HOP_EXHAUSTIVE_SYSTEM = """
You audit exhaustive multi-hop answers against long-term dialogue evidence. Build an
internal event ledger before answering: one row per distinct qualifying event or item,
with its dialogue id. Merge repeated descriptions of the same event, but do not discard
an event merely because it is expressed as winning a trophy, adopting a named pet, or
another result that satisfies the question. Exclude plans, goals, feelings, advice, and
topic-neighbor facts unless the question asks for them. A count row requires evidence of
the completed outcome: merely describing a comeback, final buzzer, celebration, or good
feeling without stating the result does not add an event. A later "winning was awesome"
or trophy reference that clearly refers back to the same earlier trophy is one event, not
another. For two-person questions, verify
each person separately before taking a union or intersection. Return valid JSON only with
the same short-answer schema; do not include the ledger or an explanation.
""".strip()


MULTI_HOP_EXHAUSTIVE_USER = """
Question:
{question}

Retrieved evidence:
{loaded_memory}

Draft answer:
{draft}

Audit completeness and event identity, then return the corrected concise answer.
""".strip()


TEMPORAL_AUDIT_SYSTEM = """
You audit temporal dialogue answers. First match the full person, event, object, place,
and requested answer type; retrieval rank alone is never evidence. Then resolve time with
this precedence: (1) original dialogue time wording plus session_datetime, (2) a direct
onset/completion event, (3) a fact's normalized time only when it preserves the original
granularity. Do not turn "next month", "last week", "a few days ago", or "about four
months" into an unjustifiably exact day. For "when did X start", subtract an explicit
duration from the source session date, unless a direct start/signing event is present.
When two source statements constrain the same event, intersect them (for example, a
month-level plan plus "last week" in an early-month session can establish early month).
A follow-up report may omit a location stated in the earlier plan; link them when person,
event type, and timeline identify the same unique event.
Return the resulting absolute month/year, not "N months ago". For "after how many weeks
did X reconnect", measure between the session that reports the earlier contact and the
reconnection session in whole conversational weeks; do not first move the earlier report
back again because it says "last week". When a deterministic elapsed-week calculation is
provided, use its value instead of repeating calendar arithmetic in prose.
For any matching deterministic temporal operation, copy its `answer_text` exactly.
Keep the requested unit in the answer (for example, "3 weeks", not bare "3").
For "where" questions return a place, never a time span. For "where during [period]"
questions, require location-bearing evidence whose stay
overlaps that period. An unrelated event inside the period does not establish its
location; a statement that the person returned from a place immediately after the period
supports that they had been in that place. Return valid JSON only with the same concise
answer/support schema and no explanation.
""".strip()


TEMPORAL_AUDIT_USER = """
Question:
{question}

Retrieved evidence:
{loaded_memory}

Draft answer:
{draft}

Recheck event identity, answer type, temporal granularity, and arithmetic.
""".strip()


OPEN_DOMAIN_GEOGRAPHY_SYSTEM = """
You independently solve geographic open-domain dialogue questions. Dialogue facts provide
place clues, while the requested answer may require public geographic knowledge. Enforce
the exact requested level: city, US state, or country. Search every retrieved trip/place
fact, not only the nearest match. If the question asks for an additional country, exclude
the country already named and map another visited city to its country. If a draft is
already the correct requested location, preserve it. Return valid JSON only with answer,
supports, and confidence; supports may only cite visible dialogue ids.
""".strip()


OPEN_DOMAIN_GEOGRAPHY_USER = """
Question:
{question}

Retrieved facts:
{loaded_memory}

Draft (may use the wrong geographic level or place):
{draft}

Return the most plausible short geographic answer.
""".strip()


OPEN_DOMAIN_GEOGRAPHY_TERM_PATTERN = re.compile(
    r"\b(?:state|states|country|countries)\b", re.I
)


OPEN_DOMAIN_GEOGRAPHY_QUESTION_PATTERN = re.compile(
    r"^\s*(?:what|which|in\s+what|in\s+which)\b", re.I
)


OPEN_DOMAIN_ALTERNATIVE_SYSTEM = """
You independently solve alternative-choice open-domain dialogue questions. The question
contains alternatives separated by "or", so a bare yes/no draft does not answer which
alternative applies. Use the retrieved facts and, when needed, ordinary public knowledge
to select the specific named alternative that best fits. Return only that concise choice,
not Yes, No, Likely yes, or Likely no. Return valid JSON only with answer, supports, and
confidence; supports may only cite visible dialogue ids.
""".strip()


OPEN_DOMAIN_ALTERNATIVE_USER = """
Question:
{question}

Retrieved facts:
{loaded_memory}

Invalid binary draft:
{draft}

Return the most plausible specific alternative from the question.
""".strip()


OPEN_DOMAIN_BINARY_ANSWER_PATTERN = re.compile(
    r"^\s*(?:likely\s+)?(?:yes|no)\s*[.!]?\s*$", re.I
)


OPEN_DOMAIN_ALTERNATIVE_PATTERN = re.compile(r"\bor\b", re.I)


OPEN_DOMAIN_EXPLICIT_BINARY_PATTERN = re.compile(r"\byes\s+or\s+no\b", re.I)


SINGLE_HOP_EXACT_ATTRIBUTE_PATTERN = re.compile(
    r"\b(color|game|movie|instrument|food|flower|band|song|book|show|name|"
    r"number|phone|emotion|feeling|sentiment|type|kind|activity|sport|hobby|"
    r"place|country|state|city|restaurant|bar|company|brand|car|pet|dog|cat|"
    r"gift|item|advice|quote)\b",
    re.I,
)


LOCOMO_ANSWER_INSTRUCTIONS = {
    1: (
        ' Resolve every requested relation jointly and return all and only the requested items. Before '
        'answering, build an internal list of qualifying evidence by dialogue id. For count questions, '
        'count distinct real events: merge repeated descriptions of one event, include result statements '
        'such as an explicitly won trophy when they satisfy the event, and exclude plans, goals, '
        'feelings, advice, implied-but-unstated outcomes, and topic-neighbor facts. Merge later comments '
        'about the same trophy or victory into that earlier event. For two-person questions, evaluate '
        'each person before taking the requested union or intersection. Output no explanation.'
    ),
    2: (
        ' First identify the exact event using all person, object, place, relation, and date constraints;'
        ' retrieval rank is not authority and a similar event must not be substituted. Resolve an exact '
        "relative date only when the original wording supports that precision; 'yesterday' of "
        "conversation time '7 May 2023' is '6 May 2023'. Original dialogue wording outranks "
        'normalized_time when the latter invents finer precision. Intersect multiple source constraints '
        "on the same event rather than choosing only one; a month-level plan plus 'last week' in an "
        'early-month report can establish early month. A follow-up report may omit the location from an '
        'earlier plan; link them when person, event type, and timeline identify the same unique event. '
        "For 'when did X start' questions, convert an explicit elapsed duration into an absolute "
        "month/year using session_datetime; do not answer only 'N months ago'. For 'after how many weeks "
        "did X reconnect', compare the two reporting-session dates in whole weeks and do not backshift "
        "the first report a second time because it says 'last week'. Use the deterministic elapsed-week "
        "calculation in the evidence when present. For 'where during a period', use only location-bearing"
        ' travel/stay evidence that overlaps the period; an unrelated event does not inherit a hometown, '
        "and returning from a place immediately after the period supports presence there. For 'when' "
        "questions, use the matched evidence's granularity: '7 May 2023', 'May 2023', '2023', or 'the "
        "week/weekend/Sunday before 25 May 2023'; preserve a week or weekend as that anchored period "
        "rather than an ISO date range. Preserve 'next month', 'last week', 'a few days ago', and "
        "approximate durations at their supported granularity, with no additional words. For 'how long' "
        'questions, return the duration exactly as written in the conversation. Do not exclude a source '
        "merely because the question's person is not the literal speaker of that turn; match the complete"
        ' event and its requested attribute before resolving its time.'
    ),
    3: (
        " No extra explanations in 'answer'. Give reasons with original text in 'reason'. "
    ),
}


LONGMEMEVAL_ANSWER_INSTRUCTIONS = {
    'multi-session': (
        ' Resolve every requested relation jointly across all relevant sessions. Before answering, build '
        'an internal ledger with one row per distinct qualifying event or item and its dialogue id. For '
        'count/list questions, merge repeated descriptions of the same event, but keep distinct completed'
        ' events. Exclude plans, goals, feelings, advice, and topic-neighbor facts unless asked. For '
        'multiple people, evaluate each person before the requested union or intersection. Output only '
        'the answer.'
    ),
    'temporal-reasoning': (
        ' Match the exact event before resolving time. Use the question date, source session_datetime, '
        'and original relative wording together; retrieval rank is not authority. Prefer any matching '
        'Deterministic Temporal Calculation and copy its answer_text exactly. Preserve the requested unit'
        ' and supported granularity, and never substitute a similar event or invent a day-of-month.'
    ),
    'knowledge-update': (
        ' Resolve all statements about the requested attribute as a state history. Prefer the latest '
        'explicitly active state as of the question date. Treat corrections, replacements, completion, '
        'cancellation, and changed plans as updates rather than independent answers.'
    ),
    'single-session-preference': (
        " Infer the user's durable preferences and constraints from the cited dialogue, then give "
        'concrete useful recommendations satisfying them. Do not merely restate the preference or invent '
        'constraints.'
    ),
    'single-session-assistant': (
        ' Treat assistant turns, quoted text, tables, and image descriptions as valid source evidence. '
        'Return the exact requested attribute from that content.'
    ),
}
