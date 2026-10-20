# MaC2 reference query-planning prompts

These are the active dataset-specific reference prompts in `code/prompts.py`.
`code/program_executor.py` selects category guidance and recommended skeletons.
LoCoMo category profiles and LongMemEval category/date guidance are included in
the generated request. Gold answers and annotated gold evidence are not inputs.

## LoCoMo

### System prompt

```text
Write one complete module-level Python search program over a read-only executable-memory SDK.
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
program directly without prose.
```

### User prompt

```text
Question:
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
memory Evidence, never a textual answer or a catalog record.
```

### Recommended scope by category

| Category | Scope |
| --- | --- |
| 1: Multi-hop | concept + session |
| 2: Temporal | session |
| 3: Open-domain | concept + session |
| 4: Single-hop | none |

## LongMemEval

### System prompt

```text
Write one complete module-level Python search program over a read-only executable-memory SDK.
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
complete program directly without prose.
```

### User prompt

```text
Question:
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
The final `result` must be memory Evidence, never a textual answer or a catalog record.
```

### Common recommended skeleton

```python
code = memory.resolve()
scoped = memory.expand_scope(code, scopes=["concept", "session"])
# RECOMMENDED_DEFAULT: retain an independent lexical route unless it adds no evidence.
lexical = memory.match()
evidence = memory.union(memory.union(code, scoped), lexical)
# OPTIONAL_QUESTION_SPECIFIC_OPERATIONS
result = memory.hydrate(evidence)
```

### Appended guidance

```text
Benchmark category: multi-session
Question date: <question_date>
Planner guidance: Use complete=True queries for counts or lists; deduplicate one repeated event before aggregate().
```

```text
Benchmark category: single-session-user
Question date: <question_date>
Planner guidance: Prefer the exact user-authored fact matching all requested attributes.
```

```text
Benchmark category: temporal-reasoning
Question date: <question_date>
Planner guidance: Match every named event before temporal filtering; the question date is authoritative for relative-time questions.
```

```text
Benchmark category: single-session-preference
Question date: <question_date>
Planner guidance: Retrieve durable user constraints before a recommendation.
```

```text
Benchmark category: knowledge-update
Question date: <question_date>
Planner guidance: Retrieve complete value history and prefer an active-state derivation over an older mention.
```

```text
Benchmark category: single-session-assistant
Question date: <question_date>
Planner guidance: Assistant-authored facts and visual captions are valid evidence when requested.
```

