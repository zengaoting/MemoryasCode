# Answer Generation

Reference for the answer templates in [code/prompts.py](../code/prompts.py). The active dataset selects the system prompt. This Markdown file is documentation and is not loaded by the pipeline.

The validated retrieval program runs against executable memory. Its returned evidence is assembled into queried contents for answer generation, with original dialogue, canonical facts, and retrieval provenance. [code/qa_engine.py](../code/qa_engine.py) applies category-specific instructions and conditional answer verification or repair, so one question may require multiple model calls.

## LoCoMo System Prompt

```text
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

Return valid JSON only.
```

## LongMemEval System Prompt

```text
   You must answer the question with queried contents.
      Rules:
      -  Original Dialogue Evidence and Deduplicated Raw Facts are authoritative. The Executed Search Trace only records retrieval provenance; it is not a fact and cannot override, narrow, aggregate, or replace the original evidence.
      -  Match the full requested entity, event, relation, and time constraints before answering. Do not substitute a merely similar event or add related facts that were not asked for.
      -  For yes/no or binary questions, output 'Yes', 'No', 'Likely yes', 'Likely no'.
      -  For time/date questions, first identify the exact event. Original dialogue wording and its session_datetime outrank a normalized fact time when their granularity conflicts. Preserve coarse expressions such as next month, last week, a few days ago, or about four months; do not invent a day-of-month.
      -  Image caption/query text shown as `visual=` is source evidence for what an image depicts and may disambiguate an otherwise generic utterance.
      -  If `Deterministic Temporal Calculations` contains an operation matching the question and cites the matched source turns, use its `answer_text` exactly.
      -  For cross-session count or list questions, internally enumerate distinct qualifying events with their dialogue ids, merge repeated mentions of the same event, and do not count plans or merely related events.
      -  For knowledge-update questions, compare all states of the requested attribute and return the latest active state as of the supplied question date; corrections, cancellations, replacements, and completed plans change the state.
      -  For preference questions, infer supported preferences and constraints, then give concrete recommendations that satisfy them instead of only restating the preference.
      -  Assistant turns, quoted text, tables, and `visual=` image descriptions are valid evidence when the question asks about assistant-provided content.
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

Return valid JSON only.
```

## User Prompt (template)

```text
Output format:

{
  "mode": "answer",
  "answer": "...",
  "supports": ["D..."],
  "confidence": 0.0
}

Question:
{{question}}

Queried contents:
{{loaded_memory}}
```

## Category-Specific Instructions

`format_final_question_for_mragent()` in [code/qa_engine.py](../code/qa_engine.py) appends the active category's instructions to the question before filling the user template:

| Dataset | Category | Additional instructions |
| --- | --- | --- |
| LoCoMo | `1` (multi-hop) | Combine requested relations, enumerate evidence, and deduplicate qualifying events for counts. |
| LoCoMo | `2` (temporal) | Match the exact event, resolve supported relative times, and preserve source time granularity. |
| LoCoMo | `3` (open-domain) | Keep `answer` concise and provide original-text reasons in `reason`. |
| LongMemEval | `multi-session` | Combine evidence across sessions and deduplicate qualifying events or items. |
| LongMemEval | `temporal-reasoning` | Use the question date and source times, including matching deterministic temporal calculations. |
| LongMemEval | `knowledge-update` | Resolve state history and return the latest active state as of the question date. |
| LongMemEval | `single-session-preference` | Give concrete recommendations consistent with supported preferences and constraints. |
| LongMemEval | `single-session-assistant` | Treat assistant content, quotes, tables, and image descriptions as source evidence. |

Other categories use the selected dataset's system prompt without an additional question suffix. The code also contains conditional verification and repair prompts; the templates above describe the initial answer-generation request.

## Template Variables

| Variable | Source |
| --- | --- |
| `{{question}}` | Question with applicable category-specific instructions. |
| `{{loaded_memory}}` | Queried contents assembled from retrieval results, including source dialogue, canonical facts, and execution provenance. |
