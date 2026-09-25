# Fact Extraction

Reference for the session-level extraction templates in [code/prompts.py](../code/prompts.py). Runtime prompt assembly and extraction are implemented in [code/memory_extractor.py](../code/memory_extractor.py). This Markdown file is documentation and is not loaded by the pipeline.

The extractor reads dialogue sessions and produces atomic facts. Canonical fact IDs are assigned later, after normalization and deduplication in [code/memory_builder.py](../code/memory_builder.py).

## System Prompt

```text
You are a memory extractor for long-term dialogue QA.

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

Return valid JSON only.
```

## User Prompt (template)

```text
Output format:

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
{{session_messages}}
```

## Runtime Additions

`_build_session_extraction_prompt()` inserts the following instruction before `Session messages:`. `{fact_limit}` comes from the session fact limit configured through `SESSION_FACT_MAX_FACTS_PER_CALL`.

```text
Extraction limit: return at most {fact_limit} high-value atomic facts. Prioritize user-specific memories, user-stated preferences, plans, events, places, relationships, and repeated interests. Also preserve answer-bearing information provided by the assistant, including concrete facts, recommendations, instructions, lists, and table entries that a later question may ask the user to recall.
```

## Template Variables

| Variable | Source |
| --- | --- |
| `{{sample_id}}` | Dataset loader's sample identifier. |
| `{{speaker_a}}`, `{{speaker_b}}` | Conversation metadata. |
| `{{session_id}}` | Session identifier. |
| `{{session_datetime}}` | Session datetime. |
| `{{session_messages}}` | Formatted source dialogue messages. |
