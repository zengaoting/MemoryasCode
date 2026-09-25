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

ANSWER_GENERATOR_SYSTEM = """
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
LONGMEM_ANSWER_GENERATOR_SYSTEM = ANSWER_GENERATOR_SYSTEM.replace(
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

    return LONGMEM_ANSWER_GENERATOR_SYSTEM if DATASET_NAME == "LongMemEval" else ANSWER_GENERATOR_SYSTEM

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
