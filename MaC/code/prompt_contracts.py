"""Versioned MaC2 reference policies; runtime chooses the dataset's planner."""

FACT_TO_CODE_POLICY_VERSION = "mac2_reference_fact_to_code_v1"
QUERY_POLICY_VERSION = "mac2_reference_categories_v1"

FACT_TO_CODE_SYSTEM = """You compile canonical dialogue facts into a code-native memory specification.
Return valid JSON only. Do not invent facts. Each fact will become one named pure Python
function in a subject module. Choose a short snake_case function_name describing the fact,
a compact open-ended query predicate, a JSON-compatible value, optional JSON-object
attributes, and whether the fact is a time-varying state. Do not infer or output any
state-to-state link in this phase. State links are resolved in a
separate phase after all facts have been semantically parsed. Do not create ordinary
fact-to-fact graph relations. All generated base functions return the same open MemoryValue
runtime type; do not classify facts into a fixed type ontology."""

FACT_TO_CODE_USER = """Compile these facts.

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
