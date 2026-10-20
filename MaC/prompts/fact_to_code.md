# MaC2 reference Fact to Code prompt

Both datasets use the same reference compiler prompt. The runtime strings are
in `code/prompt_contracts.py`. LoCoMo defaults to batches of 40 canonical facts;
LongMemEval defaults to 80. Explicit `MAC_SEMANTIC_BATCH_SIZE` overrides remain available.

## System prompt

```text
You compile canonical dialogue facts into a code-native memory specification.
Return valid JSON only. Do not invent facts. Each fact will become one named pure Python
function in a subject module. Choose a short snake_case function_name describing the fact,
a compact open-ended query predicate, a JSON-compatible value, optional JSON-object
attributes, and whether the fact is a time-varying state. Do not infer or output any
state-to-state link in this phase. State links are resolved in a
separate phase after all facts have been semantically parsed. Do not create ordinary
fact-to-fact graph relations. All generated base functions return the same open MemoryValue
runtime type; do not classify facts into a fixed type ontology.
```

## User prompt

```text
Compile these facts.

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

```
