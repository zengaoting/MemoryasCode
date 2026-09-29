# Declarative Query Planning Prompt

## Task

Generate a complete Python retrieval program that selects and composes memory operations for the given question.

## Input

The question, compact candidate-function metadata, available memory interfaces, and general human-designed retrieval guidance.

## System prompt

```text
Write one complete program over the read-only executable-memory SDK. Determine the required functions, operations, and dependencies from the question and interface information. Plan the retrieval as a whole rather than requesting further LLM decisions after each operation.

Use the human-designed guidance as reference, not as a fixed pipeline: direct lookup may suffice for a specific fact; distributed evidence may require expansion and composition; temporal questions may require state or history computations and source context. These patterns may not fit every question; adapt or omit them according to the actual evidence requirements.

Use compatible operations to retrieve, execute, filter, and combine evidence as needed. Preserve source information and assign the supporting evidence to result. Do not modify memory or generate the final natural-language answer.
```

## User prompt template

```text
Question:
{{question}}

Candidate functions:
{{functions}}

Generate a complete retrieval program for this question using the available interfaces.
```

## Output

A Python program whose result contains supporting memory evidence.
