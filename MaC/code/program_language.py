"""Safe, read-only Python subset for MaC2 search programs."""

from __future__ import annotations

import ast
from dataclasses import dataclass


class ProgramValidationError(ValueError):
    pass


_ALLOWED = {
    "resolve", "expand_scope", "match", "union", "expand_relation", "rank", "hydrate",
    "execute", "select", "search", "search_functions",
    "where", "aggregate", "validate",
}
_STATEMENTS = (ast.Assign, ast.If, ast.Expr, ast.Pass)
_EXPRESSIONS = (ast.Name, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Call,
                ast.Attribute, ast.Subscript, ast.Load, ast.Store, ast.keyword)


def validate_program(program: str) -> ast.Module:
    """Validate a read-only program with optional, dependency-aware scope expansion."""
    try:
        tree = ast.parse(program, mode="exec")
    except SyntaxError as exc:
        raise ProgramValidationError(f"Invalid Python syntax: {exc.msg}") from exc
    if not tree.body:
        raise ProgramValidationError("Program is empty")
    assigned: set[str] = set()

    def expression(node: ast.AST) -> None:
        if not isinstance(node, _EXPRESSIONS):
            raise ProgramValidationError(f"Disallowed expression: {node.__class__.__name__}")
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and node.id not in assigned and node.id != "memory":
                raise ProgramValidationError(f"Unknown variable: {node.id}")
        elif isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name) or node.value.id != "memory" or node.attr not in _ALLOWED:
                raise ProgramValidationError("Only approved memory SDK methods are allowed")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Attribute):
                raise ProgramValidationError("Only memory SDK method calls are allowed")
            expression(node.func)
            for value in [*node.args, *(item.value for item in node.keywords)]:
                expression(value)
            if any(item.arg is None for item in node.keywords):
                raise ProgramValidationError("Starred keyword arguments are not allowed")
        elif isinstance(node, ast.Subscript):
            if not isinstance(node.value, ast.Name) or node.value.id not in assigned:
                raise ProgramValidationError("Subscripts may only read a previous SDK result")
            expression(node.slice)
        elif isinstance(node, (ast.List, ast.Tuple)):
            for item in node.elts:
                expression(item)
        elif isinstance(node, ast.Dict):
            for item in [*node.keys, *node.values]:
                if item is not None:
                    expression(item)

    def block(statements: list[ast.stmt]) -> None:
        for statement in statements:
            if not isinstance(statement, _STATEMENTS):
                raise ProgramValidationError(f"Disallowed statement: {statement.__class__.__name__}")
            if isinstance(statement, ast.Assign):
                if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                    raise ProgramValidationError("Assignments require one simple variable target")
                if statement.targets[0].id == "memory":
                    raise ProgramValidationError("The memory SDK object cannot be reassigned")
                expression(statement.value)
                assigned.add(statement.targets[0].id)
            elif isinstance(statement, ast.Expr):
                if not isinstance(statement.value, ast.Call):
                    raise ProgramValidationError("Only SDK calls may be standalone expressions")
                expression(statement.value)
            elif isinstance(statement, ast.If):
                if not isinstance(statement.test, ast.Subscript):
                    raise ProgramValidationError("if conditions must read a validate result key")
                expression(statement.test)
                block(statement.body)
                block(statement.orelse)

    block(tree.body)
    if "result" not in assigned:
        raise ProgramValidationError("Program must assign the final value to result")
    _validate_flow(tree)
    return tree


def _method(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    return node.func.attr if isinstance(node.func.value, ast.Name) and node.func.value.id == "memory" else None


_EVIDENCE = "evidence"
_CATALOG = "function_catalog"
_REPORT = "validation_report"
_LITERAL = "literal"


@dataclass(frozen=True)
class _FlowValue:
    kind: str
    # Only resolve() writes the runtime resolved_function_id metadata used by
    # expand_scope().  where/rank retain individual Evidence items; merging or
    # hydrating creates a new Evidence and deliberately drops this capability.
    scope_seed: bool = False


def _argument(node: ast.Call, index: int, keyword: str | None = None) -> ast.AST | None:
    if keyword:
        for item in node.keywords:
            if item.arg == keyword:
                return item.value
    return node.args[index] if len(node.args) > index else None


def _need_evidence(value: _FlowValue, method: str) -> None:
    if value.kind != _EVIDENCE:
        raise ProgramValidationError(f"memory.{method}(...) requires evidence from a memory retrieval operation")


def _flow_value(node: ast.AST, values: dict[str, _FlowValue]) -> _FlowValue:
    if isinstance(node, ast.Name):
        return values.get(node.id, _FlowValue(_LITERAL))
    if isinstance(node, ast.Subscript):
        return _flow_value(node.value, values)
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [_flow_value(item, values) for item in node.elts]
        if items and all(item.kind == _EVIDENCE for item in items):
            return _FlowValue(_EVIDENCE, scope_seed=all(item.scope_seed for item in items))
        return _FlowValue(_LITERAL)
    if isinstance(node, ast.Dict) or isinstance(node, ast.Constant):
        return _FlowValue(_LITERAL)
    if not isinstance(node, ast.Call):
        return _FlowValue(_LITERAL)

    method = _method(node)
    if method in {"resolve", "match"}:
        if node.args or node.keywords:
            raise ProgramValidationError(f"memory.{method}() takes no arguments")
        return _FlowValue(_EVIDENCE, scope_seed=method == "resolve")
    if method in {"execute", "select", "search"}:
        return _FlowValue(_EVIDENCE)
    if method == "search_functions":
        return _FlowValue(_CATALOG)

    if method == "expand_scope":
        evidence = _argument(node, 0, "evidence")
        if evidence is None:
            raise ProgramValidationError("memory.expand_scope(...) requires resolve() scope-seed evidence")
        input_value = _flow_value(evidence, values)
        _need_evidence(input_value, method)
        if not input_value.scope_seed:
            raise ProgramValidationError(
                "memory.expand_scope(...) must consume direct resolve() evidence or resolve() filtered by where()/rank()"
            )
        return _FlowValue(_EVIDENCE)
    if method in {"where", "rank", "expand_relation", "hydrate", "aggregate"}:
        evidence = _argument(node, 0, "evidence")
        if evidence is None:
            raise ProgramValidationError(f"memory.{method}(...) requires evidence input")
        input_value = _flow_value(evidence, values)
        _need_evidence(input_value, method)
        return _FlowValue(_EVIDENCE, scope_seed=input_value.scope_seed if method in {"where", "rank"} else False)
    if method == "union":
        left, right = _argument(node, 0, "left"), _argument(node, 1, "right")
        if left is None or right is None:
            raise ProgramValidationError("memory.union(...) requires left and right evidence")
        _need_evidence(_flow_value(left, values), method)
        _need_evidence(_flow_value(right, values), method)
        return _FlowValue(_EVIDENCE)
    if method == "validate":
        evidence = _argument(node, 1, "evidence")
        if evidence is None:
            raise ProgramValidationError("memory.validate(...) requires evidence input")
        _need_evidence(_flow_value(evidence, values), method)
        return _FlowValue(_REPORT)
    return _FlowValue(_LITERAL)


def _validate_flow(tree: ast.Module) -> None:
    values: dict[str, _FlowValue] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            continue
        values[statement.targets[0].id] = _flow_value(statement.value, values)
    result = values.get("result")
    if result is None or result.kind != _EVIDENCE:
        raise ProgramValidationError("result must be provenance-carrying evidence from the memory SDK")
