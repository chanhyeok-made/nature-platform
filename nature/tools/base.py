"""BaseTool — safe defaults so most tools only implement execute().

Matches Claude Code's buildTool() pattern: sensible defaults for
concurrency, permissions, result size, etc.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from pydantic import BaseModel

from nature.protocols.tool import Tool, ToolContext, ToolResult


def _clean_schema(schema: dict) -> None:
    """Inline any `$ref` definitions and strip Pydantic metadata.

    Pydantic v2 emits nested model schemas as `{$ref: '#/$defs/X'}`
    plus a top-level `$defs` table. Most LLM tool-call APIs accept
    that shape, but it makes the prompt harder to read and breaks
    when a downstream consumer drops `$defs`. This helper resolves
    every $ref against the local `$defs` table (in-place, recursively)
    and then strips `title`, `$defs`, and similar metadata so the
    final schema is fully self-contained.
    """
    defs = schema.get("$defs", {}) or {}
    _resolve_refs(schema, defs)
    schema.pop("title", None)
    schema.pop("$defs", None)
    _strip_titles(schema)


def _resolve_refs(node: Any, defs: dict[str, dict]) -> None:
    """Walk a JSON-schema-shaped value and replace `$ref` with inline
    copies of the matching `$defs[name]` entry. Mutates in place."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            target_name = ref.split("/")[-1]
            target = defs.get(target_name)
            if target is not None:
                # Inline a deep copy so the same definition can be used
                # by multiple $refs without sharing mutable state.
                import copy
                inlined = copy.deepcopy(target)
                # Recurse into the inlined target — it might itself
                # carry nested $refs.
                _resolve_refs(inlined, defs)
                node.clear()
                node.update(inlined)
                return
        for value in list(node.values()):
            _resolve_refs(value, defs)
    elif isinstance(node, list):
        for item in node:
            _resolve_refs(item, defs)


def _strip_titles(node: Any) -> None:
    """Remove Pydantic-generated `title` keys throughout a schema tree."""
    if isinstance(node, dict):
        node.pop("title", None)
        for value in node.values():
            _strip_titles(value)
    elif isinstance(node, list):
        for item in node:
            _strip_titles(item)


class BaseTool(Tool):
    """Base class for nature tools with safe defaults.

    Subclasses must implement:
    - name (property)
    - description (property)
    - input_model (class attribute) — a Pydantic model for input
    - run(input_model, context) — the actual logic

    The base class handles:
    - JSON Schema generation from the Pydantic model
    - Input parsing and validation
    - Default permission/concurrency/read-only behavior
    """

    input_model: type[BaseModel]  # Override in subclass

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    async def run(self, params: BaseModel, context: ToolContext) -> ToolResult:
        """Execute with typed params. Override this instead of execute()."""
        ...

    @property
    def input_schema(self) -> dict[str, Any]:
        """Auto-generate JSON Schema from the Pydantic input_model."""
        schema = self.input_model.model_json_schema()
        _clean_schema(schema)
        return schema

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        """Parse input via Pydantic model, then delegate to run()."""
        try:
            params = self.input_model.model_validate(input)
        except Exception as e:
            return ToolResult(output=f"Invalid input: {e}", is_error=True)
        return await self.run(params, context)
