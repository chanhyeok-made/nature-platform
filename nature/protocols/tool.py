"""Tool protocol — definition, execution, and permission interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nature.config.constants import PermissionBehavior
from nature.config.types import JsonSchema, Metadata, ToolInput

T = TypeVar("T")


class ToolDefinition(BaseModel):
    """Tool definition sent to the LLM API."""
    name: str
    description: str
    input_schema: JsonSchema
    deferred: bool = False


class ToolContext(BaseModel):
    """Runtime context passed to tool execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cwd: str
    project_root: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    is_read_only: bool = False
    additional_directories: list[str] = Field(default_factory=list)
    # Pack-owned state bag from Frame.pack_state. Injected by
    # AreaManager; None in tests or legacy paths that predate Packs.
    pack_state: dict[str, Any] | None = None


class ToolResult(BaseModel, Generic[T]):
    """Result from a tool execution."""
    output: T
    is_error: bool = False
    metadata: Metadata = Field(default_factory=dict)


class PermissionResult(BaseModel):
    """Result of a permission check."""
    behavior: PermissionBehavior
    message: str | None = None
    updated_input: ToolInput | None = None
    reason: str | None = None


class Tool(ABC):
    """Abstract base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def input_schema(self) -> JsonSchema: ...

    @abstractmethod
    async def execute(self, input: ToolInput, context: ToolContext) -> ToolResult: ...

    def is_concurrency_safe(self, input: ToolInput) -> bool:
        return self.is_read_only(input)

    def is_read_only(self, input: ToolInput) -> bool:
        return False

    def is_destructive(self, input: ToolInput) -> bool:
        return False

    async def check_permissions(
        self, input: ToolInput, context: ToolContext
    ) -> PermissionResult:
        return PermissionResult(behavior=PermissionBehavior.PASSTHROUGH)

    async def validate_input(self, input: ToolInput, context: ToolContext) -> str | None:
        return None

    @property
    def max_result_size_chars(self) -> int:
        return 100_000

    @property
    def deferred(self) -> bool:
        return False

    def to_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            deferred=self.deferred,
        )
