"""Storage protocols — session persistence and memory management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from nature.config.types import TranscriptEntry

from nature.protocols.message import Message


# ---------------------------------------------------------------------------
# Memory types
# ---------------------------------------------------------------------------

class MemoryType(str, Enum):
    """Four closed memory types (from Claude Code analysis)."""
    USER = "user"           # User role, preferences, knowledge
    FEEDBACK = "feedback"   # Guidance on how to approach work
    PROJECT = "project"     # Non-derivable project context
    REFERENCE = "reference"  # External system pointers


class Memory(BaseModel):
    """A single memory entry."""
    name: str
    description: str
    type: MemoryType
    content: str
    file_path: str | None = None


# ---------------------------------------------------------------------------
# Memory store protocol
# ---------------------------------------------------------------------------

class MemoryStore(ABC):
    """Abstract base class for memory persistence."""

    @abstractmethod
    async def save(self, memory: Memory) -> str:
        """Save a memory entry. Returns the file path."""
        ...

    @abstractmethod
    async def load(self, file_path: str) -> Memory | None:
        """Load a memory entry by file path."""
        ...

    @abstractmethod
    async def list_all(self) -> list[Memory]:
        """List all memory entries."""
        ...

    @abstractmethod
    async def delete(self, file_path: str) -> bool:
        """Delete a memory entry. Returns True if deleted."""
        ...

    @abstractmethod
    async def load_index(self) -> str:
        """Load the MEMORY.md index content."""
        ...

    @abstractmethod
    async def save_index(self, content: str) -> None:
        """Save the MEMORY.md index content."""
        ...


# ---------------------------------------------------------------------------
# Session store protocol
# ---------------------------------------------------------------------------

class SessionMetadata(BaseModel):
    """Metadata for a session."""
    session_id: str
    project_root: str
    cwd: str
    title: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: str | None = None
    model: str | None = None


class SessionStore(ABC):
    """Abstract base class for session transcript persistence."""

    @abstractmethod
    async def append(self, session_id: str, entry: TranscriptEntry) -> None:
        """Append an entry to the session transcript (JSONL)."""
        ...

    @abstractmethod
    async def load_messages(self, session_id: str) -> list[Message]:
        """Load all messages from a session transcript."""
        ...

    @abstractmethod
    async def get_metadata(self, session_id: str) -> SessionMetadata | None:
        """Get session metadata (from head + tail of transcript)."""
        ...

    @abstractmethod
    async def list_sessions(self, project_root: str) -> list[SessionMetadata]:
        """List all sessions for a project."""
        ...
