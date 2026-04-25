"""ID generation utilities."""

from uuid import uuid4


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return uuid4().hex


def generate_tool_use_id() -> str:
    """Generate a unique tool use ID (matching Anthropic format)."""
    return f"toolu_{uuid4().hex[:24]}"


def generate_agent_id(name: str | None = None) -> str:
    """Generate a unique agent ID."""
    base = uuid4().hex[:16]
    if name:
        return f"{name}@{base}"
    return f"agent_{base}"
