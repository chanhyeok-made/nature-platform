"""Domain type aliases — eliminate bare dict[str, Any] across the framework.

These types don't add runtime behavior (they're aliases), but they
communicate intent and enable grep-based auditing.

Rules:
- ToolInput: the input dict passed to a tool (schema varies per tool)
- JsonSchema: a JSON Schema object
- JsonDict: any JSON-serializable dict (API payloads, external boundaries)
- Metadata: key-value metadata attached to results
"""

from __future__ import annotations

from typing import Any, NewType

# Tool input parameters — schema varies per tool, but always str keys
ToolInput = dict[str, Any]

# JSON Schema object (used for tool definitions)
JsonSchema = dict[str, Any]

# Generic JSON-serializable dict (API payloads, external system boundary)
JsonDict = dict[str, Any]

# Key-value metadata
Metadata = dict[str, Any]

# Hook event payload — varies per event type, always str keys
HookPayload = dict[str, Any]

# Session transcript entry
TranscriptEntry = dict[str, Any]
