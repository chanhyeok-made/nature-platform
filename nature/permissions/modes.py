"""Permission modes — determines default behavior for tool execution."""

from __future__ import annotations

from enum import Enum


class PermissionMode(str, Enum):
    """How the system handles tool permission requests.

    DEFAULT:         Prompt user for every non-whitelisted tool
    ACCEPT_EDITS:    Auto-allow file edits in project dir, prompt others
    BYPASS:          Allow all tools (tracked, for trusted environments)
    DENY:            Deny non-whitelisted operations (CI/automation)
    """
    DEFAULT = "default"
    ACCEPT_EDITS = "accept_edits"
    BYPASS = "bypass"
    DENY = "deny"
