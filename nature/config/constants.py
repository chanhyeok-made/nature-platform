"""Framework-wide enums and constants.

All magic strings are centralized here. No loose string comparisons.
"""

from __future__ import annotations

from enum import Enum


class ProviderName(str, Enum):
    """Supported LLM provider names."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"


class PermissionBehavior(str, Enum):
    """Permission check result behaviors."""
    ALLOW = "allow"
    DENY = "deny"
    PASSTHROUGH = "passthrough"
    ASK = "ask"


class StopReason(str, Enum):
    """LLM stop reasons."""
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP = "stop"
    STOP_SEQUENCE = "stop_sequence"


class ContentBlockType(str, Enum):
    """Content block types in LLM messages."""
    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    IMAGE = "image"
    THINKING = "thinking"


# ---------------------------------------------------------------------------
# Model catalog for the dashboard settings UI
# ---------------------------------------------------------------------------

# Curated list the dashboard populates its per-agent model <select>
# from. This isn't exhaustive — the goal is one-click access to the
# models we typically want to reach for. Users who need something
# else still have the text input fallback in the CLI or frame.json.
# The dashboard also exposes a "custom..." option that lets the user
# type any model id.
#
# `tier` is informational — it drives the preset defaults (heavy for
# planning/judgment, medium for most work, light for mechanical
# research/review) but doesn't otherwise constrain which model a role
# can use.
MODEL_CATALOG: list[dict] = [
    # Anthropic — current generation
    {"provider": "anthropic", "id": "claude-opus-4-6",          "label": "Claude Opus 4.6",   "tier": "heavy"},
    {"provider": "anthropic", "id": "claude-sonnet-4-6",        "label": "Claude Sonnet 4.6", "tier": "medium"},
    {"provider": "anthropic", "id": "claude-haiku-4-5",         "label": "Claude Haiku 4.5",  "tier": "light"},
    # Anthropic — stable fallbacks
    {"provider": "anthropic", "id": "claude-sonnet-4-5",        "label": "Claude Sonnet 4.5", "tier": "medium"},
    {"provider": "anthropic", "id": "claude-sonnet-4-20250514", "label": "Claude Sonnet 4 (May)", "tier": "medium"},
    # OpenAI (both direct OpenAI and OpenAI-compatible proxies)
    {"provider": "openai",    "id": "gpt-4o",                   "label": "GPT-4o",             "tier": "medium"},
    {"provider": "openai",    "id": "gpt-4o-mini",              "label": "GPT-4o mini",        "tier": "light"},
    # Ollama-popular local models (accessed via openai-compat)
    {"provider": "openai",    "id": "qwen2.5-coder:32b",        "label": "Qwen 2.5 Coder 32B (local)", "tier": "medium"},
    {"provider": "openai",    "id": "qwen2.5:72b-instruct-q4_K_M", "label": "Qwen 2.5 72B (local)", "tier": "heavy"},
    {"provider": "openai",    "id": "deepseek-r1:32b",          "label": "DeepSeek R1 32B (local)", "tier": "medium"},
    {"provider": "openai",    "id": "llama3.2:3b",              "label": "Llama 3.2 3B (local)", "tier": "light"},
]


class CacheControlType(str, Enum):
    """Cache control types for prompt caching."""
    EPHEMERAL = "ephemeral"


class InternalCategory(str, Enum):
    """Categories for internal visibility events."""
    PROMPT = "prompt"
    MEMORY = "memory"
    TOOLS = "tools"
    TURN = "turn"
    CONTEXT = "context"
    API = "api"
    API_REQUEST = "api_request"
    USAGE = "usage"
    RECOVERY = "recovery"
    PARSER = "parser"
    HOOKS = "hooks"
    PERMISSIONS = "permissions"
    ORCHESTRATOR = "orchestrator"
    TURN_PARSED = "turn_parsed"
    ERROR = "error"


# Exit commands (shared between TUI and REPL)
EXIT_COMMANDS = frozenset({"exit", "quit", "/exit", "/quit"})
