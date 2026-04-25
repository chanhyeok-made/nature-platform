"""Default constants for nature.

These match Claude Code's analyzed values where applicable.
"""

# ---------------------------------------------------------------------------
# Token budget defaults
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT_WINDOW = 200_000
DEFAULT_OUTPUT_RESERVATION = 20_000
AUTOCOMPACT_BUFFER_TOKENS = 13_000
WARNING_THRESHOLD_BUFFER_TOKENS = 20_000
ERROR_THRESHOLD_BUFFER_TOKENS = 20_000
MANUAL_COMPACT_BUFFER_TOKENS = 3_000
MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

DEFAULT_BYTES_PER_TOKEN = 4
JSON_BYTES_PER_TOKEN = 2
IMAGE_TOKEN_ESTIMATE = 2_000

# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

DEFAULT_TOOL_TIMEOUT_SECONDS = 120
MAX_TOOL_CONCURRENCY = 10
DEFAULT_MAX_RESULT_SIZE_CHARS = 100_000
BASH_PROGRESS_POLL_INTERVAL_MS = 200

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

MAX_OUTPUT_TOKENS_RECOVERY_LIMIT = 3
MAX_OUTPUT_TOKENS_ESCALATION = [8_192, 16_384, 65_536]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

DYNAMIC_BOUNDARY = "__DYNAMIC_BOUNDARY__"

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

MEMORY_INDEX_MAX_LINES = 200
MEMORY_INDEX_MAX_BYTES = 25_000
MEMORY_RECALL_MAX_FILES = 5

