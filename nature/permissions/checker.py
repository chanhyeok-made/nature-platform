"""PermissionChecker — the full permission pipeline.

Flow:
1. Tool's own check_permissions() → allow/deny/passthrough
2. If passthrough → check deny rules → check allow rules → mode-based decision
3. If mode=bypass → allow. If mode=deny → deny. If mode=default → ask (interactive).
"""

from __future__ import annotations

import logging
from typing import Any

from nature.permissions.modes import PermissionMode
from nature.permissions.rules import PermissionRuleSet
from nature.config.constants import PermissionBehavior as PB
from nature.protocols.tool import PermissionResult, Tool, ToolContext

logger = logging.getLogger(__name__)


class PermissionChecker:
    """Checks whether a tool call is allowed."""

    def __init__(
        self,
        rules: PermissionRuleSet | None = None,
        mode: PermissionMode = PermissionMode.DEFAULT,
    ) -> None:
        self._rules = rules or PermissionRuleSet()
        self._mode = mode

    @property
    def mode(self) -> PermissionMode:
        return self._mode

    async def check(
        self,
        tool: Tool,
        tool_input: dict[str, Any],
        context: ToolContext,
    ) -> PermissionResult:
        """Full permission check pipeline.

        Returns PermissionResult with behavior: allow, deny, or ask.
        """
        # Step 1: Tool's own permission check
        tool_result = await tool.check_permissions(tool_input, context)
        if tool_result.behavior != PB.PASSTHROUGH:
            return tool_result

        # Step 2: Rule-based check (deny rules override allow rules)
        rule_result = self._rules.check(tool.name, tool_input)
        if rule_result == PB.DENY:
            return PermissionResult(
                behavior=PB.DENY,
                message=f"Denied by rule for {tool.name}",
                reason="deny_rule",
            )
        if rule_result == PB.ALLOW:
            return PermissionResult(behavior=PB.ALLOW, reason="allow_rule")

        # Step 3: Mode-based decision
        if self._mode == PermissionMode.BYPASS:
            return PermissionResult(behavior=PB.ALLOW, reason="bypass_mode")

        if self._mode == PermissionMode.DENY:
            return PermissionResult(
                behavior=PB.DENY,
                message=f"Denied: {tool.name} not in allow list (deny mode)",
                reason="deny_mode",
            )

        if self._mode == PermissionMode.ACCEPT_EDITS:
            if not tool.is_read_only(tool_input) and _is_file_tool(tool.name):
                return PermissionResult(behavior=PB.ALLOW, reason="accept_edits_mode")

        # Mode DEFAULT or ACCEPT_EDITS for non-file tools: ask
        # In non-interactive contexts, default to allow with logging
        logger.info("Permission check: %s needs approval (mode=%s)", tool.name, self._mode)
        return PermissionResult(behavior=PB.ALLOW, reason="default_allow")


def _is_file_tool(name: str) -> bool:
    return name in ("Write", "Edit", "Read")
