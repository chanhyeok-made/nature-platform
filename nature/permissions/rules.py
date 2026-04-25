"""Permission rules — parse and match 'Tool(pattern)' syntax.

Examples:
    "Bash"              — matches all Bash calls
    "Bash(git *)"       — matches Bash with commands starting with 'git '
    "Read(*.py)"        — matches Read for .py files
    "Write"             — matches all Write calls
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PermissionRule:
    """A parsed permission rule."""
    tool_name: str
    pattern: str | None = None  # None = matches all calls to this tool
    raw: str = ""

    def matches(self, tool_name: str, tool_input: dict[str, Any]) -> bool:
        """Check if this rule matches a tool call."""
        if self.tool_name != tool_name:
            return False
        if self.pattern is None:
            return True
        # Match pattern against any string value in input
        for v in tool_input.values():
            if isinstance(v, str) and fnmatch.fnmatch(v, self.pattern):
                return True
        return False


def parse_rule(rule_str: str) -> PermissionRule:
    """Parse a 'Tool(pattern)' string into a PermissionRule.

    Examples:
        "Bash"           → PermissionRule(tool_name="Bash", pattern=None)
        "Bash(git *)"    → PermissionRule(tool_name="Bash", pattern="git *")
        "Read(*.py)"     → PermissionRule(tool_name="Read", pattern="*.py")
    """
    match = re.match(r"^(\w+)(?:\((.+)\))?$", rule_str.strip())
    if not match:
        return PermissionRule(tool_name=rule_str.strip(), raw=rule_str)
    tool_name = match.group(1)
    pattern = match.group(2)
    return PermissionRule(tool_name=tool_name, pattern=pattern, raw=rule_str)


@dataclass
class PermissionRuleSet:
    """Collection of allow/deny rules."""
    allow: list[PermissionRule] = field(default_factory=list)
    deny: list[PermissionRule] = field(default_factory=list)

    def check(self, tool_name: str, tool_input: dict[str, Any]) -> PermissionBehavior:
        """Check rules against a tool call.

        Deny rules take priority over allow rules.
        """
        from nature.config.constants import PermissionBehavior
        for rule in self.deny:
            if rule.matches(tool_name, tool_input):
                return PermissionBehavior.DENY
        for rule in self.allow:
            if rule.matches(tool_name, tool_input):
                return PermissionBehavior.ALLOW
        return PermissionBehavior.ASK

    @classmethod
    def from_settings(cls, settings_permissions: dict) -> PermissionRuleSet:
        """Build from settings.json permissions section.

        Expected: {"allow": ["Bash(git *)"], "deny": ["Bash(rm -rf *)"]}
        """
        allow = [parse_rule(r) for r in settings_permissions.get("allow", [])]
        deny = [parse_rule(r) for r in settings_permissions.get("deny", [])]
        return cls(allow=allow, deny=deny)
