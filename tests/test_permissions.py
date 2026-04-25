"""Tests for permission system."""

import pytest

from nature.permissions.checker import PermissionChecker
from nature.permissions.modes import PermissionMode
from nature.config.constants import PermissionBehavior as PB
from nature.permissions.rules import PermissionRule, PermissionRuleSet, parse_rule
from nature.protocols.tool import ToolContext, ToolResult


class TestParseRule:
    def test_simple_tool(self):
        rule = parse_rule("Bash")
        assert rule.tool_name == "Bash"
        assert rule.pattern is None

    def test_tool_with_pattern(self):
        rule = parse_rule("Bash(git *)")
        assert rule.tool_name == "Bash"
        assert rule.pattern == "git *"

    def test_file_pattern(self):
        rule = parse_rule("Read(*.py)")
        assert rule.tool_name == "Read"
        assert rule.pattern == "*.py"


class TestPermissionRule:
    def test_match_all(self):
        rule = PermissionRule(tool_name="Bash")
        assert rule.matches("Bash", {"command": "anything"}) is True
        assert rule.matches("Read", {"file_path": "x"}) is False

    def test_match_pattern(self):
        rule = PermissionRule(tool_name="Bash", pattern="git *")
        assert rule.matches("Bash", {"command": "git status"}) is True
        assert rule.matches("Bash", {"command": "rm -rf /"}) is False


class TestPermissionRuleSet:
    def test_deny_overrides_allow(self):
        rules = PermissionRuleSet(
            allow=[parse_rule("Bash(git *)")],
            deny=[parse_rule("Bash(git push --force)")],
        )
        assert rules.check("Bash", {"command": "git status"}) == PB.ALLOW
        assert rules.check("Bash", {"command": "git push --force"}) == PB.DENY
        assert rules.check("Bash", {"command": "rm -rf"}) == PB.ASK

    def test_from_settings(self):
        rules = PermissionRuleSet.from_settings({
            "allow": ["Bash(git *)", "Read"],
            "deny": ["Bash(rm *)"],
        })
        assert len(rules.allow) == 2
        assert len(rules.deny) == 1


class TestPermissionChecker:
    @pytest.mark.asyncio
    async def test_bypass_mode(self):
        checker = PermissionChecker(mode=PermissionMode.BYPASS)
        from nature.tools.builtin.bash import BashTool
        tool = BashTool()
        ctx = ToolContext(cwd="/tmp")
        result = await checker.check(tool, {"command": "anything"}, ctx)
        assert result.behavior == PB.ALLOW

    @pytest.mark.asyncio
    async def test_deny_mode(self):
        checker = PermissionChecker(mode=PermissionMode.DENY)
        from nature.tools.builtin.bash import BashTool
        tool = BashTool()
        ctx = ToolContext(cwd="/tmp")
        result = await checker.check(tool, {"command": "ls"}, ctx)
        assert result.behavior == PB.DENY

    @pytest.mark.asyncio
    async def test_rule_deny(self):
        rules = PermissionRuleSet(deny=[parse_rule("Bash(rm *)")])
        checker = PermissionChecker(rules=rules)
        from nature.tools.builtin.bash import BashTool
        tool = BashTool()
        ctx = ToolContext(cwd="/tmp")
        result = await checker.check(tool, {"command": "rm -rf /tmp"}, ctx)
        assert result.behavior == PB.DENY

    @pytest.mark.asyncio
    async def test_rule_allow(self):
        rules = PermissionRuleSet(allow=[parse_rule("Bash(git *)")])
        checker = PermissionChecker(rules=rules)
        from nature.tools.builtin.bash import BashTool
        tool = BashTool()
        ctx = ToolContext(cwd="/tmp")
        result = await checker.check(tool, {"command": "git status"}, ctx)
        assert result.behavior == PB.ALLOW
