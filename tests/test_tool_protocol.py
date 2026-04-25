"""Tests for tool protocol."""

from nature.protocols.tool import (
    PermissionResult,
    Tool,
    ToolContext,
    ToolDefinition,
    ToolResult,
)


class TestToolDefinition:
    def test_basic_definition(self):
        td = ToolDefinition(
            name="TestTool",
            description="A test tool",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        assert td.name == "TestTool"
        assert td.deferred is False

    def test_deferred_definition(self):
        td = ToolDefinition(
            name="HeavyTool",
            description="A heavy tool",
            input_schema={},
            deferred=True,
        )
        assert td.deferred is True


class TestToolContext:
    def test_basic_context(self):
        ctx = ToolContext(cwd="/tmp/project", project_root="/tmp/project")
        assert ctx.cwd == "/tmp/project"
        assert ctx.is_read_only is False

    def test_read_only_context(self):
        ctx = ToolContext(cwd="/tmp", is_read_only=True)
        assert ctx.is_read_only is True


class TestToolResult:
    def test_success_result(self):
        result = ToolResult(output="file contents here")
        assert result.output == "file contents here"
        assert result.is_error is False

    def test_error_result(self):
        result = ToolResult(output="not found", is_error=True)
        assert result.is_error is True


class TestPermissionResult:
    def test_passthrough(self):
        pr = PermissionResult(behavior="passthrough")
        assert pr.behavior == "passthrough"
        assert pr.message is None

    def test_deny_with_reason(self):
        pr = PermissionResult(
            behavior="deny",
            message="Operation not allowed",
            reason="dangerous_command",
        )
        assert pr.behavior == "deny"
        assert pr.message == "Operation not allowed"


class TestTokenBudget:
    def test_thresholds(self):
        from nature.protocols.context import TokenBudget, TokenWarningState

        budget = TokenBudget()
        assert budget.effective_window == 180_000
        assert budget.autocompact_threshold == 167_000
        assert budget.warning_threshold == 160_000
        assert budget.block_threshold == 177_000

    def test_warning_states(self):
        from nature.protocols.context import TokenBudget, TokenWarningState

        budget = TokenBudget()
        # effective=180K, warning=160K, autocompact=167K, block=177K
        assert budget.get_warning_state(100_000) == TokenWarningState.OK
        assert budget.get_warning_state(163_000) == TokenWarningState.WARNING
        assert budget.get_warning_state(170_000) == TokenWarningState.AUTOCOMPACT
        assert budget.get_warning_state(178_000) == TokenWarningState.ERROR
