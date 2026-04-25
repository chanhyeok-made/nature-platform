"""Tests for text-based tool call parsing."""

from nature.agent.text_tool_parser import extract_tool_calls_from_text


KNOWN_TOOLS = {"Bash", "Read", "Write", "Edit", "Glob", "Grep"}


class TestExtractToolCalls:
    def test_inline_json_arguments(self):
        """Parse OpenAI-style tool call with 'arguments' key."""
        text = '{"name": "Glob", "arguments": {"pattern": "**/*.py"}}'
        remaining, tools = extract_tool_calls_from_text(text, KNOWN_TOOLS)
        assert len(tools) == 1
        assert tools[0].name == "Glob"
        assert tools[0].input == {"pattern": "**/*.py"}
        assert remaining == ""

    def test_inline_json_input(self):
        """Parse Anthropic-style tool call with 'input' key."""
        text = '{"name": "Read", "input": {"file_path": "/tmp/test.py"}}'
        remaining, tools = extract_tool_calls_from_text(text, KNOWN_TOOLS)
        assert len(tools) == 1
        assert tools[0].name == "Read"
        assert tools[0].input == {"file_path": "/tmp/test.py"}

    def test_with_surrounding_text(self):
        """Parse tool call embedded in regular text."""
        text = 'Let me check that file.\n{"name": "Read", "arguments": {"file_path": "/tmp/x"}}\nHere are the results.'
        remaining, tools = extract_tool_calls_from_text(text, KNOWN_TOOLS)
        assert len(tools) == 1
        assert tools[0].name == "Read"
        assert "Let me check" in remaining

    def test_fenced_code_block(self):
        """Parse tool call in fenced code block."""
        text = 'I will search:\n```json\n{"name": "Grep", "arguments": {"pattern": "def main"}}\n```'
        remaining, tools = extract_tool_calls_from_text(text, KNOWN_TOOLS)
        assert len(tools) == 1
        assert tools[0].name == "Grep"

    def test_unknown_tool_ignored(self):
        """Ignore tool calls with unknown names."""
        text = '{"name": "UnknownTool", "arguments": {"x": 1}}'
        remaining, tools = extract_tool_calls_from_text(text, KNOWN_TOOLS)
        assert len(tools) == 0
        assert remaining == text

    def test_no_tool_call(self):
        """No tool calls in plain text."""
        text = "This is just a regular response with no JSON."
        remaining, tools = extract_tool_calls_from_text(text, KNOWN_TOOLS)
        assert len(tools) == 0
        assert remaining == text

    def test_empty_text(self):
        remaining, tools = extract_tool_calls_from_text("", KNOWN_TOOLS)
        assert tools == []

    def test_no_known_tools(self):
        text = '{"name": "Bash", "arguments": {"command": "ls"}}'
        remaining, tools = extract_tool_calls_from_text(text, set())
        assert tools == []

    def test_malformed_json(self):
        """Malformed JSON is ignored."""
        text = '{"name": "Bash", "arguments": {broken}'
        remaining, tools = extract_tool_calls_from_text(text, KNOWN_TOOLS)
        assert len(tools) == 0

    def test_nested_braces_in_edit(self):
        """Edit tool with code containing braces in old_string/new_string."""
        text = '{"name": "Edit", "arguments": {"file_path": "/tmp/test.py", "old_string": "def foo():\\n    return {}", "new_string": "def foo():\\n    return {\\\"key\\\": 1}"}}'
        remaining, tools = extract_tool_calls_from_text(text, KNOWN_TOOLS)
        assert len(tools) == 1
        assert tools[0].name == "Edit"
        assert "file_path" in tools[0].input

    def test_python_call_syntax(self):
        """Parse Python function call style: Glob(pattern="**/*")"""
        text = 'Let me search.\nGlob(pattern="**/*")'
        remaining, tools = extract_tool_calls_from_text(text, KNOWN_TOOLS)
        assert len(tools) == 1
        assert tools[0].name == "Glob"
        assert tools[0].input["pattern"] == "**/*"

    def test_python_call_multiple_args(self):
        """Parse Python call with multiple args."""
        text = 'Read(file_path="/tmp/test.py", offset=10)'
        remaining, tools = extract_tool_calls_from_text(text, KNOWN_TOOLS)
        assert len(tools) == 1
        assert tools[0].name == "Read"
        assert tools[0].input["file_path"] == "/tmp/test.py"

    def test_python_call_with_surrounding_text(self):
        text = 'I will check the files.\nGlob(pattern="**/*.py")\nLet me also look at:'
        remaining, tools = extract_tool_calls_from_text(text, KNOWN_TOOLS)
        assert len(tools) == 1
        assert "check the files" in remaining

    def test_multiple_tool_calls(self):
        """Multiple tool calls in one text."""
        text = 'First:\n{"name": "Glob", "arguments": {"pattern": "*.py"}}\nThen:\n{"name": "Read", "arguments": {"file_path": "/tmp/a.py"}}'
        remaining, tools = extract_tool_calls_from_text(text, KNOWN_TOOLS)
        assert len(tools) == 2
        assert tools[0].name == "Glob"
        assert tools[1].name == "Read"
