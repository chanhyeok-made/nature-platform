"""Tests for core message types."""

from nature.protocols.message import (
    Message,
    Role,
    TextContent,
    ToolResultContent,
    ToolUseContent,
    Usage,
)


class TestMessage:
    def test_user_factory(self):
        msg = Message.user("hello")
        assert msg.role == Role.USER
        assert len(msg.content) == 1
        assert msg.text == "hello"

    def test_assistant_factory(self):
        msg = Message.assistant("world")
        assert msg.role == Role.ASSISTANT
        assert msg.text == "world"

    def test_tool_result_factory(self):
        msg = Message.tool_result("toolu_abc", "result content")
        assert msg.role == Role.USER
        block = msg.content[0]
        assert isinstance(block, ToolResultContent)
        assert block.tool_use_id == "toolu_abc"
        assert block.content == "result content"
        assert block.is_error is False

    def test_tool_result_error(self):
        msg = Message.tool_result("toolu_abc", "error!", is_error=True)
        block = msg.content[0]
        assert isinstance(block, ToolResultContent)
        assert block.is_error is True

    def test_text_property(self):
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                TextContent(text="hello "),
                TextContent(text="world"),
            ],
        )
        assert msg.text == "hello world"

    def test_tool_use_blocks(self):
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                TextContent(text="Let me read that file."),
                ToolUseContent(name="Read", input={"file_path": "/tmp/test.py"}),
                ToolUseContent(name="Grep", input={"pattern": "def"}),
            ],
        )
        blocks = msg.tool_use_blocks
        assert len(blocks) == 2
        assert blocks[0].name == "Read"
        assert blocks[1].name == "Grep"
        assert msg.has_tool_use is True

    def test_no_tool_use(self):
        msg = Message.assistant("just text")
        assert msg.has_tool_use is False
        assert msg.tool_use_blocks == []

    def test_message_serialization(self):
        msg = Message.user("test")
        data = msg.model_dump()
        restored = Message.model_validate(data)
        assert restored.text == "test"


class TestUsage:
    def test_total_tokens(self):
        usage = Usage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_cache_tokens(self):
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=80,
            cache_read_input_tokens=20,
        )
        assert usage.total_tokens == 150
        assert usage.cache_creation_input_tokens == 80
