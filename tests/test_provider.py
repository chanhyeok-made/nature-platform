"""Tests for the provider layer."""

from nature.protocols.message import Message, StreamEvent, StreamEventType, Usage
from nature.protocols.provider import CacheControl, ProviderConfig
from nature.protocols.tool import ToolDefinition
from nature.providers.base import BaseLLMProvider
from nature.providers.registry import ProviderRegistry


class TestProviderConfig:
    def test_defaults(self):
        config = ProviderConfig()
        assert config.model == "claude-sonnet-4-20250514"
        assert config.max_output_tokens == 8192

    def test_custom(self):
        config = ProviderConfig(model="claude-opus-4-20250514", api_key="sk-test")
        assert config.model == "claude-opus-4-20250514"
        assert config.api_key == "sk-test"


class TestProviderRegistry:
    def test_register_and_create(self):
        registry = ProviderRegistry()

        class FakeProvider(BaseLLMProvider):
            async def stream(self, messages, system, tools=None, **kw):
                yield StreamEvent(type=StreamEventType.MESSAGE_START)

            async def count_tokens(self, messages, system, tools=None):
                return 0

        registry.register("fake", FakeProvider)
        assert "fake" in registry.available

        provider = registry.create("fake", ProviderConfig())
        assert provider.model_id == "claude-sonnet-4-20250514"

    def test_unknown_provider(self):
        registry = ProviderRegistry()
        try:
            registry.create("nonexistent", ProviderConfig())
            assert False, "Should have raised"
        except KeyError as e:
            assert "nonexistent" in str(e)


class TestBaseLLMProvider:
    def test_usage_accumulation(self):
        class FakeProvider(BaseLLMProvider):
            async def stream(self, messages, system, tools=None, **kw):
                yield StreamEvent(type=StreamEventType.MESSAGE_START)

            async def count_tokens(self, messages, system, tools=None):
                return 0

        provider = FakeProvider(ProviderConfig())
        assert provider.total_usage.total_tokens == 0

        provider._accumulate_usage(Usage(input_tokens=100, output_tokens=50))
        assert provider.total_usage.input_tokens == 100
        assert provider.total_usage.output_tokens == 50
        assert provider.total_usage.total_tokens == 150

        provider._accumulate_usage(Usage(input_tokens=200, output_tokens=100))
        assert provider.total_usage.input_tokens == 300
        assert provider.total_usage.output_tokens == 150


class TestAnthropicProviderConversion:
    """Test the conversion functions without needing the anthropic SDK."""

    def test_system_blocks_no_boundary(self):
        from nature.providers.anthropic import _build_system_blocks

        blocks = _build_system_blocks(["Hello", "World"])
        assert len(blocks) == 1
        assert blocks[0]["text"] == "Hello\n\nWorld"

    def test_system_blocks_with_boundary(self):
        from nature.config.defaults import DYNAMIC_BOUNDARY
        from nature.providers.anthropic import _build_system_blocks

        system = [f"Static part\n\n{DYNAMIC_BOUNDARY}\n\nDynamic part"]
        blocks = _build_system_blocks(system, CacheControl(type="ephemeral"))
        assert len(blocks) == 2
        assert "Static part" in blocks[0]["text"]
        assert "cache_control" in blocks[0]
        assert "Dynamic part" in blocks[1]["text"]
        assert "cache_control" not in blocks[1]

    def test_system_blocks_empty(self):
        from nature.providers.anthropic import _build_system_blocks

        assert _build_system_blocks([]) == []

    def test_message_to_api(self):
        from nature.providers.anthropic import _message_to_api

        msg = Message.user("hello")
        api_msg = _message_to_api(msg)
        assert api_msg["role"] == "user"
        assert api_msg["content"][0]["type"] == "text"
        assert api_msg["content"][0]["text"] == "hello"

    def test_tool_result_to_api(self):
        from nature.providers.anthropic import _message_to_api

        msg = Message.tool_result("toolu_123", "file contents", is_error=False)
        api_msg = _message_to_api(msg)
        block = api_msg["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "toolu_123"
        assert "is_error" not in block  # False means omitted

    def test_tool_def_to_api(self):
        from nature.providers.anthropic import _tool_def_to_api

        td = ToolDefinition(
            name="Read",
            description="Read a file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        api_td = _tool_def_to_api(td)
        assert api_td["name"] == "Read"
        assert "defer_loading" not in api_td

    def test_deferred_tool_def_to_api(self):
        from nature.providers.anthropic import _tool_def_to_api

        td = ToolDefinition(
            name="HeavyTool",
            description="Heavy",
            input_schema={},
            deferred=True,
        )
        api_td = _tool_def_to_api(td)
        assert api_td["defer_loading"] is True
