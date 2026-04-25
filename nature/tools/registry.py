"""Tool registry — discover and assemble the tool pool."""

from __future__ import annotations

from nature.protocols.tool import Tool


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    @property
    def names(self) -> list[str]:
        return sorted(self._tools.keys())


def get_default_tools() -> list[Tool]:
    """Return the default set of builtin tools.

    The Agent delegation tool is NOT included — in the Frame+Event
    architecture the AreaManager registers `nature.frame.agent_tool`
    directly when the role's `allowed_tools` list contains "Agent", so
    filesystem tools and delegation are kept on separate tracks.
    """
    from nature.tools.builtin.bash import BashTool
    from nature.tools.builtin.read import ReadTool
    from nature.tools.builtin.write import WriteTool
    from nature.tools.builtin.edit import EditTool
    from nature.tools.builtin.glob_tool import GlobTool
    from nature.tools.builtin.grep_tool import GrepTool
    from nature.tools.builtin.todowrite import TodoWriteTool
    from nature.tools.builtin.web_fetch import WebFetchTool
    from nature.tools.builtin.web_search import WebSearchTool

    return [
        BashTool(),
        ReadTool(),
        WriteTool(),
        EditTool(),
        GlobTool(),
        GrepTool(),
        TodoWriteTool(),
        WebFetchTool(),
        WebSearchTool(),
    ]
