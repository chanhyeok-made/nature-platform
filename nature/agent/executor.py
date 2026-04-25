"""Streaming tool executor — concurrent tool dispatch with safety partitioning.

Tools marked as concurrency-safe run in parallel via asyncio.gather.
Unsafe tools run sequentially. This matches Claude Code's
StreamingToolExecutor pattern.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from nature.protocols.message import Message, ToolUseContent
from nature.protocols.tool import Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

MAX_TOOL_CONCURRENCY = 10


async def _execute_single_tool(
    tool: Tool,
    tool_use: ToolUseContent,
    context: ToolContext,
) -> Message:
    """Execute a single tool and return a tool_result message."""
    try:
        # Validate input
        validation_error = await tool.validate_input(tool_use.input, context)
        if validation_error:
            return Message.tool_result(tool_use.id, validation_error, is_error=True)

        # Execute
        result = await tool.execute(tool_use.input, context)

        # Format output
        output = result.output
        if not isinstance(output, str):
            import json
            output = json.dumps(output, ensure_ascii=False, default=str)

        # Truncate if too large
        max_size = tool.max_result_size_chars
        if len(output) > max_size:
            output = output[:max_size] + f"\n\n[Output truncated at {max_size:,} chars]"

        return Message.tool_result(tool_use.id, output, is_error=result.is_error)

    except Exception as e:
        logger.error("Tool %s failed: %s", tool.name, e)
        return Message.tool_result(tool_use.id, f"Error: {e}", is_error=True)


async def execute_tools(
    tool_uses: list[ToolUseContent],
    tools: list[Tool],
    context: ToolContext,
) -> list[Message]:
    """Execute tool calls with concurrency partitioning.

    Partitions tool calls into batches:
    - Consecutive concurrency-safe tools run in parallel
    - Non-safe tools run alone (sequentially)

    Returns tool_result messages in the same order as tool_uses.
    """
    if not tool_uses:
        return []

    tool_map = {t.name: t for t in tools}
    results: list[Message] = []

    # Partition into batches
    batches: list[list[tuple[ToolUseContent, Tool]]] = []
    current_batch: list[tuple[ToolUseContent, Tool]] = []
    current_batch_safe = True

    for tu in tool_uses:
        tool = tool_map.get(tu.name)
        if tool is None:
            # Unknown tool — return error immediately
            results.append(Message.tool_result(
                tu.id,
                f"Unknown tool: {tu.name}",
                is_error=True,
            ))
            continue

        is_safe = tool.is_concurrency_safe(tu.input)

        if not current_batch:
            current_batch.append((tu, tool))
            current_batch_safe = is_safe
        elif is_safe and current_batch_safe:
            current_batch.append((tu, tool))
        else:
            batches.append(current_batch)
            current_batch = [(tu, tool)]
            current_batch_safe = is_safe

    if current_batch:
        batches.append(current_batch)

    # Execute batches
    for batch in batches:
        if len(batch) == 1:
            tu, tool = batch[0]
            result = await _execute_single_tool(tool, tu, context)
            results.append(result)
        else:
            # Parallel execution (capped concurrency)
            sem = asyncio.Semaphore(MAX_TOOL_CONCURRENCY)

            async def _run(tu: ToolUseContent, tool: Tool) -> Message:
                async with sem:
                    return await _execute_single_tool(tool, tu, context)

            batch_results = await asyncio.gather(
                *[_run(tu, tool) for tu, tool in batch]
            )
            results.extend(batch_results)

    return results
