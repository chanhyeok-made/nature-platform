"""Probe runner — drive one probe against one model end-to-end.

Runs outside nature's server / frame machinery. We talk to the
target provider directly, execute tools locally, and evaluate the
resulting trace against the probe's success criteria. That isolates
"model capability" from "nature framework orchestration" — the probe
result is a property of the model alone.

Per probe:
  1. Materialize the probe's workspace (optional) into a tempdir.
  2. Build provider for `host::model` (Anthropic or OpenAI-compat).
  3. Filter the global tool registry to the probe's allowed_tools.
  4. Run a minimal agent loop:
       assistant = stream(messages, system, tools)
       for each tool_use in assistant:
         execute locally → ToolResult
       append assistant + tool_result msgs
       repeat until stop_reason != tool_use OR max_turns
  5. Build ProbeTrace, run success.evaluate(), return.

Errors at each stage (provider instantiation, tool execution, LLM
streaming, timeout) are captured as `runner_errors` on the trace so
the evaluator can distinguish "model produced wrong output" from
"model couldn't be reached" from "call timed out."
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nature.config.hosts import (
    HostConfig,
    load_hosts_config,
    parse_model_ref,
)
from nature.probe.probes import Probe
from nature.probe.success import ProbeOutcome, ProbeTrace, ToolUseRecord, evaluate
from nature.protocols.message import (
    Message,
    Role,
    StreamEventType,
    TextContent,
    ToolResultContent,
    ToolUseContent,
)
from nature.protocols.provider import LLMProvider, ProviderConfig
from nature.protocols.tool import Tool, ToolContext

logger = logging.getLogger(__name__)


_MIN_SYSTEM_PROMPT = (
    "You are being evaluated by a probe harness. Follow the user's "
    "instructions precisely, using the provided tools when appropriate. "
    "Emit tool calls as real tool_use blocks — do not describe them "
    "in prose. When the task is complete, produce a final text "
    "response in the format the user requested (if any)."
)


@dataclass
class ProbeRunResult:
    """Everything the caller needs for reporting one probe run."""

    probe_id: str
    model_ref: str
    outcome: ProbeOutcome
    trace: ProbeTrace
    latency_ms: int
    tokens_in: int = 0
    tokens_out: int = 0
    runner_errors: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# Provider resolution
# ──────────────────────────────────────────────────────────────────────


def _build_provider(
    model_ref: str, *, project_dir: Path | None = None,
) -> tuple[LLMProvider, str, HostConfig]:
    """Parse `host::model` (or bare `model` → default host), load the
    host config layered builtin→user→project, and construct the
    provider. Returns (provider, bare_model_name, host_config)."""
    host_name, bare = parse_model_ref(model_ref)
    hosts = load_hosts_config(project_dir=project_dir)
    host_name = host_name or hosts.default_host
    if host_name not in hosts.hosts:
        raise ValueError(
            f"unknown host {host_name!r} in ref {model_ref!r}. "
            f"Known: {sorted(hosts.hosts)}"
        )
    host = hosts.hosts[host_name]
    api_key = host.resolved_api_key() or ""
    cfg = ProviderConfig(
        model=bare, api_key=api_key, base_url=host.base_url,
        host_name=host_name,
    )
    if host.provider == "anthropic":
        from nature.providers.anthropic import AnthropicProvider
        inner: LLMProvider = AnthropicProvider(cfg)
    else:
        from nature.providers.openai_compat import OpenAICompatProvider
        inner = OpenAICompatProvider(cfg)

    # Wrap in text-tool adapter for models whose Ollama backend
    # doesn't accept `tools=` or whose coder tune emits tool calls
    # as text (`{"name":"Read",...}`). The capability decision is
    # centralized in model_capabilities so the probe runner, agent
    # loop, and server registry all use the same table.
    from nature.providers.model_capabilities import lookup as _lookup_caps
    caps = _lookup_caps(model_ref)
    if caps.text_tool_adaptation:
        from nature.providers.text_tool_wrapper import TextToolAdapterProvider
        inner = TextToolAdapterProvider(inner)
    return inner, bare, host


# ──────────────────────────────────────────────────────────────────────
# Tool resolution (restricted to the probe's allowed_tools)
# ──────────────────────────────────────────────────────────────────────


def _build_tools(allowed: list[str]) -> dict[str, Tool]:
    """Load nature's builtin tools + Agent delegation tool, then
    filter to just the names the probe allows. Unknown names raise."""
    from nature.tools.registry import get_default_tools
    from nature.frame.agent_tool import AgentTool

    pool: dict[str, Tool] = {t.name: t for t in get_default_tools()}
    pool["Agent"] = AgentTool()

    result: dict[str, Tool] = {}
    unknown: list[str] = []
    for name in allowed:
        if name in pool:
            result[name] = pool[name]
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(
            f"probe allows unknown tools {unknown!r}. "
            f"Available: {sorted(pool)}"
        )
    return result


# ──────────────────────────────────────────────────────────────────────
# Workspace materialization
# ──────────────────────────────────────────────────────────────────────


def _materialize_workspace(probe: Probe, root: Path) -> None:
    """Lay down probe.workspace.files under `root`. Nested dirs are
    created as needed; existing content at the same path is
    overwritten so repeated probe runs are deterministic."""
    if probe.workspace is None:
        return
    for f in probe.workspace.files:
        dest = root / f.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f.content, encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# Streaming collector — pull a full assistant message from a stream
# ──────────────────────────────────────────────────────────────────────


async def _collect_stream(
    provider: LLMProvider,
    *,
    model: str,
    messages: list[Message],
    system: list[str],
    tools: list,
    timeout_sec: float,
) -> tuple[list, int, int]:
    """Run one stream, return (content_blocks, tokens_in, tokens_out).

    Content blocks are the full TextContent / ToolUseContent /
    ThinkingContent list that comprises the assistant message. The
    streaming events are accumulated into the blocks via their
    `content_block_start` + `content_block_delta` + `content_block_stop`
    markers — this mirrors what nature's agent loop does, without
    the state-transition event emission side effect.
    """
    collected: list = []
    current_json_buf = ""
    tokens_in = 0
    tokens_out = 0

    async def _run() -> None:
        nonlocal current_json_buf, tokens_in, tokens_out
        import json as _json

        async for ev in provider.stream(
            messages=messages,
            system=system,
            tools=tools,
            model=model,
        ):
            if ev.type == StreamEventType.CONTENT_BLOCK_START:
                if ev.content_block is not None:
                    collected.append(ev.content_block.model_copy(deep=True))
                    current_json_buf = ""
            elif ev.type == StreamEventType.CONTENT_BLOCK_DELTA:
                # OpenAICompatProvider emits CONTENT_BLOCK_DELTA for
                # text *without* a prior CONTENT_BLOCK_START (unlike
                # Anthropic). Auto-open a TextContent block so we
                # don't drop text from Ollama/OpenAI-shape providers.
                if ev.delta_text is not None:
                    if not collected or not isinstance(collected[-1], TextContent):
                        collected.append(TextContent(text=""))
                    collected[-1].text += ev.delta_text
                elif ev.delta_tool_input is not None and collected and isinstance(
                    collected[-1], ToolUseContent
                ):
                    current_json_buf += ev.delta_tool_input
                    try:
                        collected[-1].input = (
                            _json.loads(current_json_buf) if current_json_buf else {}
                        )
                    except _json.JSONDecodeError:
                        pass  # partial — finalized at block_stop
            elif ev.type == StreamEventType.CONTENT_BLOCK_STOP:
                # Only finalize input from the buffered deltas if we
                # actually saw any. A synthetic tool_use block from
                # the TextToolAdapterProvider already carries its
                # input directly on the START event — overwriting
                # with an empty buffer would clobber that input back
                # to `{}` and cause a Pydantic validation failure
                # when the tool runs.
                if (
                    collected
                    and isinstance(collected[-1], ToolUseContent)
                    and current_json_buf
                ):
                    try:
                        collected[-1].input = _json.loads(current_json_buf)
                    except _json.JSONDecodeError:
                        collected[-1].input = {"_raw": current_json_buf}
                current_json_buf = ""
            elif ev.type in (StreamEventType.MESSAGE_STOP, StreamEventType.MESSAGE_DELTA):
                if ev.usage is not None:
                    # Anthropic reports cumulative usage on MESSAGE_STOP;
                    # pick the last one we see.
                    tokens_in = (
                        (ev.usage.input_tokens or 0)
                        + (ev.usage.cache_read_input_tokens or 0)
                        + (ev.usage.cache_creation_input_tokens or 0)
                    )
                    tokens_out = ev.usage.output_tokens or 0

    try:
        await asyncio.wait_for(_run(), timeout=timeout_sec)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"LLM stream exceeded {timeout_sec}s") from exc

    return collected, tokens_in, tokens_out


# ──────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────


async def _run_probe_async(
    probe: Probe,
    model_ref: str,
    *,
    project_dir: Path | None = None,
) -> ProbeRunResult:
    runner_errors: list[str] = []
    started = time.time()
    tokens_in_total = 0
    tokens_out_total = 0
    tool_uses: list[ToolUseRecord] = []
    final_text = ""
    turn_count = 0
    hit_max_turns = False

    # Provider
    try:
        provider, bare_model, _host = _build_provider(model_ref, project_dir=project_dir)
    except Exception as exc:  # noqa: BLE001
        return ProbeRunResult(
            probe_id=probe.id,
            model_ref=model_ref,
            outcome=ProbeOutcome(
                passed=False, criteria=[], fail_category="runner:provider_error",
            ),
            trace=ProbeTrace(
                tool_uses=[], final_text="", turn_count=0, hit_max_turns=False,
                runner_errors=(f"provider_build: {exc}",),
            ),
            latency_ms=int((time.time() - started) * 1000),
            runner_errors=[f"provider_build: {exc}"],
        )

    # Per-model timeout scale. 70B / reasoning models need extra head-
    # room that generic probes don't declare; the capabilities table
    # supplies the multiplier so probes stay model-agnostic.
    from nature.providers import model_capabilities as _mc
    _caps = _mc.lookup(model_ref)
    _timeout = probe.timeout_sec * max(_caps.stream_timeout_multiplier, 1.0)

    # Tools
    try:
        tools_by_name = _build_tools(probe.allowed_tools)
    except Exception as exc:  # noqa: BLE001
        return ProbeRunResult(
            probe_id=probe.id,
            model_ref=model_ref,
            outcome=ProbeOutcome(
                passed=False, criteria=[], fail_category="runner:tool_error",
            ),
            trace=ProbeTrace(
                tool_uses=[], final_text="", turn_count=0, hit_max_turns=False,
                runner_errors=(f"tool_build: {exc}",),
            ),
            latency_ms=int((time.time() - started) * 1000),
            runner_errors=[f"tool_build: {exc}"],
        )

    tool_defs = [t.to_definition() for t in tools_by_name.values()]

    # Workspace
    ws_root = Path(tempfile.mkdtemp(prefix=f"nature-probe-{probe.id}-"))
    try:
        _materialize_workspace(probe, ws_root)

        # Interpolate `{workspace}` in the probe's prompt so the
        # model receives the actual tempdir path. Without this,
        # models guess paths (/hello.txt, /workspace/, ...) and the
        # Read/Edit tools fail because they expect absolute paths.
        # The substitution is on the prompt only; `probe.prompt` on
        # disk stays the un-interpolated template.
        prompt_text = probe.prompt.replace("{workspace}", str(ws_root))
        system_prompt = (probe.system or _MIN_SYSTEM_PROMPT).replace(
            "{workspace}", str(ws_root)
        )
        messages: list[Message] = [
            Message(role=Role.USER, content=[TextContent(text=prompt_text)]),
        ]

        tool_ctx = ToolContext(
            cwd=str(ws_root),
            project_root=str(ws_root),
            session_id=f"probe-{probe.id}-{uuid.uuid4().hex[:8]}",
            agent_id="probe-solo",
            is_read_only=False,
            additional_directories=[],
            pack_state={},
        )

        # Agent loop
        while True:
            if turn_count >= probe.max_turns:
                hit_max_turns = True
                break
            turn_count += 1
            try:
                blocks, tin, tout = await _collect_stream(
                    provider,
                    model=bare_model,
                    messages=messages,
                    system=[system_prompt],
                    tools=tool_defs,
                    timeout_sec=_timeout,
                )
            except TimeoutError as exc:
                runner_errors.append(f"llm_timeout: {exc}")
                break
            except Exception as exc:  # noqa: BLE001
                runner_errors.append(f"llm_error: {type(exc).__name__}: {exc}")
                break

            tokens_in_total += tin
            tokens_out_total += tout

            # Append assistant message
            messages.append(Message(role=Role.ASSISTANT, content=blocks))

            # Collect text
            text_parts = [b.text for b in blocks if isinstance(b, TextContent)]
            if text_parts:
                final_text = "\n".join(text_parts)

            # Execute tool_uses (if any).
            tool_use_blocks = [b for b in blocks if isinstance(b, ToolUseContent)]
            if not tool_use_blocks:
                break  # done — no tools requested

            tool_result_blocks: list[ToolResultContent] = []
            for tu in tool_use_blocks:
                idx = len(tool_uses)
                record = ToolUseRecord(
                    index=idx, name=tu.name, input=dict(tu.input or {}),
                )
                tool = tools_by_name.get(tu.name)
                if tool is None:
                    record.result_text = (
                        f"ERR: tool {tu.name!r} not allowed in this probe"
                    )
                    record.result_is_error = True
                else:
                    try:
                        tool_res = await tool.execute(tu.input or {}, tool_ctx)
                        record.result_text = str(tool_res.output)[:4000]
                        record.result_is_error = bool(tool_res.is_error)
                    except Exception as exc:  # noqa: BLE001
                        record.result_text = f"ERR: {type(exc).__name__}: {exc}"
                        record.result_is_error = True
                tool_uses.append(record)
                tool_result_blocks.append(ToolResultContent(
                    tool_use_id=tu.id,
                    content=record.result_text or "",
                    is_error=record.result_is_error,
                ))

            # Append tool_result message
            messages.append(Message(role=Role.USER, content=list(tool_result_blocks)))

        trace = ProbeTrace(
            tool_uses=tool_uses,
            final_text=final_text,
            turn_count=turn_count,
            hit_max_turns=hit_max_turns,
            workspace_root=ws_root,
            runner_errors=tuple(runner_errors),
        )
        outcome = evaluate(probe, trace)

        return ProbeRunResult(
            probe_id=probe.id,
            model_ref=model_ref,
            outcome=outcome,
            trace=trace,
            latency_ms=int((time.time() - started) * 1000),
            tokens_in=tokens_in_total,
            tokens_out=tokens_out_total,
            runner_errors=list(runner_errors),
        )
    finally:
        # Probes are stateless — workspace is disposable.
        shutil.rmtree(ws_root, ignore_errors=True)


def run_probe(
    probe: Probe,
    model_ref: str,
    *,
    project_dir: Path | None = None,
) -> ProbeRunResult:
    """Sync wrapper around `_run_probe_async` for CLI use."""
    return asyncio.run(_run_probe_async(probe, model_ref, project_dir=project_dir))


__all__ = ["ProbeRunResult", "run_probe"]
