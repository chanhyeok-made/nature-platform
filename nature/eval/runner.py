"""Eval cell runner — drive one (task × preset) pair end to end.

Per cell the runner:

1. Prepares a workspace (git worktree / git clone / patch apply),
   following the `setup.type` discriminator.
2. Starts a dedicated `nature server` on a free port, with
   `NATURE_HOME` redirected to a temp dir so event logs don't
   pollute the user's main store.
3. Creates a session with the named preset and pushes the task's
   prompt (plus a standard "do not run pytest yourself" preamble —
   the harness verifies).
4. Polls the session until it resolves / times out / errors.
5. Tallies cost + token totals from the session's jsonl event log.
6. Runs the acceptance evaluator (pytest for Phase 1).
7. Tears down: SIGTERM server, drop worktree/clone, rm temp home.
   The session's event log is copied into both
   `results/logs/<tag>.jsonl` and the user's main `~/.nature/events/`
   (with a sidecar tagging its eval origin) so every cell remains
   browsable in the dashboard afterwards.

Cells are independent — same cell id run twice produces two
distinct rows in the results file.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import threading
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from nature.eval.tasks import (
    AcceptanceLLMJudge,
    AcceptancePytest,
    SetupApplyDiffFromCommit,
    SetupApplyPatchFile,
    SetupGitCloneAtRef,
    SetupNone,
    Task,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Result dataclass (the shape stored to results.json)
# ──────────────────────────────────────────────────────────────────────


@dataclass
class CellResult:
    """One (task × preset) cell outcome.

    Phase 2a expanded the metric set — earlier runs stored only the
    four core numbers (cost, tokens_in/out, latency). Anything added
    here surfaces as `None` on older run files, so the diff/report
    code treats missing fields as "not recorded" rather than 0.
    """

    task_id: str
    preset: str
    started_at: float
    # Seed index within a multi-seed run (0-based). `None` on older
    # run files that predate multi-seed support — treated as "single
    # execution, seed unknown" by aggregation code.
    seed: int | None = None
    finished_at: float | None = None
    passed: bool = False
    latency_sec: float | None = None
    cost_usd: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    # New in Phase 2a ------------------------------------------------
    cache_read_tokens: int | None = None
    turn_count: int | None = None          # llm.response count
    tool_call_count: int | None = None     # tool.completed count
    sub_frame_count: int | None = None     # frame.opened with parent
    max_delegation_depth: int | None = None
    agents_used: list[str] | None = None   # distinct role names seen
    # New in M3.6 (all event-derivable — re-extractable via
    # `nature eval rebuild-metrics`): --------------------------------
    body_compactions: int | None = None        # body.compacted count
    cost_by_agent: dict[str, float] | None = None  # role → attributed $
    provider_errors: int | None = None         # llm.error count
    tool_error_count: int | None = None        # tool.completed is_error
    cache_hit_rate: float | None = None        # cache_read / total_in
    avg_turn_latency_ms: float | None = None   # mean response duration
    # Fork lineage (from SESSION_STARTED payload). Both None for
    # organically created sessions.
    source_session_id: str | None = None
    forked_from_event_id: int | None = None
    # M4 quality axes (test-time only — workspace is torn down, not
    # re-extractable from the event log). --------------------------
    # Regression check: pytest on the acceptance test file(s) BEFORE
    # and AFTER the agent ran. `regression_count` = tests passing at
    # baseline that now fail. Captures "did the fix break other
    # tests in the same file."
    pre_suite_pass_count: int | None = None
    post_suite_pass_count: int | None = None
    regression_count: int | None = None
    # Diff stats — `git diff --numstat HEAD` after agent edits.
    # Minimality signal ("3-line fix done in 300 lines").
    diff_lines_added: int | None = None
    diff_lines_removed: int | None = None
    diff_files_touched: int | None = None
    # Reference-fix similarity — only populated when the task's
    # `apply_commit` IS the upstream fix (n1/s1-4/x1-3). For n2/n3
    # the apply_commit is the spec-test commit, so these stay None
    # (a dedicated `reference_commit` field can land later when we
    # need ref similarity for those). Both ratios in [0, 1].
    ref_files_overlap: float | None = None     # Jaccard over file sets
    ref_line_ratio: float | None = None        # agent_lines / ref_lines
    # ----------------------------------------------------------------
    pytest_exit: int | None = None
    pytest_tail: str | None = None
    # Populated only for llm_judge acceptance cells.
    judge_reason: str | None = None
    judge_raw: str | None = None       # raw final text from judge (truncated)
    judge_session_id: str | None = None
    session_id: str | None = None
    event_log_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


@dataclass
class SessionMetrics:
    """Everything extracted from one session's jsonl event log.

    Returned by `_extract_metrics` so the runner can copy the fields
    into its `CellResult` in one place. Every field here is derivable
    from the event stream — the event log is the source of truth and
    this struct is a cache. Adding a new metric here + extending
    `_extract_metrics` is enough to backfill historical runs via
    `nature eval rebuild-metrics`.
    """

    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_tokens: int = 0
    turn_count: int = 0
    tool_call_count: int = 0
    sub_frame_count: int = 0
    max_delegation_depth: int = 0
    agents_used: list[str] = field(default_factory=list)
    # Tier-1 extensions (2026-04-20). All derivable from events.
    body_compactions: int = 0
    cost_by_agent: dict[str, float] = field(default_factory=dict)
    provider_errors: int = 0
    tool_error_count: int = 0
    avg_turn_latency_ms: float = 0.0
    # Fork lineage (from SESSION_STARTED.payload, or None for non-fork
    # sessions). Duplicated into CellResult so aux-experiment analysis
    # doesn't need to consult the store sidecar.
    source_session_id: str | None = None
    forked_from_event_id: int | None = None


# ──────────────────────────────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────────────────────────────


_PORT_ALLOCATION_LOCK = threading.Lock()
_ALLOCATED_PORTS: set[int] = set()


def _port_is_free(port: int) -> bool:
    """Check whether `port` is free on both IPv4 and IPv6 loopback.

    The server binds via `asyncio.start_server` which, on most
    platforms, registers both address families — checking only IPv4
    misses the `::1` conflict that actually hits at bind time.
    """
    with socket.socket(socket.AF_INET6) as s6:
        try:
            s6.bind(("::1", port))
        except OSError:
            return False
    with socket.socket() as s4:
        try:
            s4.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _free_port() -> int:
    """Return a port P such that P and P+1 are both free *and* neither
    overlaps another in-flight allocation from this process.

    The server binds HTTP on `NATURE_SERVER_PORT` and WebSocket on
    `port + 1` (see `nature.server.app`), so cells in a
    ThreadPoolExecutor must reserve a 2-port window — otherwise a
    neighbour cell that picked `P-1` reserves `[P-1, P]` which
    collides with our `[P, P+1]` at `P`.

    The module-level `_ALLOCATED_PORTS` set tracks the per-pair
    endpoints currently in flight so the window-check under the lock
    covers both system state and sibling allocations. Entries are
    removed in `run_cell`'s teardown.
    """
    with _PORT_ALLOCATION_LOCK:
        for _ in range(64):
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                candidate = s.getsockname()[1]
            # The WS port is `candidate + 1`; if that overflows the
            # 16-bit range, `socket.bind` raises OverflowError rather
            # than a recoverable OSError — so reject upfront and let
            # the OS pick a different port.
            if candidate + 1 > 65535:
                continue
            window = {candidate, candidate + 1}
            if window & _ALLOCATED_PORTS:
                continue
            if not _port_is_free(candidate):
                continue
            if not _port_is_free(candidate + 1):
                continue
            _ALLOCATED_PORTS.update(window)
            return candidate
        raise RuntimeError(
            "could not find a 2-port window after 64 attempts — "
            "system ephemeral port range may be exhausted",
        )


def _release_port(port: int) -> None:
    """Release a port window reserved by `_free_port`."""
    with _PORT_ALLOCATION_LOCK:
        _ALLOCATED_PORTS.discard(port)
        _ALLOCATED_PORTS.discard(port + 1)


def _http_json(
    method: str, url: str, body: dict | None = None, timeout: float = 30,
) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = Request(
        url,
        data=data,
        method=method,
        headers={"content-type": "application/json"} if data else {},
    )
    with urlopen(req, timeout=timeout) as r:
        raw = r.read() or b"{}"
        return json.loads(raw) if raw else {}


def _wait_for_server(port: int, timeout: float = 60) -> None:
    """Poll the server until it responds or `timeout` seconds elapse.

    The default is 60 s to tolerate parallel cell startup: with
    `--concurrency N`, N server subprocesses launch nearly together
    and compete for CPU during their Python import graphs — 25 s was
    tight under N=8 on development hardware."""
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            _http_json("GET", f"http://127.0.0.1:{port}/api/sessions", timeout=2)
            return
        except (URLError, HTTPError, ConnectionError, OSError) as exc:
            last_err = exc
            time.sleep(0.3)
    raise RuntimeError(f"server on :{port} never responded (last={last_err})")


def _wait_for_done(port: int, sid: str, timeout_sec: float) -> dict:
    deadline = time.time() + timeout_sec
    last: dict = {}
    while time.time() < deadline:
        try:
            last = _http_json(
                "GET", f"http://127.0.0.1:{port}/api/sessions/{sid}", timeout=5,
            )
        except (URLError, HTTPError, OSError):
            time.sleep(2.0)
            continue
        if not last.get("has_active_run") and last.get("state") in (
            "awaiting_user", "resolved", "error", "closed",
        ):
            return last
        time.sleep(2.0)
    raise TimeoutError(
        f"session {sid} stuck (state={last.get('state')!r}, "
        f"has_active_run={last.get('has_active_run')!r}) after {timeout_sec}s"
    )


# ──────────────────────────────────────────────────────────────────────
# Git helpers
# ──────────────────────────────────────────────────────────────────────


def _git(repo: Path, *args: str, check: bool = True, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
        input=input_text,
    )


def _apply_commit_diff(
    workspace: Path, source_repo: Path, commit: str, diff_scope: list[str],
) -> None:
    """Dump the diff of `commit` restricted to `diff_scope` and `git
    apply` it inside `workspace`. The diff is read from `source_repo`
    (which may be the workspace itself when repo=local; for git-clone
    targets it's the clone directory)."""
    show = _git(source_repo, "show", commit, "--", *diff_scope)
    if not show.stdout.strip():
        raise RuntimeError(
            f"commit {commit} has no diff for files {diff_scope}"
        )
    _git(workspace, "apply", "--", input_text=show.stdout)


# ──────────────────────────────────────────────────────────────────────
# Setup strategies — prepare the workspace
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _WorkspaceHandle:
    """Ephemeral workspace state returned by the setup stage."""

    path: Path
    cleanup: list = field(default_factory=list)  # callables run in teardown


def _plant_project_config(repo_root: Path, workspace: Path) -> None:
    """Seed the workspace with a copy of the main project's `.nature/`
    directory (presets, agents, hosts, packs, …) so eval cells can
    resolve presets and custom configs that aren't committed to the
    checked-out ref.

    Skipped when the workspace already ships its own `.nature/` (e.g.
    an external repo that bundles its own presets). Skipped silently
    if the main project has no `.nature/`.
    """
    src = repo_root / ".nature"
    if not src.is_dir():
        return
    dst = workspace / ".nature"
    if dst.exists():
        # Workspace already carries a `.nature/`; don't clobber.
        return
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
        "eval", "events", "server.pid",
    ))


def _commit_seed_baseline(workspace: Path) -> None:
    """Fold every file the setup stage planted (golden test diff,
    patch, `.nature/` config copy) into a single HEAD commit so the
    agent's changes are the only delta visible to `git diff HEAD`.

    Without this step, newly-created seed files stay untracked;
    the llm_judge path — which feeds `git diff HEAD` to the judge —
    would see an empty diff and misjudge the agent's work. We
    intentionally use a throwaway committer identity and skip hooks
    so the repo config is untouched; the commit lives only inside
    the ephemeral worktree/clone.
    """
    # `-A` picks up new files, modifications, and deletions alike.
    _git(workspace, "add", "-A")
    proc = subprocess.run(
        [
            "git", "-C", str(workspace),
            "-c", "user.email=eval@nature",
            "-c", "user.name=nature-eval",
            "-c", "commit.gpgsign=false",
            "commit", "--quiet", "--allow-empty", "--no-verify",
            "-m", "eval: seed baseline",
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        # Commit failed (e.g., identity issues in a constrained env).
        # Surface the stderr so the failure mode is visible rather
        # than silently handing the judge an empty diff.
        raise RuntimeError(
            f"failed to commit eval seed baseline: {proc.stderr.strip()}"
        )


def _setup_workspace(task: Task, repo_root: Path) -> _WorkspaceHandle:
    """Dispatch on `task.setup.type` to prepare the workspace and
    return a `_WorkspaceHandle` the caller tears down after the cell."""
    setup = task.setup
    target = task.target
    ws_base = Path(tempfile.gettempdir())
    # Short random suffix guarantees uniqueness when `--concurrency > 1`
    # starts multiple cells in the same second — the wall-clock `int`
    # alone collides for `(same task × different preset)` pairs.
    tag = f"nature-eval-{task.id}-{int(time.time())}-{uuid.uuid4().hex[:8]}"

    if target.repo == "local":
        workspace = ws_base / tag
        _git(repo_root, "worktree", "add", "--detach", str(workspace), target.ref)

        def _cleanup_worktree() -> None:
            subprocess.run(
                ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(workspace)],
                capture_output=True,
            )

        handle = _WorkspaceHandle(path=workspace, cleanup=[_cleanup_worktree])

        if isinstance(setup, SetupApplyDiffFromCommit):
            _apply_commit_diff(
                workspace, source_repo=repo_root,
                commit=setup.apply_commit, diff_scope=setup.diff_scope,
            )
        elif isinstance(setup, SetupApplyPatchFile):
            patch_path = (task.source_dir / setup.patch_path).resolve()
            if not patch_path.exists():
                raise FileNotFoundError(
                    f"patch {patch_path} referenced by task {task.id} does not exist",
                )
            _git(workspace, "apply", "--", input_text=patch_path.read_text(encoding="utf-8"))
        elif isinstance(setup, SetupNone):
            pass
        else:
            raise ValueError(
                f"task {task.id}: local target does not support setup {setup.type!r}",
            )
        _plant_project_config(repo_root, workspace)
        _commit_seed_baseline(workspace)
        return handle

    # Remote git repo → clone + checkout.
    workspace = ws_base / tag
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-tags", target.repo, str(workspace)],
        check=True, capture_output=True, text=True,
    )
    _git(workspace, "checkout", "--detach", target.ref)

    def _cleanup_clone() -> None:
        shutil.rmtree(workspace, ignore_errors=True)

    handle = _WorkspaceHandle(path=workspace, cleanup=[_cleanup_clone])

    if isinstance(setup, SetupGitCloneAtRef):
        if setup.apply_commit:
            _apply_commit_diff(
                workspace, source_repo=workspace,
                commit=setup.apply_commit, diff_scope=setup.diff_scope,
            )
    elif isinstance(setup, SetupApplyPatchFile):
        patch_path = (task.source_dir / setup.patch_path).resolve()
        _git(workspace, "apply", "--", input_text=patch_path.read_text(encoding="utf-8"))
    elif isinstance(setup, SetupNone):
        pass
    else:
        raise ValueError(
            f"task {task.id}: remote target does not support setup {setup.type!r}",
        )
    _plant_project_config(repo_root, workspace)
    _commit_seed_baseline(workspace)
    return handle


# ──────────────────────────────────────────────────────────────────────
# Cost accounting — same algorithm as the legacy run_one.py
# ──────────────────────────────────────────────────────────────────────


def _extract_metrics(
    session_jsonl: Path, repo_root: Path,
) -> SessionMetrics:
    """Walk the session event log once and compute every metric the
    runner reports.

    The event-log `input_tokens` field is the non-cache-input count,
    so we compute cost directly instead of going through
    `nature.utils.cost.calculate_cost` (which expects input_tokens
    to be the pre-subtraction total). Frame depth is a reconstruction
    walk over each `frame.opened` event's `parent_id`.
    """
    metrics = SessionMetrics()
    if not session_jsonl.exists():
        return metrics

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from nature.utils.cost import _get_pricing  # noqa: WPS433

    model_by_req: dict[str, str] = {}
    parent_of: dict[str, str | None] = {}
    agents_seen: list[str] = []
    agents_set: set[str] = set()
    # frame_id → role_name for cost-per-agent attribution.
    role_by_frame: dict[str, str] = {}
    # Accumulator for average turn latency (annotation.stored carries
    # duration_ms per response).
    turn_latencies_ms: list[float] = []

    def _depth(frame_id: str) -> int:
        """Chain length from `frame_id` back to a root (parent_id=None).
        Root frames return 0; a child of the root returns 1."""
        depth = 0
        cur: str | None = frame_id
        seen: set[str] = set()
        while cur is not None and cur in parent_of and cur not in seen:
            seen.add(cur)
            parent = parent_of.get(cur)
            if parent is None:
                break
            depth += 1
            cur = parent
        return depth

    for line in session_jsonl.read_text(encoding="utf-8").splitlines():
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = evt.get("type")
        p = evt.get("payload") or {}

        if t == "session.started":
            metrics.source_session_id = p.get("parent_session_id")
            metrics.forked_from_event_id = p.get("forked_from_event_id")

        elif t == "frame.opened":
            fid = evt.get("frame_id")
            parent_id = p.get("parent_id")
            role_name = p.get("role_name")
            if fid:
                parent_of[fid] = parent_id
                if role_name:
                    role_by_frame[fid] = role_name
                if parent_id is not None:
                    metrics.sub_frame_count += 1
                    metrics.max_delegation_depth = max(
                        metrics.max_delegation_depth, _depth(fid),
                    )
            if role_name and role_name not in agents_set:
                agents_set.add(role_name)
                agents_seen.append(role_name)

        elif t == "tool.completed":
            metrics.tool_call_count += 1
            if p.get("is_error"):
                metrics.tool_error_count += 1

        elif t == "llm.request":
            model_by_req[p.get("request_id", "")] = p.get("model", "")

        elif t == "llm.response":
            metrics.turn_count += 1
            rid = p.get("request_id", "")
            model = model_by_req.get(rid, "")
            u = p.get("usage") or {}
            regular_in = int(u.get("input_tokens", 0) or 0)
            out = int(u.get("output_tokens", 0) or 0)
            cache_write = int(u.get("cache_creation_input_tokens", 0) or 0)
            cache_read = int(u.get("cache_read_input_tokens", 0) or 0)
            pricing = _get_pricing(model)
            call_cost = (
                (regular_in / 1_000_000) * pricing["input"]
                + (out / 1_000_000) * pricing["output"]
                + (cache_write / 1_000_000) * pricing["cache_write"]
                + (cache_read / 1_000_000) * pricing["cache_read"]
            )
            metrics.cost_usd += call_cost
            metrics.tokens_in += regular_in
            metrics.tokens_out += out
            metrics.cache_read_tokens += cache_read
            # Attribute this call's cost to the role that owns the
            # frame the llm.response landed in. Sub-agent delegations
            # open their own frames, so this split follows the frame
            # tree regardless of how deep the delegation chain is.
            fid = evt.get("frame_id") or ""
            role = role_by_frame.get(fid, "unknown")
            metrics.cost_by_agent[role] = (
                metrics.cost_by_agent.get(role, 0.0) + call_cost
            )

        elif t == "llm.error":
            metrics.provider_errors += 1

        elif t == "body.compacted":
            metrics.body_compactions += 1

        elif t == "annotation.stored":
            dur = p.get("duration_ms")
            if dur is not None:
                try:
                    turn_latencies_ms.append(float(dur))
                except (TypeError, ValueError):
                    pass

    metrics.agents_used = agents_seen  # order = first-seen
    if turn_latencies_ms:
        metrics.avg_turn_latency_ms = (
            sum(turn_latencies_ms) / len(turn_latencies_ms)
        )
    return metrics


def _copy_metrics_onto_cell(cell: "CellResult", m: SessionMetrics) -> None:
    """Stamp the extracted SessionMetrics onto a CellResult.

    Keeps the CellResult a thin cache of what the event log says —
    `rebuild-metrics` re-runs the extractor and this copy, so every
    field here must come exclusively from `m`.
    """
    cell.cost_usd = round(m.cost_usd, 6)
    cell.tokens_in = m.tokens_in
    cell.tokens_out = m.tokens_out
    cell.cache_read_tokens = m.cache_read_tokens
    cell.turn_count = m.turn_count
    cell.tool_call_count = m.tool_call_count
    cell.sub_frame_count = m.sub_frame_count
    cell.max_delegation_depth = m.max_delegation_depth
    cell.agents_used = list(m.agents_used)
    cell.body_compactions = m.body_compactions
    cell.cost_by_agent = {
        k: round(v, 6) for k, v in m.cost_by_agent.items()
    }
    cell.provider_errors = m.provider_errors
    cell.tool_error_count = m.tool_error_count
    cell.avg_turn_latency_ms = round(m.avg_turn_latency_ms, 2)
    total_input = m.tokens_in + m.cache_read_tokens
    cell.cache_hit_rate = (
        round(m.cache_read_tokens / total_input, 4) if total_input > 0 else 0.0
    )
    cell.source_session_id = m.source_session_id
    cell.forked_from_event_id = m.forked_from_event_id


# ──────────────────────────────────────────────────────────────────────
# Acceptance evaluators — run the scoring check
# ──────────────────────────────────────────────────────────────────────


JUDGE_PROMPT_TEMPLATE = """You are an eval judge. Decide whether the workspace changes \
below satisfy the rubric. Output exactly one JSON object on the final line of \
your response, with no trailing prose:

  {{"verdict": "pass", "reason": "<one sentence, <= 200 chars>"}}

Return `pass` iff the rubric is fully met; otherwise `fail`. Keep the reason \
concise and specific.

## Rubric

{rubric}

## Workspace diff (current working tree vs base ref)

{diff_block}

## Agent's final message

{final_block}
"""


def _build_judge_prompt(
    rubric: str, diff: str, final_message: str | None,
) -> str:
    diff_block = diff.strip() if diff.strip() else "(no changes — agent did not modify any files)"
    if final_message and final_message.strip():
        final_block = final_message.strip()
    else:
        final_block = "(no final response)"
    # Cap sizes so a huge diff doesn't blow past the judge's context.
    # 40k chars is a conservative upper bound for a single LLM call;
    # prefer truncating the diff over skipping it entirely.
    if len(diff_block) > 40_000:
        diff_block = diff_block[:40_000] + "\n… [truncated]"
    if len(final_block) > 8_000:
        final_block = final_block[:8_000] + "\n… [truncated]"
    return JUDGE_PROMPT_TEMPLATE.format(
        rubric=rubric.strip(),
        diff_block=diff_block,
        final_block=final_block,
    )


_JUDGE_JSON_RE = __import__("re").compile(
    r"\{[^{}]*?\"verdict\"\s*:\s*\"[a-zA-Z]+\"[^{}]*?\}",
    __import__("re").DOTALL,
)


def _parse_judge_verdict(text: str) -> tuple[bool, str]:
    """Extract (passed, reason) from the judge's final message.

    Preferred path: pick the *last* JSON object that carries a
    `"verdict"` field. Fallback: scan the final 400 characters for
    a standalone PASS/FAIL keyword. Raises ValueError when neither
    succeeds so the caller can mark the cell as an error rather
    than silently calling the verdict incorrectly."""
    import re

    matches = _JUDGE_JSON_RE.findall(text or "")
    for raw in reversed(matches):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        verdict = str(obj.get("verdict", "")).strip().lower()
        reason = str(obj.get("reason", "")).strip()[:250]
        if verdict in ("pass", "fail"):
            return verdict == "pass", reason

    # Loose fallback — look for a clear verdict keyword at the tail.
    tail = (text or "")[-400:].upper()
    has_pass = re.search(r"\bPASS\b", tail) is not None
    has_fail = re.search(r"\bFAIL\b", tail) is not None
    if has_pass and not has_fail:
        return True, "(verdict extracted from plain text)"
    if has_fail and not has_pass:
        return False, "(verdict extracted from plain text)"

    raise ValueError("judge response has no parseable verdict")


def _extract_final_user_text(events: list[dict]) -> str:
    """Pull the last assistant-to-user message text out of a snapshot.

    The event log contains many MESSAGE_APPENDED entries; we want the
    final one whose `to` is `user`, which is the root frame's bubble
    reply."""
    for ev in reversed(events):
        if ev.get("type") != "message.appended":
            continue
        payload = ev.get("payload") or {}
        if payload.get("to") != "user":
            continue
        chunks: list[str] = []
        for block in payload.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                chunks.append(block.get("text") or "")
        if chunks:
            return "\n".join(chunks)
    return ""


def _run_llm_judge_acceptance(
    workspace: Path,
    spec: AcceptanceLLMJudge,
    *,
    port: int,
    agent_session_id: str,
) -> tuple[bool, str, str, str]:
    """Drive a judge session and return (passed, reason, raw_text,
    judge_session_id).

    The judge runs on the same eval server as the agent session — one
    extra POST /api/sessions with `judge_preset`, followed by the usual
    poll. The judge never has to see the agent's internal events; it
    gets a plain-text diff + optional final message assembled by the
    runner.
    """
    # 1. git diff HEAD inside the workspace — captures seed patch plus
    #    everything the agent wrote. The judge reads the rubric to
    #    decide whether the combined change meets the spec.
    diff_proc = subprocess.run(
        ["git", "-C", str(workspace), "diff", "HEAD"],
        capture_output=True, text=True,
    )
    diff = diff_proc.stdout

    # 2. Agent's final message to the user, best-effort.
    try:
        snap = _http_json(
            "GET", f"http://127.0.0.1:{port}/api/sessions/{agent_session_id}/snapshot",
            timeout=10,
        )
        final_msg = _extract_final_user_text(snap.get("events") or [])
    except Exception:
        final_msg = ""

    prompt = _build_judge_prompt(spec.rubric, diff, final_msg)

    # 3. Judge session.
    created = _http_json(
        "POST", f"http://127.0.0.1:{port}/api/sessions",
        {"preset": spec.judge_preset},
    )
    jsid = created["session_id"]
    _http_json(
        "POST", f"http://127.0.0.1:{port}/api/sessions/{jsid}/messages",
        {"text": prompt},
    )
    _wait_for_done(port, jsid, timeout_sec=float(spec.timeout_sec))

    # 4. Pull judge's final text and parse the verdict.
    snap = _http_json(
        "GET", f"http://127.0.0.1:{port}/api/sessions/{jsid}/snapshot",
        timeout=10,
    )
    judge_text = _extract_final_user_text(snap.get("events") or [])
    passed, reason = _parse_judge_verdict(judge_text)
    return passed, reason, judge_text[:2000], jsid


def _acceptance_files(spec: AcceptancePytest) -> list[str]:
    """Extract the distinct test file paths from `spec.node_ids`.

    `tests/foo.py::test_bar[case]` → `tests/foo.py`. Used by the
    regression check so we only re-run the files that contain the
    acceptance targets, not the entire workspace test suite (which
    can be minutes for httpx or similar).
    """
    out: list[str] = []
    for nid in spec.node_ids:
        head = nid.split("::", 1)[0]
        if head and not head.startswith("-") and head not in out:
            out.append(head)
    return out


def _collect_passing_nodes(
    workspace: Path, spec: AcceptancePytest,
) -> set[str] | None:
    """Run pytest on the acceptance files in verbose mode, return the
    set of node_ids that currently PASS.

    Returns None when pytest can't be collected at all (e.g. missing
    files — typical at baseline for synthetic tasks whose tests are
    planted by the seed). Used twice per cell: once before the agent
    runs (baseline) and once after (post) — the delta is
    `regression_count`.
    """
    files = _acceptance_files(spec)
    if not files:
        return set()
    env = os.environ.copy()
    if spec.pythonpath:
        extra = os.pathsep.join(
            str((workspace / p).resolve()) for p in spec.pythonpath
        )
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{extra}{os.pathsep}{existing}" if existing else extra
        )
    cmd = [
        sys.executable, "-m", "pytest",
        "-v", "--tb=no", "--no-header",
        "-p", "no:cacheprovider",
        *spec.extra_args, *files,
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=str(workspace), env=env,
            capture_output=True, text=True, timeout=120,
        )
    except Exception:  # noqa: BLE001
        return None
    passing: set[str] = set()
    for line in proc.stdout.splitlines():
        # pytest -v line shape:  tests/x.py::test_y PASSED                [ 40%]
        if " PASSED" not in line:
            continue
        node = line.split(" PASSED", 1)[0].strip()
        if node:
            passing.add(node)
    return passing


def _collect_diff_stats(workspace: Path) -> dict | None:
    """Parse `git diff --numstat HEAD` into {added, removed, files}.

    Runs after the agent has finished editing. Returns None if the
    workspace isn't a git worktree (shouldn't happen — every cell's
    workspace starts from `_commit_seed_baseline`). Binary file
    entries show `-` in numstat — we skip those for line totals and
    count them toward the file count.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(workspace), "diff", "--numstat", "HEAD"],
            capture_output=True, text=True, check=True, timeout=20,
        )
    except Exception:  # noqa: BLE001
        return None
    added = 0
    removed = 0
    files: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        a, r, path = parts[0], parts[1], parts[2]
        try:
            added += int(a)
        except ValueError:
            pass
        try:
            removed += int(r)
        except ValueError:
            pass
        files.add(path)
    return {
        "lines_added": added,
        "lines_removed": removed,
        "files_touched": len(files),
        "files": sorted(files),
    }


def _collect_reference_stats(
    workspace: Path, apply_commit: str,
) -> dict | None:
    """Parse `git show --numstat <apply_commit>` into the ref-diff
    summary the similarity comparison needs.

    The caller decides whether `apply_commit` is the canonical fix —
    for n1/s/x tasks it is; for n2/n3 it points at a spec-test
    commit and the caller skips this step.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(workspace),
             "show", "--numstat", "--pretty=format:", apply_commit],
            capture_output=True, text=True, check=True, timeout=20,
        )
    except Exception:  # noqa: BLE001
        return None
    added = 0
    removed = 0
    files: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        a, r, path = parts[0], parts[1], parts[2]
        try:
            added += int(a)
        except ValueError:
            pass
        try:
            removed += int(r)
        except ValueError:
            pass
        files.add(path)
    return {
        "lines_added": added,
        "lines_removed": removed,
        "files": sorted(files),
    }


def _ref_is_canonical_fix(task: Task) -> bool:
    """Does `task.setup.apply_commit` point at the upstream fix?

    True for n1 (feature shipped in the same commit as its spec
    tests), s1-s4 (no commit involved but the apply_commit field is
    empty so this falls through), x1-x3 (the acceptance tests were
    added by the same commit that fixed the bug). False for n2/n3
    where apply_commit is a dedicated spec-test commit separate from
    the implementation commit.
    """
    return task.id not in {"n2-pytest-pythonpath", "n3-seed-baseline-commit"}


def _run_pytest_acceptance(
    workspace: Path, spec: AcceptancePytest,
) -> tuple[int, str | None]:
    """Execute pytest on `spec.node_ids` inside the workspace.

    Returns (exit_code, tail). `tail` is the last 60 lines of stdout+stderr
    when the run failed, None otherwise. Uses the repo's configured
    interpreter via `sys.executable` so nature's pytest plugins still
    load; the workspace is just a worktree / clone, not a venv of its
    own.

    `spec.pythonpath` lets the task author pin which workspace subdirs
    land on `sys.path` first — critical for `src/`-layout repos
    (pluggy, attrs, …) so the agent's edits are what pytest imports
    instead of whatever copy the ambient venv has installed.
    """
    env = os.environ.copy()
    if spec.pythonpath:
        extra = os.pathsep.join(
            str((workspace / p).resolve()) for p in spec.pythonpath
        )
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{extra}{os.pathsep}{existing}" if existing else extra
        )
    cmd = [
        sys.executable, "-m", "pytest", "-x", "--no-header", "-q",
        *spec.extra_args, *spec.node_ids,
    ]
    proc = subprocess.run(
        cmd, cwd=str(workspace), env=env, capture_output=True, text=True,
        timeout=max(spec.timeout_sec, 30),
    )
    if proc.returncode == 0:
        return 0, None
    tail = "\n".join(
        (proc.stdout + "\n" + proc.stderr).splitlines()[-60:]
    )
    return proc.returncode, tail


# ──────────────────────────────────────────────────────────────────────
# Preamble — inlined into every task prompt
# ──────────────────────────────────────────────────────────────────────


_PROMPT_PREAMBLE_TEMPLATE = (
    "## Eval harness rules (important)\n\n"
    "1. **DO NOT run pytest, install packages, or set up a venv.** "
    "The eval harness runs the acceptance tests for you after this "
    "session ends. Any time you spend `pip install`ing, `venv`ing, "
    "or running pytest is wasted — it will not affect the score.\n"
    "2. Your only job is to make the minimum code edits that make the "
    "acceptance tests pass. Focus on the files the task prompt points at.\n"
    "3. Stop as soon as you have made the edit. Do not verify with "
    "`python -c` either. The harness verifies.\n"
    "4. You are running inside a git worktree / clone. Any seed tests or "
    "patches are already applied to the working tree when you see them.\n"
    "5. **Your working directory is `{workspace}`.** Every path in the "
    "task prompt is relative to this directory. Do not prepend "
    "`/repo/`, `/app/`, or any made-up parent — if the prompt says "
    "`jsonptr.py`, read `jsonptr.py` directly (or prefix with "
    "`{workspace}/` when an absolute path is required).\n\n"
    "---\n\n"
)


def _prompt_preamble(workspace: Path) -> str:
    """Render the per-cell preamble with the concrete workspace path.

    Stage 1 runs surfaced path hallucinations where agents guessed
    `/repo/<file>` instead of reading relative paths from cwd. Pinning
    the workspace path in the preamble removes the guessing step and
    keeps the first turn from burning on fruitless `find /`-style
    exploration.
    """
    return _PROMPT_PREAMBLE_TEMPLATE.format(workspace=str(workspace))


# Keep the old constant name alive for callers that still reference it
# (tests, scripts). Empty workspace renders cleanly enough.
PROMPT_PREAMBLE = _PROMPT_PREAMBLE_TEMPLATE.format(workspace="<workspace>")


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────


def run_cell(
    task: Task,
    preset: str,
    *,
    seed: int | None = None,
    repo_root: Path | None = None,
    logs_dir: Path | None = None,
    nature_bin: str | None = None,
    timeout_override: float | None = None,
) -> CellResult:
    """Run one (task × preset) cell and return the result.

    `seed` — 0-based index within a multi-seed sweep. Stored on the
    result so the aggregation layer can group cells by (task, preset)
    and report mean / std / pass-rate. When unset, the cell is
    recorded as a one-off execution (seed=None), matching the
    pre-Phase-2a single-run shape.

    `repo_root` defaults to `os.getcwd()` and is only meaningful for
    tasks with `target.repo == "local"` (worktree source) or for
    looking up the pricing table in `_compute_cost`. `logs_dir`
    defaults to `<repo_root>/.nature/eval/results/logs/` — cell
    event logs are copied there plus into the user's main events
    dir so the dashboard can replay them.
    """
    repo_root = Path(repo_root or os.getcwd()).resolve()
    logs_dir = Path(logs_dir or (repo_root / ".nature" / "eval" / "results" / "logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    result = CellResult(
        task_id=task.id, preset=preset, started_at=started, seed=seed,
    )

    workspace: _WorkspaceHandle | None = None
    server_proc: subprocess.Popen | None = None
    nature_home: Path | None = None
    port: int | None = None
    sid: str | None = None
    t_session_start: float | None = None

    try:
        # 1. Workspace.
        workspace = _setup_workspace(task, repo_root)

        # 2. Server with isolated NATURE_HOME.
        port = _free_port()
        nature_home = Path(tempfile.mkdtemp(prefix=f"nature-eval-home-{task.id}-"))
        env = os.environ.copy()
        env["NATURE_HOME"] = str(nature_home)
        env["NATURE_SERVER_PORT"] = str(port)
        # Pin bind to localhost even if the user's default shell env
        # has NATURE_SERVER_HOST set to something wider — eval cells
        # should never accept external connections.
        env["NATURE_SERVER_HOST"] = "localhost"
        bin_path = (
            nature_bin or shutil.which("nature")
            or str(repo_root / ".venv" / "bin" / "nature")
        )
        server_log_path = (
            (logs_dir / f"server-{task.id}-{port}-{int(time.time())}.log")
        )
        server_log_fh = server_log_path.open("w", encoding="utf-8")
        server_proc = subprocess.Popen(
            [bin_path, "server", "start", "--cwd", str(workspace.path)],
            env=env, cwd=str(repo_root),
            stdout=server_log_fh, stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_server(port, timeout=60)
        except RuntimeError as exc:
            # Surface the server's own output so parallel-startup
            # diagnostics aren't swallowed.
            server_log_fh.flush()
            try:
                tail = "\n".join(
                    server_log_path.read_text(
                        encoding="utf-8", errors="replace",
                    ).splitlines()[-30:]
                )
            except Exception:
                tail = "(could not read server log)"
            raise RuntimeError(
                f"{exc}\n--- server log tail ({server_log_path}) ---\n{tail}",
            ) from exc

        # 2b. Quality baseline — pytest on the acceptance file(s)
        # BEFORE the agent runs, so we can spot regressions later.
        # Only meaningful for pytest acceptance; llm_judge cells skip.
        if isinstance(task.acceptance, AcceptancePytest):
            baseline_pass = _collect_passing_nodes(
                workspace.path, task.acceptance,
            )
            result.pre_suite_pass_count = (
                len(baseline_pass) if baseline_pass is not None else None
            )
        else:
            baseline_pass = None

        # 3. Create session with preset.
        t_session_start = time.time()
        created = _http_json(
            "POST", f"http://127.0.0.1:{port}/api/sessions", {"preset": preset},
        )
        sid = created["session_id"]
        result.session_id = sid

        # 4. Send prompt.
        _http_json(
            "POST", f"http://127.0.0.1:{port}/api/sessions/{sid}/messages",
            {"text": _prompt_preamble(workspace.path) + task.prompt},
        )

        # 5. Wait for the agent to finish.
        # `timeout_override` lets the caller decouple the session-level
        # watchdog from the task's acceptance-test timeout. Used by
        # `nature eval run --timeout-override` to measure actual wall
        # clock for hypothesis tests where we don't want the default
        # 420s gate to confound the result.
        if timeout_override is not None:
            agent_timeout = float(timeout_override)
        elif isinstance(task.acceptance, AcceptancePytest):
            agent_timeout = float(task.acceptance.timeout_sec)
        else:
            agent_timeout = 300.0
        _wait_for_done(port, sid, timeout_sec=agent_timeout)
        result.latency_sec = round(time.time() - t_session_start, 2)

        # 6. Acceptance — dispatch on the task's acceptance spec.
        if isinstance(task.acceptance, AcceptancePytest):
            exit_code, tail = _run_pytest_acceptance(
                workspace.path, task.acceptance,
            )
            result.pytest_exit = exit_code
            result.passed = exit_code == 0
            result.pytest_tail = tail
        elif isinstance(task.acceptance, AcceptanceLLMJudge):
            passed, reason, raw, jsid = _run_llm_judge_acceptance(
                workspace.path, task.acceptance,
                port=port, agent_session_id=sid,
            )
            result.passed = passed
            result.judge_reason = reason
            result.judge_raw = raw
            result.judge_session_id = jsid
        else:
            raise NotImplementedError(
                f"acceptance type {task.acceptance.type!r} not implemented yet",
            )

        # 7. Quality axes — diff stats + post-suite regression check +
        # reference-fix similarity. Run regardless of acceptance
        # verdict so we can see "passed but broke other tests" or
        # "failed but still produced a minimal edit".
        diff_stats = _collect_diff_stats(workspace.path)
        if diff_stats is not None:
            result.diff_lines_added = diff_stats["lines_added"]
            result.diff_lines_removed = diff_stats["lines_removed"]
            result.diff_files_touched = diff_stats["files_touched"]

        if isinstance(task.acceptance, AcceptancePytest):
            post_pass = _collect_passing_nodes(
                workspace.path, task.acceptance,
            )
            if post_pass is not None:
                result.post_suite_pass_count = len(post_pass)
                if baseline_pass is not None:
                    result.regression_count = len(baseline_pass - post_pass)

        # Reference similarity — only when apply_commit is the fix.
        if (
            isinstance(task.setup, (SetupApplyDiffFromCommit, SetupGitCloneAtRef))
            and getattr(task.setup, "apply_commit", None)
            and _ref_is_canonical_fix(task)
            and diff_stats is not None
        ):
            ref_stats = _collect_reference_stats(
                workspace.path, task.setup.apply_commit,
            )
            if ref_stats is not None:
                agent_files = set(diff_stats.get("files", []))
                ref_files = set(ref_stats.get("files", []))
                union = agent_files | ref_files
                if union:
                    result.ref_files_overlap = round(
                        len(agent_files & ref_files) / len(union), 4,
                    )
                ref_total = ref_stats["lines_added"] + ref_stats["lines_removed"]
                agent_total = (
                    diff_stats["lines_added"] + diff_stats["lines_removed"]
                )
                if ref_total > 0:
                    result.ref_line_ratio = round(agent_total / ref_total, 3)

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        # Metric extraction (best-effort even on errors).
        if sid is not None and nature_home is not None:
            if result.latency_sec is None and t_session_start is not None:
                result.latency_sec = round(time.time() - t_session_start, 2)
            session_jsonl = nature_home / "events" / f"{sid}.jsonl"
            try:
                m = _extract_metrics(session_jsonl, repo_root)
                _copy_metrics_onto_cell(result, m)
            except Exception as exc:
                if result.error is None:
                    result.error = f"metric_extract_failed: {exc}"

            # Archive the event log + hydrate into user's main events.
            try:
                if session_jsonl.exists():
                    tag = f"{task.id}-{preset}-{int(started)}"
                    dest = logs_dir / f"{tag}.jsonl"
                    shutil.copy2(session_jsonl, dest)
                    result.event_log_path = str(dest)

                    try:
                        from nature.config.settings import get_nature_home
                        main_events = get_nature_home() / "events"
                    except Exception:
                        main_events = Path.home() / ".nature" / "events"
                    main_events.mkdir(parents=True, exist_ok=True)
                    main_dest = main_events / f"{sid}.jsonl"
                    if not main_dest.exists():
                        shutil.copy2(session_jsonl, main_dest)
                        (main_events / f"{sid}.eval.json").write_text(
                            json.dumps({
                                "source": "nature-eval",
                                "task_id": task.id,
                                "preset": preset,
                                "tag": tag,
                            }, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8",
                        )
            except Exception:  # noqa: BLE001
                pass

        # Teardown.
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        if port is not None:
            _release_port(port)
        if workspace is not None:
            for fn in workspace.cleanup:
                try:
                    fn()
                except Exception:
                    pass
        if nature_home is not None and nature_home.exists():
            shutil.rmtree(nature_home, ignore_errors=True)

        result.finished_at = time.time()

    return result
