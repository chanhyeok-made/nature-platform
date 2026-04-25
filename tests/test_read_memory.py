"""Tests for ReadMemory — segments, dedup, staleness, merge, eviction."""

from __future__ import annotations

import os
import tempfile
import time

import pytest

from nature.context.read_memory import (
    LineRange,
    ReadMemory,
    ReadMemoryEntry,
    ReadSegment,
    is_fully_covered,
    merge_ranges,
    read_line_range,
)


# ──────────────────────────────────────────────────────────────────────
# LineRange helpers
# ──────────────────────────────────────────────────────────────────────


def test_merge_ranges_empty():
    assert merge_ranges([]) == []


def test_merge_ranges_non_overlapping():
    r = merge_ranges([LineRange(0, 5), LineRange(10, 15)])
    assert len(r) == 2
    assert r[0].start == 0 and r[0].end == 5
    assert r[1].start == 10 and r[1].end == 15


def test_merge_ranges_overlapping():
    r = merge_ranges([LineRange(0, 10), LineRange(5, 15)])
    assert len(r) == 1
    assert r[0].start == 0 and r[0].end == 15


def test_merge_ranges_adjacent():
    r = merge_ranges([LineRange(0, 5), LineRange(5, 10)])
    assert len(r) == 1
    assert r[0].start == 0 and r[0].end == 10


def test_is_fully_covered_true():
    seen = [LineRange(0, 20)]
    assert is_fully_covered(seen, LineRange(5, 10)) is True


def test_is_fully_covered_false():
    seen = [LineRange(0, 5), LineRange(10, 15)]
    assert is_fully_covered(seen, LineRange(3, 12)) is False


# ──────────────────────────────────────────────────────────────────────
# ReadMemoryEntry
# ──────────────────────────────────────────────────────────────────────


def _make_entry(
    path: str = "/tmp/test.py",
    lines: list[str] | None = None,
    start: int = 0,
    mtime_ns: int = 100,
) -> ReadMemoryEntry:
    if lines is None:
        lines = [f"line{i}" for i in range(10)]
    text = "\n".join(lines)
    return ReadMemoryEntry(
        path=path,
        mtime_ns=mtime_ns,
        total_lines=len(lines) + start,
        segments=[ReadSegment(start, start + len(lines), text)],
    )


def test_entry_expired_when_segments_none():
    e = ReadMemoryEntry(
        path="/x", mtime_ns=0, total_lines=0, segments=None
    )
    assert e.expired is True
    assert e.content is None
    assert e.seen_ranges == []


def test_entry_active_when_segments_present():
    e = _make_entry()
    assert e.expired is False
    assert e.content is not None
    assert "line0" in e.content


def test_entry_covers():
    e = _make_entry(lines=[f"L{i}" for i in range(20)], start=0)
    assert e.covers(0, 10) is True
    assert e.covers(0, 20) is True
    assert e.covers(15, 25) is False


def test_entry_get_lines():
    lines = [f"L{i}" for i in range(10)]
    e = _make_entry(lines=lines, start=0)
    got = e.get_lines(3, 6)
    assert got == ["L3", "L4", "L5"]


def test_entry_get_lines_returns_none_for_uncovered():
    e = _make_entry(lines=["a", "b"], start=0)
    assert e.get_lines(5, 10) is None


def test_entry_add_segment_disjoint():
    e = _make_entry(lines=["a", "b"], start=0)
    e.add_segment(ReadSegment(5, 7, "c\nd"))
    assert len(e.segments) == 2
    assert e.covers(0, 2) is True
    assert e.covers(5, 7) is True
    assert e.covers(2, 5) is False


def test_entry_add_segment_overlapping():
    e = _make_entry(lines=["a", "b", "c"], start=0)  # [0,3)
    e.add_segment(ReadSegment(2, 5, "C\nD\nE"))       # [2,5) overlaps at 2
    assert len(e.segments) == 1
    s = e.segments[0]
    assert s.start == 0 and s.end == 5
    # overlap region (line 2): new segment wins → "C" not "c"
    merged_lines = s.text.splitlines()
    assert merged_lines[2] == "C"
    assert merged_lines[0] == "a"


def test_entry_expire():
    e = _make_entry()
    assert not e.expired
    e.expire()
    assert e.expired
    assert e.segments is None
    assert e.content is None
    assert e.path == "/tmp/test.py"  # metadata preserved


def test_entry_content_bytes():
    e = _make_entry(lines=["hello", "world"])
    assert e.content_bytes > 0
    e.expire()
    assert e.content_bytes == 0


# ──────────────────────────────────────────────────────────────────────
# ReadMemory — get/set/staleness
# ──────────────────────────────────────────────────────────────────────


def _write_tmp(content: str) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    f.write(content)
    f.close()
    return f.name


def test_memory_set_and_get():
    mem = ReadMemory()
    path = _write_tmp("hello\nworld\n")
    try:
        mtime = os.stat(path).st_mtime_ns
        e = ReadMemoryEntry(
            path=path, mtime_ns=mtime, total_lines=2,
            segments=[ReadSegment(0, 2, "hello\nworld")],
        )
        mem.set(path, e)
        got = mem.get(path)
        assert got is not None
        assert got.path == path
        assert got.hit_count == 1
    finally:
        os.unlink(path)


def test_memory_get_returns_none_for_unknown():
    mem = ReadMemory()
    assert mem.get("/nonexistent/file.py") is None


def test_memory_stale_deletes_entry():
    """mtime mismatch on active entry → delete (live stale)."""
    path = _write_tmp("v1\n")
    try:
        mtime_old = os.stat(path).st_mtime_ns
        e = ReadMemoryEntry(
            path=path, mtime_ns=mtime_old, total_lines=1,
            segments=[ReadSegment(0, 1, "v1")],
        )
        mem = ReadMemory()
        mem.set(path, e)

        # Modify file → mtime changes
        time.sleep(0.01)
        with open(path, "w") as f:
            f.write("v2\n")

        got = mem.get(path)
        assert got is None
        assert not mem.has(path)
    finally:
        os.unlink(path)


def test_memory_expired_entry_skips_mtime_check():
    """Expired entry → no mtime check, returns as-is."""
    path = _write_tmp("v1\n")
    try:
        e = ReadMemoryEntry(
            path=path, mtime_ns=1,  # wrong mtime on purpose
            total_lines=1, segments=None,  # expired
        )
        mem = ReadMemory()
        mem.set(path, e)

        got = mem.get(path)
        assert got is not None
        assert got.expired is True
    finally:
        os.unlink(path)


# ──────────────────────────────────────────────────────────────────────
# ReadMemory — merge
# ──────────────────────────────────────────────────────────────────────


def test_merge_child_into_parent():
    parent = ReadMemory()
    child = ReadMemory()
    child.set("/a.py", ReadMemoryEntry(
        path="/a.py", mtime_ns=100, total_lines=10,
        segments=[ReadSegment(0, 10, "text")], depth=0,
    ))
    parent.merge(child, depth_increment=1)
    got = parent._entries.get(str(os.path.realpath("/a.py")) if os.path.exists("/a.py") else "/a.py")
    # Use the actual key
    assert len(parent) == 1
    entry = list(parent._entries.values())[0]
    assert entry.depth == 1


def test_merge_lower_depth_wins():
    parent = ReadMemory()
    parent.set("/b.py", ReadMemoryEntry(
        path="/b.py", mtime_ns=100, total_lines=5,
        segments=[ReadSegment(0, 5, "parent")], depth=0,
    ))
    child = ReadMemory()
    child.set("/b.py", ReadMemoryEntry(
        path="/b.py", mtime_ns=100, total_lines=5,
        segments=[ReadSegment(0, 5, "child")], depth=0,
    ))
    parent.merge(child, depth_increment=1)
    entry = list(parent._entries.values())[0]
    assert entry.depth == 0  # parent's own depth=0 wins over child's depth=0+1=1
    assert "parent" in entry.segments[0].text


# ──────────────────────────────────────────────────────────────────────
# ReadMemory — eviction
# ──────────────────────────────────────────────────────────────────────


def test_evict_to_budget_expires_least_used_deepest_first():
    mem = ReadMemory(max_bytes=100)  # very small budget
    # Entry A: depth=0, hit_count=5 (well-used, shallow)
    mem.set("/a.py", ReadMemoryEntry(
        path="/a.py", mtime_ns=1, total_lines=1,
        segments=[ReadSegment(0, 1, "A" * 60)], depth=0, hit_count=5,
    ))
    # Entry B: depth=2, hit_count=0 (unused, deep) → should expire first
    mem.set("/b.py", ReadMemoryEntry(
        path="/b.py", mtime_ns=1, total_lines=1,
        segments=[ReadSegment(0, 1, "B" * 60)], depth=2, hit_count=0,
    ))
    assert mem.total_bytes > 100
    mem.evict_to_budget()

    b = list(mem._entries.values())
    # B should be expired (hit_count=0, depth=2 → evicted first)
    b_entry = [e for e in b if "b.py" in e.path][0]
    assert b_entry.expired is True
    # A should still be active
    a_entry = [e for e in b if "a.py" in e.path][0]
    assert a_entry.expired is False


# ──────────────────────────────────────────────────────────────────────
# read_line_range helper
# ──────────────────────────────────────────────────────────────────────


def test_read_line_range():
    path = _write_tmp("\n".join(f"line{i}" for i in range(20)) + "\n")
    try:
        lines, total = read_line_range(path, offset=5, limit=3)
        assert lines == ["line5", "line6", "line7"]
        assert total == 20
    finally:
        os.unlink(path)


def test_read_line_range_past_eof():
    path = _write_tmp("a\nb\nc\n")
    try:
        lines, total = read_line_range(path, offset=10, limit=5)
        assert lines == []
        assert total > 0
    finally:
        os.unlink(path)
