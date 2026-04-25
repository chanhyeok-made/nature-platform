"""ReadMemory — ledger of what the model has seen on disk.

NOT a file cache. This is a structured record of "which file, which
line ranges, at what mtime" the model observed via Read. Consumers:

- **Read tool**: dedup (same range + same mtime → stub), cache serve
  (different range of a cached file → serve from segments without
  disk I/O), and state set after reading.
- **Edit guard (Pack Gate)**: strict check — old_string must appear in
  a previously-read segment. If not → Block("Read first").
- **Frame tree propagation**: child.read_memory merges into parent on
  resolve, with depth+1. Scatter-gather aggregation is automatic.

Lifecycle:
- Created by the file_state Pack at frame open and stored in
  `Frame.pack_state["read_memory"]`.
- Tools access it via `ToolContext.pack_state["read_memory"]`.
- Event-sourced via READ_MEMORY_SET (metadata only, no content).
  On resume, entries are born expired (segments=None).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────


@dataclass
class LineRange:
    """A half-open line range [start, end)."""

    start: int
    end: int


@dataclass
class ReadSegment:
    """One contiguous chunk of lines read from a file."""

    start: int   # inclusive, 0-based
    end: int     # exclusive
    text: str    # joined line content for this range


@dataclass
class ReadMemoryEntry:
    """What the model knows about a single file path."""

    path: str
    mtime_ns: int
    total_lines: int
    segments: list[ReadSegment] | None   # None = expired
    depth: int = 0                        # 0=self, 1=child, 2=grandchild
    hit_count: int = 0

    @property
    def expired(self) -> bool:
        return self.segments is None

    @property
    def content(self) -> str | None:
        """All read segments joined in order. None if expired."""
        if self.segments is None:
            return None
        if not self.segments:
            return ""
        sorted_segs = sorted(self.segments, key=lambda s: s.start)
        return "\n".join(seg.text for seg in sorted_segs)

    @property
    def seen_ranges(self) -> list[LineRange]:
        if self.segments is None:
            return []
        raw = [LineRange(s.start, s.end) for s in self.segments]
        return merge_ranges(raw)

    @property
    def content_bytes(self) -> int:
        if self.segments is None:
            return 0
        return sum(len(s.text.encode("utf-8")) for s in self.segments)

    def covers(self, start: int, end: int) -> bool:
        """True if [start, end) is fully within seen_ranges."""
        return is_fully_covered(self.seen_ranges, LineRange(start, end))

    def get_lines(self, start: int, end: int) -> list[str] | None:
        """Extract lines [start, end) from stored segments. None if
        the range isn't fully covered by a single segment."""
        if self.segments is None:
            return None
        for seg in self.segments:
            if seg.start <= start and seg.end >= end:
                lines = seg.text.splitlines()
                return lines[start - seg.start : end - seg.start]
        return None

    def add_segment(self, new: ReadSegment) -> None:
        """Merge overlapping, append disjoint. No-op if expired."""
        if self.segments is None:
            return
        merged: list[ReadSegment] = []
        new_lines = new.text.splitlines()
        current = new

        for seg in sorted(self.segments, key=lambda s: s.start):
            if seg.end < current.start or seg.start > current.end:
                # No overlap
                merged.append(seg)
            else:
                # Overlap — merge. Newer read takes precedence for
                # overlapping lines.
                union_start = min(seg.start, current.start)
                union_end = max(seg.end, current.end)
                old_lines = seg.text.splitlines()
                cur_lines = current.text.splitlines()
                result_lines: list[str] = []
                for i in range(union_start, union_end):
                    if current.start <= i < current.end:
                        result_lines.append(cur_lines[i - current.start])
                    elif seg.start <= i < seg.end:
                        result_lines.append(old_lines[i - seg.start])
                    else:
                        result_lines.append("")
                current = ReadSegment(
                    union_start, union_end, "\n".join(result_lines)
                )

        merged.append(current)
        self.segments = sorted(merged, key=lambda s: s.start)

    def expire(self) -> None:
        """Clear segments but keep metadata (path, mtime, depth)."""
        self.segments = None


# ──────────────────────────────────────────────────────────────────────
# ReadMemory — the per-frame container
# ──────────────────────────────────────────────────────────────────────

DEFAULT_MAX_BYTES = 25 * 1024 * 1024  # 25 MB


class ReadMemory:
    """Per-frame read-state ledger, bounded by total content bytes."""

    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self._entries: dict[str, ReadMemoryEntry] = {}
        self._max_bytes = max_bytes

    @staticmethod
    def _key(path: str) -> str:
        try:
            return str(Path(path).resolve())
        except (OSError, ValueError):
            return path

    def get(self, path: str) -> ReadMemoryEntry | None:
        """Lookup + live staleness check.

        - Expired entries: no mtime check (content already gone).
          Returns the entry as-is — caller knows it's expired.
        - Active entries: mtime check. Mismatch → delete (stale).
        """
        key = self._key(path)
        entry = self._entries.get(key)
        if entry is None:
            return None

        if entry.expired:
            entry.hit_count += 1
            return entry

        try:
            disk_mtime = os.stat(path).st_mtime_ns
        except OSError:
            del self._entries[key]
            return None

        if disk_mtime != entry.mtime_ns:
            del self._entries[key]
            return None

        entry.hit_count += 1
        return entry

    def set(self, path: str, entry: ReadMemoryEntry) -> None:
        key = self._key(path)
        self._entries[key] = entry

    def has(self, path: str) -> bool:
        return self._key(path) in self._entries

    def invalidate(self, path: str) -> None:
        self._entries.pop(self._key(path), None)

    def merge(self, child: "ReadMemory", depth_increment: int = 1) -> None:
        """Absorb child frame's read_memory. depth += increment.
        Lower depth (more direct knowledge) wins on conflict."""
        for key, entry in child._entries.items():
            new_depth = entry.depth + depth_increment
            existing = self._entries.get(key)
            if existing is not None and existing.depth <= new_depth:
                continue
            self._entries[key] = replace(entry, depth=new_depth)

    def evict_to_budget(self) -> None:
        """Expire entries until total content bytes ≤ max_bytes.
        Order: hit_count ASC, depth DESC (least-used deepest first)."""
        if self._total_bytes() <= self._max_bytes:
            return
        ranked = sorted(
            [e for e in self._entries.values() if not e.expired],
            key=lambda e: (e.hit_count, -e.depth),
        )
        for entry in ranked:
            entry.expire()
            if self._total_bytes() <= self._max_bytes:
                break

    def _total_bytes(self) -> int:
        return sum(e.content_bytes for e in self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes()

    @property
    def paths(self) -> list[str]:
        return sorted(self._entries.keys())


# ──────────────────────────────────────────────────────────────────────
# Range helpers
# ──────────────────────────────────────────────────────────────────────


def merge_ranges(ranges: list[LineRange]) -> list[LineRange]:
    """Merge overlapping/adjacent ranges into a minimal sorted list."""
    if not ranges:
        return []
    s = sorted(ranges, key=lambda r: r.start)
    merged = [LineRange(s[0].start, s[0].end)]
    for r in s[1:]:
        if r.start <= merged[-1].end:
            merged[-1] = LineRange(merged[-1].start, max(merged[-1].end, r.end))
        else:
            merged.append(LineRange(r.start, r.end))
    return merged


def is_fully_covered(seen: list[LineRange], query: LineRange) -> bool:
    """True if query [start, end) is entirely within one seen range."""
    for r in seen:
        if r.start <= query.start and r.end >= query.end:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────
# File read helper
# ──────────────────────────────────────────────────────────────────────


def read_line_range(
    path: str, offset: int, limit: int
) -> tuple[list[str], int]:
    """Read [offset, offset+limit) lines without loading the full file.

    Returns (selected_lines, total_line_count). Iterates line-by-line
    so memory usage is O(limit), not O(file_size).
    """
    selected: list[str] = []
    total = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            total = i + 1
            if offset <= i < offset + limit:
                selected.append(line.rstrip("\n"))
    return selected, total


__all__ = [
    "LineRange",
    "ReadSegment",
    "ReadMemoryEntry",
    "ReadMemory",
    "merge_ranges",
    "is_fully_covered",
    "read_line_range",
]
