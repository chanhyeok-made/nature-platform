"""WebFetch — fetch a URL and return its text content.

Designed for research workloads: a researcher agent hands a URL, gets
back the visible text (scripts / styles / nav chrome stripped) so it
can reason over the page content without paying the tokens for raw
HTML tags.

Output cap: 40KB of text by default (configurable per-call). A
truncation marker is appended when the page exceeds the cap so the
model knows something was cut.

Deliberately stays on the Python stdlib (httpx + html.parser) — no
trafilatura / BeautifulSoup dependency — so the tool stays portable.
For pages where clean extraction matters more than simplicity, a
follow-up PR can swap in a heavier parser behind the same interface.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any

import httpx
from pydantic import BaseModel, Field

from nature.protocols.tool import ToolContext, ToolResult
from nature.tools.base import BaseTool

MAX_TEXT_BYTES = 40 * 1024
REQUEST_TIMEOUT_SEC = 20.0
USER_AGENT = "nature-agent/1.0 (+https://github.com/chanhyeok/nature)"


class _VisibleTextExtractor(HTMLParser):
    """Collect visible text from an HTML document.

    Skips `<script>`, `<style>`, `<noscript>`, `<svg>` subtrees where
    content is never meant to be read. Collapses whitespace runs to
    single spaces, preserves paragraph-ish breaks on block-level
    tags so the output reads like the page does.
    """

    _SKIP_TAGS = frozenset({"script", "style", "noscript", "svg", "canvas", "template"})
    _BLOCK_TAGS = frozenset({
        "p", "div", "section", "article", "header", "footer", "nav",
        "main", "aside", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "tr", "pre", "blockquote", "hr", "br",
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        # Track title separately so callers can report it.
        self._in_title = False
        self.title: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title and self.title is None:
            title = data.strip()
            if title:
                self.title = title
        self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse whitespace: runs of spaces/tabs → single space,
        # but keep at most 2 consecutive newlines (paragraph break).
        raw = re.sub(r"[ \t\f\v]+", " ", raw)
        raw = re.sub(r"\n[ \t]+", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


class WebFetchInput(BaseModel):
    url: str = Field(description="Absolute HTTP(S) URL to fetch")
    max_bytes: int | None = Field(
        default=None,
        description=(
            "Cap the extracted text at this many bytes (default: 40KB). "
            "Output is truncated with a [TRUNCATED] marker when it exceeds."
        ),
    )


class WebFetchTool(BaseTool):
    input_model = WebFetchInput

    @property
    def name(self) -> str:
        return "WebFetch"

    @property
    def description(self) -> str:
        return (
            "Fetch a web page and return its visible text (scripts, styles "
            "and markup stripped). Use when the researcher needs to read "
            "external documentation, API references, or blog posts. Output "
            "starts with the page title + URL, then the cleaned body text. "
            "Capped at ~40KB; truncation is marked explicitly."
        )

    def is_read_only(self, input: dict[str, Any]) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any]) -> bool:
        return True

    async def run(self, params: WebFetchInput, context: ToolContext) -> ToolResult:
        url = params.url
        if not (url.startswith("http://") or url.startswith("https://")):
            return ToolResult(
                output=f"Invalid URL (must start with http:// or https://): {url}",
                is_error=True,
            )

        cap = params.max_bytes or MAX_TEXT_BYTES
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=REQUEST_TIMEOUT_SEC,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            try:
                resp = await client.get(url)
            except httpx.HTTPError as exc:
                return ToolResult(
                    output=f"Request failed: {type(exc).__name__}: {exc}",
                    is_error=True,
                )

        if resp.status_code >= 400:
            return ToolResult(
                output=f"HTTP {resp.status_code} from {url}",
                is_error=True,
            )

        ctype = resp.headers.get("content-type", "").lower()
        body = resp.text

        if "text/html" in ctype or "<html" in body[:1024].lower():
            parser = _VisibleTextExtractor()
            try:
                parser.feed(body)
                parser.close()
            except Exception as exc:  # noqa: BLE001
                return ToolResult(
                    output=f"HTML parse failed: {exc}",
                    is_error=True,
                )
            text = parser.get_text()
            title = parser.title or "(no title)"
        else:
            # Plain text / markdown / json — return as-is.
            text = body
            title = resp.url.path or str(resp.url)

        truncated = False
        if len(text.encode("utf-8")) > cap:
            encoded = text.encode("utf-8")[:cap]
            text = encoded.decode("utf-8", errors="ignore")
            truncated = True

        header = f"# {title}\n# URL: {resp.url}\n# Content-Type: {ctype or '(unknown)'}\n"
        tail = "\n\n[TRUNCATED]" if truncated else ""
        return ToolResult(output=f"{header}\n{text}{tail}", is_error=False)


__all__ = ["WebFetchTool", "WebFetchInput"]
