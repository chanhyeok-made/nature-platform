"""WebSearch — layered web search with graceful provider fallback.

Two providers in priority order:

1. **Brave Search API** — when `BRAVE_SEARCH_API_KEY` is set.
   - Generous free tier (2K queries/mo), simple JSON response,
     no LLM-rephrased snippets.
2. **DuckDuckGo HTML (no key)** — fallback when Brave key is unset
   or Brave returns an error. Scrapes the public DDG lite HTML
   endpoint; no API key, no billing. Acceptable for personal and
   research workloads; high-volume production should still use a
   keyed API.

Returns a formatted text block: each result is `N. Title — URL`
followed by the snippet on the next line, with a one-line header
naming the provider that actually answered. A researcher or core
agent feeds the URL back to WebFetch when it wants the full page.
"""

from __future__ import annotations

import os
import re
from html import unescape
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
from pydantic import BaseModel, Field

from nature.protocols.tool import ToolContext, ToolResult
from nature.tools.base import BaseTool

BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
DDG_LITE_ENDPOINT = "https://lite.duckduckgo.com/lite/"
DEFAULT_MAX_RESULTS = 5
REQUEST_TIMEOUT_SEC = 15.0
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query string")
    max_results: int | None = Field(
        default=None,
        description="How many results to return (default: 5, max: 20)",
    )


class WebSearchTool(BaseTool):
    input_model = WebSearchInput

    @property
    def name(self) -> str:
        return "WebSearch"

    @property
    def description(self) -> str:
        return (
            "Search the web. Returns up to 5 results (configurable, "
            "max 20) with title, URL, and snippet. Uses Brave Search "
            "when BRAVE_SEARCH_API_KEY is set, falls back to "
            "DuckDuckGo (no key required) otherwise. Follow up with "
            "WebFetch on a specific URL to read the full page."
        )

    def is_read_only(self, input: dict[str, Any]) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any]) -> bool:
        return True

    async def run(self, params: WebSearchInput, context: ToolContext) -> ToolResult:
        n = max(1, min(params.max_results or DEFAULT_MAX_RESULTS, 20))
        query = params.query.strip()
        if not query:
            return ToolResult(output="Empty query.", is_error=True)

        api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
        # Prefer Brave when a key is present; fall back to DuckDuckGo
        # either when the key is missing or when the Brave call errored
        # for a reason other than rate-limiting (we don't want DDG to
        # mask abuse signals).
        if api_key:
            brave_result = await _search_brave(query, n, api_key)
            if not brave_result.is_error:
                return brave_result
            # If Brave failed on auth/HTTP error (not rate limit),
            # fall through to DDG. Rate-limit errors stay as-is so
            # users notice the quota issue.
            if "429" in (brave_result.output or ""):
                return brave_result
        return await _search_duckduckgo(query, n)


async def _search_brave(query: str, n: int, api_key: str) -> ToolResult:
    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT_SEC,
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        },
    ) as client:
        try:
            resp = await client.get(
                BRAVE_SEARCH_ENDPOINT,
                params={"q": query, "count": str(n)},
            )
        except httpx.HTTPError as exc:
            return ToolResult(
                output=f"Brave request failed: {type(exc).__name__}: {exc}",
                is_error=True,
            )

    if resp.status_code == 401:
        return ToolResult(
            output="Brave Search API key rejected (401). Check BRAVE_SEARCH_API_KEY.",
            is_error=True,
        )
    if resp.status_code == 429:
        return ToolResult(
            output="Brave Search rate limit hit (429). Retry later.",
            is_error=True,
        )
    if resp.status_code >= 400:
        return ToolResult(
            output=f"Brave Search HTTP {resp.status_code}: {resp.text[:300]}",
            is_error=True,
        )
    try:
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return ToolResult(
            output=f"Brave Search returned non-JSON: {exc}",
            is_error=True,
        )
    web_results = (data.get("web") or {}).get("results") or []
    if not web_results:
        return ToolResult(output=f"No Brave results for: {query}", is_error=False)
    return _format_results("brave", query, [
        {
            "title": (r.get("title") or "(untitled)").strip(),
            "url": (r.get("url") or "").strip(),
            "snippet": (r.get("description") or "")
                .replace("<strong>", "").replace("</strong>", "").strip(),
        }
        for r in web_results[:n]
    ])


# DDG lite's response shape (as of 2026-04): result anchors look like
#   <a rel="nofollow" href="URL" class='result-link'>TITLE</a>
# with attributes in unpredictable order and mixed quote styles. We
# capture link rows first (ordered), then snippet rows, and zip them.
_DDG_LINK_RE = re.compile(
    r'''<a\b[^>]*?href=["']([^"']+)["'][^>]*?class=['"]result-link['"][^>]*>(.*?)</a>''',
    re.DOTALL | re.IGNORECASE,
)
_DDG_SNIPPET_RE = re.compile(
    r'''class=['"]result-snippet['"][^>]*>(.*?)</td>''',
    re.DOTALL | re.IGNORECASE,
)


async def _search_duckduckgo(query: str, n: int) -> ToolResult:
    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT_SEC,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
        },
        follow_redirects=True,
    ) as client:
        try:
            # DDG lite accepts both GET and POST; POST is less cache-
            # exposed and avoids the result URLs leaking into access
            # logs as query strings.
            resp = await client.post(DDG_LITE_ENDPOINT, data={"q": query})
        except httpx.HTTPError as exc:
            return ToolResult(
                output=f"DuckDuckGo request failed: {type(exc).__name__}: {exc}",
                is_error=True,
            )
    if resp.status_code == 202:
        # DDG occasionally rate-limits with 202 + a "making sure you're
        # not a bot" page. Treat as soft-fail.
        return ToolResult(
            output="DuckDuckGo is rate-limiting anonymous requests. "
                   "Set BRAVE_SEARCH_API_KEY for a higher-quality keyed "
                   "provider (free at https://api.search.brave.com).",
            is_error=True,
        )
    if resp.status_code >= 400:
        return ToolResult(
            output=f"DuckDuckGo HTTP {resp.status_code}",
            is_error=True,
        )
    html = resp.text
    links = list(_DDG_LINK_RE.finditer(html))
    snippets = [m.group(1) for m in _DDG_SNIPPET_RE.finditer(html)]
    results: list[dict[str, str]] = []
    for i, m in enumerate(links):
        url = unescape(m.group(1))
        # DDG proxies external links via `/l/?uddg=<encoded>`; unwrap.
        if url.startswith("/l/?") or url.startswith("//duckduckgo.com/l/"):
            parsed = urlparse(url if url.startswith("http") else "https:" + url)
            qs = parse_qs(parsed.query)
            real = qs.get("uddg", [None])[0]
            if real:
                url = real
        title = _strip_tags(m.group(2))
        snippet = _strip_tags(snippets[i]) if i < len(snippets) else ""
        if url and title:
            results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= n:
            break
    if not results:
        return ToolResult(
            output=f"No DuckDuckGo results for: {query} "
                   "(or response shape changed — set BRAVE_SEARCH_API_KEY "
                   "for a more stable keyed provider).",
            is_error=False,
        )
    return _format_results("duckduckgo", query, results)


def _strip_tags(s: str) -> str:
    """Quick-and-dirty tag stripper for DDG snippets. We're not
    rendering — the agent just needs readable text."""
    return unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def _format_results(provider: str, query: str, results: list[dict[str, str]]) -> ToolResult:
    lines = [f"# Search via {provider}: {query}"]
    for i, r in enumerate(results, start=1):
        lines.append(f"\n{i}. {r['title']}")
        lines.append(f"   {r['url']}")
        if r.get("snippet"):
            lines.append(f"   {r['snippet']}")
    return ToolResult(output="\n".join(lines), is_error=False)


__all__ = ["WebSearchTool", "WebSearchInput"]
