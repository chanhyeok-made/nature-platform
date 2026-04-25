"""Tests for built-in tools."""

import os

import pytest

from nature.protocols.tool import ToolContext
from nature.tools.builtin.bash import BashTool
from nature.tools.builtin.read import ReadTool
from nature.tools.builtin.write import WriteTool
from nature.tools.builtin.edit import EditTool
from nature.tools.builtin.glob_tool import GlobTool
from nature.tools.builtin.grep_tool import GrepTool
from nature.tools.registry import get_default_tools


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(cwd=str(tmp_path), project_root=str(tmp_path))


@pytest.fixture
def sample_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n")
    return str(f)


class TestBashTool:
    @pytest.mark.asyncio
    async def test_echo(self, ctx):
        tool = BashTool()
        result = await tool.execute({"command": "echo hello"}, ctx)
        assert "hello" in result.output
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_exit_code(self, ctx):
        tool = BashTool()
        result = await tool.execute({"command": "false"}, ctx)
        assert result.is_error
        assert "Exit code: 1" in result.output

    @pytest.mark.asyncio
    async def test_blocked_command(self, ctx):
        tool = BashTool()
        err = await tool.validate_input({"command": "rm -rf /"}, ctx)
        assert err is not None
        assert "Blocked" in err

    @pytest.mark.asyncio
    async def test_read_only_detection(self):
        tool = BashTool()
        assert tool.is_read_only({"command": "ls -la"}) is True
        assert tool.is_read_only({"command": "rm file"}) is False
        assert tool.is_read_only({"command": "git status"}) is True

    @pytest.mark.asyncio
    async def test_timeout(self, ctx):
        tool = BashTool()
        result = await tool.execute({"command": "sleep 10", "timeout": 1}, ctx)
        assert result.is_error
        assert "timed out" in result.output


class TestReadTool:
    @pytest.mark.asyncio
    async def test_read_file(self, ctx, sample_file):
        tool = ReadTool()
        result = await tool.execute({"file_path": sample_file}, ctx)
        assert "line 1" in result.output
        assert "line 5" in result.output
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_read_with_offset(self, ctx, sample_file):
        tool = ReadTool()
        result = await tool.execute({"file_path": sample_file, "offset": 2, "limit": 2}, ctx)
        assert "3\tline 3" in result.output
        assert "4\tline 4" in result.output
        assert "line 1" not in result.output

    @pytest.mark.asyncio
    async def test_file_not_found(self, ctx):
        tool = ReadTool()
        result = await tool.execute({"file_path": "/nonexistent/file.txt"}, ctx)
        assert result.is_error
        assert "not found" in result.output.lower()

    @pytest.mark.asyncio
    async def test_concurrency_safe(self):
        tool = ReadTool()
        assert tool.is_concurrency_safe({"file_path": "/any"}) is True


class TestWriteTool:
    @pytest.mark.asyncio
    async def test_write_new_file(self, ctx, tmp_path):
        tool = WriteTool()
        path = str(tmp_path / "new.txt")
        result = await tool.execute({"file_path": path, "content": "hello world"}, ctx)
        assert not result.is_error
        assert os.path.exists(path)
        assert open(path).read() == "hello world"

    @pytest.mark.asyncio
    async def test_write_creates_dirs(self, ctx, tmp_path):
        tool = WriteTool()
        path = str(tmp_path / "sub" / "dir" / "file.txt")
        result = await tool.execute({"file_path": path, "content": "deep"}, ctx)
        assert not result.is_error
        assert open(path).read() == "deep"

    @pytest.mark.asyncio
    async def test_read_only_blocked(self, tmp_path):
        tool = WriteTool()
        ctx = ToolContext(cwd=str(tmp_path), is_read_only=True)
        err = await tool.validate_input({"file_path": "/tmp/x", "content": "y"}, ctx)
        assert err is not None


class TestEditTool:
    @pytest.mark.asyncio
    async def test_edit_replace(self, ctx, sample_file):
        tool = EditTool()
        result = await tool.execute({
            "file_path": sample_file,
            "old_string": "line 3",
            "new_string": "LINE THREE",
        }, ctx)
        assert not result.is_error
        content = open(sample_file).read()
        assert "LINE THREE" in content
        assert "line 3" not in content

    @pytest.mark.asyncio
    async def test_edit_not_found(self, ctx, sample_file):
        tool = EditTool()
        result = await tool.execute({
            "file_path": sample_file,
            "old_string": "nonexistent",
            "new_string": "replacement",
        }, ctx)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_edit_ambiguous(self, ctx, tmp_path):
        f = tmp_path / "dup.txt"
        f.write_text("aaa\nbbb\naaa\n")
        tool = EditTool()
        result = await tool.execute({
            "file_path": str(f),
            "old_string": "aaa",
            "new_string": "ccc",
        }, ctx)
        assert result.is_error
        assert "2 times" in result.output

    @pytest.mark.asyncio
    async def test_edit_replace_all(self, ctx, tmp_path):
        f = tmp_path / "dup.txt"
        f.write_text("aaa\nbbb\naaa\n")
        tool = EditTool()
        result = await tool.execute({
            "file_path": str(f),
            "old_string": "aaa",
            "new_string": "ccc",
            "replace_all": True,
        }, ctx)
        assert not result.is_error
        assert f.read_text() == "ccc\nbbb\nccc\n"


class TestGlobTool:
    @pytest.mark.asyncio
    async def test_glob_py_files(self, ctx, tmp_path):
        (tmp_path / "a.py").write_text("pass")
        (tmp_path / "b.txt").write_text("hello")
        (tmp_path / "c.py").write_text("pass")

        tool = GlobTool()
        result = await tool.execute({"pattern": "*.py"}, ctx)
        assert "a.py" in result.output
        assert "c.py" in result.output
        assert "b.txt" not in result.output

    @pytest.mark.asyncio
    async def test_glob_no_matches(self, ctx):
        tool = GlobTool()
        result = await tool.execute({"pattern": "*.nonexistent"}, ctx)
        assert "No files" in result.output

    @pytest.mark.asyncio
    async def test_glob_rejects_path_outside_cwd(self, ctx):
        """Regression: session 8ca51065 — Sonnet issued
        Glob(pattern='**/*', path='/') and the framework happily
        started walking the entire macOS filesystem. The boundary
        check must reject any path that resolves outside cwd."""
        tool = GlobTool()
        result = await tool.execute({"pattern": "*.py", "path": "/"}, ctx)
        assert result.is_error
        assert "outside the project's working directory" in result.output

    @pytest.mark.asyncio
    async def test_glob_rejects_parent_escape(self, ctx, tmp_path):
        """`..` traversal that escapes cwd is rejected."""
        tool = GlobTool()
        result = await tool.execute(
            {"pattern": "*.py", "path": "../.."}, ctx,
        )
        assert result.is_error
        assert "outside the project's working directory" in result.output

    @pytest.mark.asyncio
    async def test_glob_rejects_absolute_pattern(self, ctx):
        """Absolute pattern bypasses path boundary via os.path.join's
        absolute-path-wins rule. Reject up-front."""
        tool = GlobTool()
        result = await tool.execute({"pattern": "/etc/*"}, ctx)
        assert result.is_error
        assert "must be relative" in result.output

    @pytest.mark.asyncio
    async def test_glob_allows_subpath_inside_cwd(self, ctx, tmp_path):
        """A relative path that resolves inside cwd is fine."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.py").write_text("pass")
        tool = GlobTool()
        result = await tool.execute({"pattern": "*.py", "path": "sub"}, ctx)
        assert not result.is_error
        assert "a.py" in result.output


class TestGrepTool:
    @pytest.mark.asyncio
    async def test_grep_pattern(self, ctx, sample_file):
        tool = GrepTool()
        result = await tool.execute({"pattern": "line [35]", "path": sample_file}, ctx)
        assert "line 3" in result.output
        assert "line 5" in result.output

    @pytest.mark.asyncio
    async def test_grep_no_match(self, ctx, sample_file):
        tool = GrepTool()
        result = await tool.execute({"pattern": "zzzzz", "path": sample_file}, ctx)
        assert "No matches" in result.output

    @pytest.mark.asyncio
    async def test_grep_case_insensitive(self, ctx, tmp_path):
        f = tmp_path / "case.txt"
        f.write_text("Hello World\nhello world\nHELLO WORLD\n")
        tool = GrepTool()
        result = await tool.execute({
            "pattern": "hello",
            "path": str(f),
            "case_insensitive": True,
        }, ctx)
        # Should match all 3 lines
        assert result.output.count("ello") >= 2


class TestRegistry:
    def test_get_default_tools(self):
        tools = get_default_tools()
        names = {t.name for t in tools}
        # Agent is NOT in the default set — the Frame AreaManager registers
        # `nature.frame.agent_tool.AgentTool` separately based on each role's
        # `allowed_tools`.
        assert names == {"Bash", "Read", "Write", "Edit", "Glob", "Grep", "TodoWrite"}

    def test_tool_definitions(self):
        tools = get_default_tools()
        for tool in tools:
            defn = tool.to_definition()
            assert defn.name == tool.name
            assert defn.description
            assert isinstance(defn.input_schema, dict)
