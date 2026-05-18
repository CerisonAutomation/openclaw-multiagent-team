"""Tests for tools/inspector — tools, agents, relay, scheduler, server."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


# ── tools.py ────────────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_basic_python(self):
        from tools.inspector.tools import compute_metrics
        src = "def foo(x):\n    return x + 1\n"
        m = compute_metrics(src, "python")
        assert m["bytes"] == len(src.encode())
        assert m["lines_total"] == 3
        assert m["lines_code"] == 2
        assert len(m["sha256"]) == 64
        assert "ast" in m

    def test_no_code(self):
        from tools.inspector.tools import compute_metrics
        m = compute_metrics("", "python")
        assert m["bytes"] == 0
        assert m["lines_code"] == 0

    def test_javascript(self):
        from tools.inspector.tools import compute_metrics
        src = "const x = 1;\nfunction f() { return x; }\n"
        m = compute_metrics(src, "javascript")
        assert "ast" not in m
        assert m["function_count"] >= 1

    def test_comment_ratio(self):
        from tools.inspector.tools import compute_metrics
        src = "# comment\n# another\ncode = 1\n"
        m = compute_metrics(src, "python")
        assert m["lines_comment"] == 2
        assert m["comment_ratio"] > 0


class TestAnalyzePythonAST:
    def test_functions_and_classes(self):
        from tools.inspector.tools import analyze_python_ast
        src = "class Foo:\n    def bar(self):\n        pass\n    async def baz(self):\n        pass\n"
        m = analyze_python_ast(src)
        assert "bar" in m.functions
        assert "baz" in m.async_functions
        assert "Foo" in m.classes
        assert m.cyclomatic_complexity >= 1

    def test_imports(self):
        from tools.inspector.tools import analyze_python_ast
        src = "import os\nfrom pathlib import Path\n"
        m = analyze_python_ast(src)
        assert "os" in m.imports
        assert "pathlib" in m.imports

    def test_cyclomatic_complexity(self):
        from tools.inspector.tools import analyze_python_ast
        src = "def f(x):\n    if x:\n        for i in range(x):\n            pass\n    return x\n"
        m = analyze_python_ast(src)
        assert m.cyclomatic_complexity >= 3

    def test_syntax_error(self):
        from tools.inspector.tools import analyze_python_ast
        m = analyze_python_ast("def f(:")
        assert m.parse_error != ""


class TestScanSecrets:
    def test_clean_code(self):
        from tools.inspector.tools import scan_secrets
        hits = scan_secrets("x = 1\nprint('hello')\n")
        assert hits == []

    def test_aws_key(self):
        from tools.inspector.tools import scan_secrets
        src = "key = 'AKIAIOSFODNN7EXAMPLE'\n"
        hits = scan_secrets(src)
        assert len(hits) >= 1
        assert any(h.pattern_name == "AWS Access Key" for h in hits)

    def test_jwt(self):
        from tools.inspector.tools import scan_secrets
        src = "token = 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.abc123def456ghi789jkl'\n"
        hits = scan_secrets(src)
        assert any("JWT" in h.pattern_name for h in hits)

    def test_line_numbers_correct(self):
        from tools.inspector.tools import scan_secrets
        src = "x = 1\nkey = 'AKIAIOSFODNN7EXAMPLE'\ny = 2\n"
        hits = scan_secrets(src)
        aws_hits = [h for h in hits if h.pattern_name == "AWS Access Key"]
        assert aws_hits[0].line == 2


class TestRunShell:
    def test_simple_command(self):
        from tools.inspector.tools import run_shell
        r = run_shell("echo hello")
        assert r.ok
        assert "hello" in r.output

    def test_blocked_command(self):
        from tools.inspector.tools import run_shell
        r = run_shell("rm -rf / --no-preserve-root")
        assert not r.ok
        assert "blocked" in r.error.lower()

    def test_timeout(self):
        from tools.inspector.tools import run_shell
        r = run_shell("sleep 10", timeout=1)
        assert not r.ok
        assert "timed out" in r.error.lower()

    def test_nonzero_exit(self):
        from tools.inspector.tools import run_shell
        r = run_shell("exit 42", timeout=5)
        assert not r.ok
        assert r.exit_code == 42


class TestAnalyzeRepo:
    def test_self(self):
        from tools.inspector.tools import analyze_repo
        r = analyze_repo(".")
        assert r.ok
        assert r.data["total_files"] > 0
        assert "python" in r.data["languages"]

    def test_nonexistent(self):
        from tools.inspector.tools import analyze_repo
        r = analyze_repo("/this/does/not/exist")
        assert not r.ok


# ── agents.py ───────────────────────────────────────────────────────────────

class TestExtractJson:
    def test_direct_json(self):
        from tools.inspector.agents import _extract_json
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        from tools.inspector.agents import _extract_json
        assert _extract_json('```json\n{"x": 2}\n```') == {"x": 2}

    def test_embedded(self):
        from tools.inspector.agents import _extract_json
        assert _extract_json('Here is the result: {"score": 7.5} done') == {"score": 7.5}

    def test_parse_error(self):
        from tools.inspector.agents import _extract_json
        result = _extract_json("not json at all")
        assert "_parse_error" in result

    def test_array(self):
        from tools.inspector.agents import _extract_json
        assert _extract_json("[1, 2, 3]") == [1, 2, 3]


class TestSkillIndex:
    def test_list_skills(self):
        from tools.inspector.agents import skill_index
        skills = skill_index.list()
        names = [s["name"] for s in skills]
        assert "intent" in names
        assert "critic" in names
        assert "loop_runner" in names
        assert len(skills) >= 9

    def test_get_existing(self):
        from tools.inspector.agents import skill_index
        skill = skill_index.get("fix")
        assert skill is not None
        assert skill.role == "fixer"
        assert skill.output_format == "json"

    def test_get_missing(self):
        from tools.inspector.agents import skill_index
        assert skill_index.get("nonexistent_skill") is None

    def test_register_custom(self):
        from tools.inspector.agents import SkillIndex
        from tools.inspector.config import SkillDef
        idx = SkillIndex()
        custom = SkillDef("my_skill", "desc", "coder", "json", "system prompt", 0.2, 512)
        idx.register(custom)
        assert idx.get("my_skill") is not None
        names = [s["name"] for s in idx.list()]
        assert "my_skill" in names

    @pytest.mark.asyncio
    async def test_run_skill_no_provider(self):
        from tools.inspector.agents import skill_index
        # No Ollama/Jan in test env — should return error SkillResult, not raise
        result = await skill_index.run("intent", "test input")
        # Either ok (provider available) or error (not available) — must not raise
        assert result.skill == "intent"
        assert isinstance(result.ok, bool)


# ── relay.py ────────────────────────────────────────────────────────────────

class TestRelayToolSelection:
    def test_review_intent_selects_summarize_and_critic(self):
        from tools.inspector.relay import _select_tools
        intent = {"primary_intent": "review", "domain": "frontend", "language": "javascript"}
        tools = _select_tools(intent, has_source=True)
        names = [n for _, n in tools]
        assert "summarize" in names
        assert "critic" in names
        assert "secrets" in names

    def test_fix_intent(self):
        from tools.inspector.relay import _select_tools
        intent = {"primary_intent": "fix", "domain": "backend", "language": "python"}
        tools = _select_tools(intent, has_source=True)
        names = [n for _, n in tools]
        assert "fix" in names
        assert "security" in names
        assert "lint" in names
        assert "ast" in names

    def test_no_source_returns_empty(self):
        from tools.inspector.relay import _select_tools
        tools = _select_tools({"primary_intent": "review"}, has_source=False)
        assert tools == []

    def test_security_domain_adds_security_skill(self):
        from tools.inspector.relay import _select_tools
        intent = {"primary_intent": "explain", "domain": "api", "language": "python"}
        tools = _select_tools(intent, has_source=True)
        names = [n for _, n in tools]
        assert "security" in names

    def test_no_duplicates(self):
        from tools.inspector.relay import _select_tools
        intent = {"primary_intent": "review", "domain": "backend", "language": "python"}
        tools = _select_tools(intent, has_source=True)
        assert len(tools) == len(set(tools))


class TestRelaySynthesize:
    def test_basic_assembly(self):
        from tools.inspector.relay import _synthesize
        intent = {"primary_intent": "review", "domain": "backend", "language": "python",
                  "complexity": "low", "stated_goal": "Check my code", "issues_spotted": [],
                  "recommended_next": "Add tests"}
        outputs = {
            "metrics": {"lines_code": 42, "bytes": 1000, "cyclomatic_avg": 1.5,
                        "sha256": "abc123", "todo_markers": 0},
            "secrets": [],
        }
        msg = _synthesize("review my code", intent, outputs)
        assert "REVIEW" in msg
        assert "backend" in msg
        assert "42 lines" in msg
        assert "✓ clean" in msg
        assert "Add tests" in msg

    def test_secret_hit_shows_warning(self):
        from tools.inspector.relay import _synthesize
        intent = {"primary_intent": "fix", "domain": "backend", "language": "python",
                  "complexity": "low", "stated_goal": "", "issues_spotted": [],
                  "recommended_next": ""}
        outputs = {
            "secrets": [{"pattern": "AWS Access Key", "line": 5, "snippet": "AKIA...", "severity": "critical"}],
        }
        msg = _synthesize("fix", intent, outputs)
        assert "🔴" in msg
        assert "AWS Access Key" in msg

    def test_critic_output_formatted(self):
        from tools.inspector.relay import _synthesize
        intent = {"primary_intent": "review", "domain": "backend", "language": "python",
                  "complexity": "low", "stated_goal": "", "issues_spotted": [],
                  "recommended_next": ""}
        outputs = {
            "critic": {
                "correctness": 7.0, "clarity": 6.0, "structure": 8.0,
                "testability": 5.0, "security": 7.0, "performance": 7.0,
                "maintainability": 6.5, "completeness": 7.5,
                "lowest_dimension": "testability",
                "critical_fix": "Add unit tests",
                "summary": "Decent code",
            }
        }
        msg = _synthesize("review", intent, outputs)
        assert "Critique" in msg
        assert "testability" in msg


class TestRelayAgent:
    @pytest.mark.asyncio
    async def test_handle_no_provider(self):
        """Relay agent handles missing LLM provider gracefully."""
        from tools.inspector.relay import RelayAgent
        from tools.inspector.agents import SkillIndex

        # Patch the skill_index to return a canned intent result
        mock_si = MagicMock(spec=SkillIndex)
        from tools.inspector.agents import SkillResult
        mock_result = SkillResult(
            skill="intent", model_used="mock", raw="{}",
            output={"primary_intent": "review", "domain": "backend", "language": "python",
                    "complexity": "low", "stated_goal": "test", "issues_spotted": [],
                    "recommended_next": "Write tests"},
            ok=True,
        )
        mock_si.run = AsyncMock(return_value=mock_result)

        agent = RelayAgent(mock_si)
        src = "def foo(x):\n    return x\n"
        result = await agent.handle("review my code", source=src)

        assert result.intent["primary_intent"] == "review"
        assert "REVIEW" in result.message
        assert result.latency_ms >= 0
        assert len(result.tools_used) > 0

    @pytest.mark.asyncio
    async def test_history_accumulates(self):
        from tools.inspector.relay import RelayAgent
        from tools.inspector.agents import SkillIndex, SkillResult

        mock_si = MagicMock(spec=SkillIndex)
        mock_si.run = AsyncMock(return_value=SkillResult(
            skill="intent", model_used="mock", raw="{}",
            output={"primary_intent": "explain", "domain": "other", "language": "python",
                    "complexity": "low", "stated_goal": "", "issues_spotted": [],
                    "recommended_next": ""},
            ok=True,
        ))

        agent = RelayAgent(mock_si)
        await agent.handle("first message")
        await agent.handle("second message")
        history = agent.history()
        assert len(history) == 4  # 2 user + 2 assistant

    def test_clear_history(self):
        from tools.inspector.relay import RelayAgent
        from tools.inspector.agents import SkillIndex

        mock_si = MagicMock(spec=SkillIndex)
        agent = RelayAgent(mock_si)
        agent._history.append(MagicMock())
        agent.clear_history()
        assert agent.history() == []


# ── server.py (inspector) ───────────────────────────────────────────────────

@pytest.fixture
def inspector_client():
    from tools.inspector.server import app
    return TestClient(app)


class TestInspectorServer:
    def test_healthz(self, inspector_client):
        r = inspector_client.get("/healthz")
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert "ts" in data
        assert "ws_connections" in data

    def test_skills_list(self, inspector_client):
        r = inspector_client.get("/api/skills")
        assert r.status_code == 200
        skills = r.json()["skills"]
        names = [s["name"] for s in skills]
        assert "intent" in names
        assert "loop_runner" in names
        assert len(skills) >= 9

    def test_analyze_code_python(self, inspector_client):
        src = "def foo(x):\n    return x + 1\n"
        r = inspector_client.post("/api/analyze/code", json={"source": src})
        assert r.status_code == 200
        data = r.json()
        assert data["language"] == "python"
        assert data["metrics"]["lines_code"] >= 1
        assert isinstance(data["secrets"], list)
        assert data["lint"] is not None  # python gets lint

    def test_analyze_code_auto_detects_json(self, inspector_client):
        src = '{"name": "Alice", "age": 30}'
        r = inspector_client.post("/api/analyze/code", json={"source": src})
        assert r.status_code == 200
        assert r.json()["language"] == "json"

    def test_analyze_code_secret(self, inspector_client):
        src = "api_key = 'AKIAIOSFODNN7EXAMPLE'\nprint(api_key)\n"
        r = inspector_client.post("/api/analyze/code", json={"source": src})
        assert r.status_code == 200
        secrets = r.json()["secrets"]
        assert len(secrets) >= 1

    def test_analyze_repo(self, inspector_client):
        r = inspector_client.post("/api/analyze/repo", json={"path": "."})
        assert r.status_code == 200
        data = r.json()
        assert data["repo"]["total_files"] > 0
        assert "python" in data["repo"]["languages"]

    def test_analyze_shell(self, inspector_client):
        r = inspector_client.post("/api/analyze/shell", json={"cmd": "echo ok"})
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert "ok" in data["output"]

    def test_analyze_shell_blocked(self, inspector_client):
        r = inspector_client.post("/api/analyze/shell", json={"cmd": "rm -rf /"})
        assert r.status_code == 200
        assert r.json()["ok"] is False

    def test_providers_endpoint(self, inspector_client):
        r = inspector_client.get("/api/providers")
        assert r.status_code == 200
        data = r.json()
        assert "ollama" in data
        assert "jan" in data
        assert "active" in data

    def test_relay_no_source(self, inspector_client):
        """Relay with no source code returns gracefully (no tools to run)."""
        r = inspector_client.post("/api/relay", json={
            "message": "hello",
            "source": "",
            "session_id": "test_no_src",
        })
        # May fail with 500 if no LLM, but must not crash the server
        assert r.status_code in (200, 500, 503)

    def test_relay_history_empty(self, inspector_client):
        r = inspector_client.get("/api/relay/history/nonexistent_session")
        assert r.status_code == 200
        assert r.json()["history"] == []

    def test_relay_clear_history(self, inspector_client):
        r = inspector_client.delete("/api/relay/history/nonexistent_session")
        assert r.status_code == 200
        assert r.json()["existed"] is False

    def test_loop_plan_missing_provider(self, inspector_client):
        r = inspector_client.post("/api/loop/plan", json={"task": "fix failing tests"})
        # 500 expected when no LLM available; endpoint must exist and not crash
        assert r.status_code in (200, 500, 503)

    def test_schedule_list(self, inspector_client):
        r = inspector_client.get("/api/schedule")
        assert r.status_code == 200
        assert "jobs" in r.json()


# ── stop_check.py ───────────────────────────────────────────────────────────

class TestStopHook:
    """Tests for .claude/hooks/stop_check.py — run via subprocess to test exit codes."""

    HOOK = str(Path(__file__).parent.parent / ".claude" / "hooks" / "stop_check.py")

    def _run_hook(self, stdin_json: dict, env_extra: dict | None = None,
                  cwd: str | None = None) -> tuple[int, str, str]:
        import os
        import subprocess
        env = {**os.environ}
        for k in ("LOOP_PROMISE_TOKEN", "LOOP_PROMISE_FILE", "LOOP_MAX_ITER"):
            env.pop(k, None)
        if env_extra:
            env.update(env_extra)
        r = subprocess.run(
            ["python", self.HOOK],
            input=json.dumps(stdin_json),
            capture_output=True, text=True, env=env, timeout=5,
            cwd=cwd,
        )
        return r.returncode, r.stdout, r.stderr

    def test_no_loop_active(self, tmp_path):
        rc, out, err = self._run_hook({"stop_hook_active": False})
        assert rc == 0

    def test_idempotency_guard(self, tmp_path):
        rc, out, err = self._run_hook(
            {"stop_hook_active": True},
            {"LOOP_PROMISE_TOKEN": "DONE", "LOOP_PROMISE_FILE": str(tmp_path / "p.txt")},
        )
        assert rc == 0

    def test_promise_missing_blocks(self, tmp_path):
        rc, out, err = self._run_hook(
            {"stop_hook_active": False},
            {"LOOP_PROMISE_TOKEN": "DONE", "LOOP_PROMISE_FILE": str(tmp_path / "p.txt"),
             "LOOP_MAX_ITER": "10"},
            cwd=str(tmp_path),
        )
        assert rc == 2
        assert "DONE" in out

    def test_promise_present_allows(self, tmp_path):
        pf = tmp_path / "done.txt"
        pf.write_text("DONE")
        rc, out, err = self._run_hook(
            {"stop_hook_active": False},
            {"LOOP_PROMISE_TOKEN": "DONE", "LOOP_PROMISE_FILE": str(pf),
             "LOOP_MAX_ITER": "10"},
            cwd=str(tmp_path),
        )
        assert rc == 0

    def test_max_iter_cap(self, tmp_path):
        # Write iter counter at max-1 so the next call pushes it to max
        iter_file = tmp_path / ".loop_iter"
        iter_file.write_text("9")
        rc, out, err = self._run_hook(
            {"stop_hook_active": False},
            {"LOOP_PROMISE_TOKEN": "DONE", "LOOP_PROMISE_FILE": str(tmp_path / "p.txt"),
             "LOOP_MAX_ITER": "10"},
            cwd=str(tmp_path),
        )
        assert rc == 0  # max reached → allowed to stop
