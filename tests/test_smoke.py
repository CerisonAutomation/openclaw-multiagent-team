"""End-to-end smoke test using the mock provider — no API keys needed."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openclaw import gates
from openclaw.agents import classify_intent, critique, plan_architecture
from openclaw.audit import AuditLog
from openclaw.orchestrator import Orchestrator, RunConfig
from openclaw.providers import MockProvider, get_provider
from openclaw.server import app
from openclaw.tools import analyze_repo, generate_image, run_shell


# ── providers ───────────────────────────────────────────────────────────────

def test_mock_provider_chat() -> None:
    p = MockProvider()
    out = p.chat("system", "hello world")
    assert "MOCK" in out
    assert "SEAL:" in out


def test_mock_provider_intent_json() -> None:
    p = MockProvider()
    data = p.json("Classify the user's request", "build me a todo app")
    assert data["primary_intent"] in ("technical", "build", "fix", "extend", "deploy", "review", "explain")


def test_get_provider_defaults_to_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENCLAW_PROVIDER", raising=False)
    p = get_provider()
    assert p.name == "mock"


# ── agents ──────────────────────────────────────────────────────────────────

def test_classify_intent_with_mock() -> None:
    intent = classify_intent(MockProvider(), "build a Next.js todo app")
    assert intent.get("stated_goal")
    assert intent.get("primary_intent")


def test_plan_architecture_with_mock() -> None:
    arch = plan_architecture(MockProvider(), {"primary_intent": "build", "complexity": "moderate", "domain": "fullstack"})
    assert arch["framework"]
    assert arch["files_to_create"]


def test_critique_returns_scores() -> None:
    scores = critique(MockProvider(), "task", "output")
    for dim in ("clarity", "structure", "completeness"):
        assert isinstance(scores[dim], (int, float))


# ── gates ───────────────────────────────────────────────────────────────────

def test_gate_intent_clarity_pass() -> None:
    g = gates.gate_intent_clarity({"primary_intent": "build", "stated_goal": "a thing"})
    assert g.passed


def test_gate_intent_clarity_fail_on_parse_error() -> None:
    g = gates.gate_intent_clarity({"_parse_error": True})
    assert not g.passed


def test_gate_no_placeholders_catches_todo() -> None:
    g = gates.gate_no_placeholders("here is a TODO and a placeholder")
    assert not g.passed


def test_gate_quality_threshold() -> None:
    scores = {"clarity": 8.0, "structure": 7.5, "completeness": 6.0}
    g = gates.gate_quality_threshold(scores, threshold=7.5)
    assert not g.passed   # completeness below
    g2 = gates.gate_quality_threshold(scores, threshold=5.0)
    assert g2.passed


# ── tools ───────────────────────────────────────────────────────────────────

def test_run_shell_simple() -> None:
    res = run_shell("echo hello")
    assert res.ok
    assert "hello" in res.output


def test_run_shell_blocks_dangerous() -> None:
    res = run_shell("rm -rf /")
    assert not res.ok
    assert "blocked" in res.error.lower()


def test_analyze_repo_self() -> None:
    summary = analyze_repo(".")
    assert "python" in summary["languages"]
    assert "pyproject.toml" in summary["top_level_entries"]


def test_generate_image_placeholder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = tmp_path / "test.png"
    res = generate_image("a red circle on blue background", out)
    assert res.ok
    svg = (tmp_path / "test.svg")
    assert svg.exists()
    assert "<svg" in svg.read_text()


# ── orchestrator ────────────────────────────────────────────────────────────

def test_orchestrator_build_end_to_end(tmp_path: Path) -> None:
    cfg = RunConfig(workspace=str(tmp_path / "ws"), max_critique_loops=1, verbose=False)
    orch = Orchestrator(provider=MockProvider(), config=cfg)
    result = orch.run("build me a Next.js todo app with a landing page")
    assert result.ok
    assert "SEAL:" in result.seal
    assert result.audit.intent
    assert result.audit.architecture
    assert "coder" in result.audit.active_agents
    # Files should have been written for each architect-proposed file
    for fs in result.audit.architecture.get("files_to_create", []):
        assert (Path(result.workspace) / fs["path"]).exists()


def test_orchestrator_fix_existing_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "package.json").write_text('{"name":"x","dependencies":{"react":"19"}}')
    cfg = RunConfig(workspace=str(repo), max_critique_loops=1, verbose=False)
    orch = Orchestrator(provider=MockProvider(), config=cfg)
    result = orch.run("the build is broken", repo_path=str(repo))
    assert result.audit.task
    # repo analysis recorded
    phases = [p["phase"] for p in result.audit.phases]
    assert "repo_analysis" in phases


def test_audit_log_serializes(tmp_path: Path) -> None:
    log = AuditLog(task="x")
    log.fire("intent")
    log.add_tool_call("shell", {"cmd": "echo"}, True, "hello")
    log.add_phase("PHASE 1", {"result": "ok"})
    p = log.write(tmp_path / "audit.json")
    loaded = json.loads(p.read_text())
    assert loaded["task"] == "x"
    assert "intent" in loaded["active_agents"]


# ── FastAPI server ──────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("OPENCLAW_PROVIDER", "mock")
    return TestClient(app)


def test_server_root(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["name"] == "openclaw"


def test_server_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"]


def test_server_analyze(client: TestClient) -> None:
    r = client.post("/api/analyze", json={"repo_path": "."})
    assert r.status_code == 200
    assert "python" in r.json()["languages"]


def test_server_build(client: TestClient, tmp_path: Path) -> None:
    body = {
        "task": "build me a tiny landing page",
        "workspace": str(tmp_path / "ws"),
        "iterations": 1,
        "provider": "mock",
    }
    r = client.post("/api/build", json=body)
    assert r.status_code in (200, 422)   # 422 only if all gates trip in mock mode
    payload = r.json()
    assert "seal" in payload
    assert "audit" in payload
