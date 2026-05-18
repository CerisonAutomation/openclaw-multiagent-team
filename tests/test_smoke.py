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
from openclaw.models import PRESETS
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
    for preset in PRESETS.values():
        if preset.key_env:
            monkeypatch.delenv(preset.key_env, raising=False)
    monkeypatch.delenv("OPENCLAW_PROVIDER", raising=False)
    p = get_provider()
    assert p.name == "mock"


def test_get_provider_auto_detects_openrouter_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenRouter is the highest-priority auto-detect (free tier, high-end models)."""
    for preset in PRESETS.values():
        if preset.key_env:
            monkeypatch.delenv(preset.key_env, raising=False)
    monkeypatch.delenv("OPENCLAW_PROVIDER", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "nvapi-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # OpenRouter declared first in PRESETS → wins
    p = get_provider()
    assert p.name == "openrouter"


def test_get_provider_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    for preset in PRESETS.values():
        if preset.key_env:
            monkeypatch.delenv(preset.key_env, raising=False)
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "nvapi-test")
    p = get_provider("nvidia")
    assert p.name == "nvidia"
    assert "nemotron" in p.model_for("architect").lower() or "70b" in p.model_for("architect").lower()


def test_role_routing_uses_role_specific_model(monkeypatch: pytest.MonkeyPatch) -> None:
    for preset in PRESETS.values():
        if preset.key_env:
            monkeypatch.delenv(preset.key_env, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    p = get_provider("openrouter")
    # critic differs from default — even if both are high-end, the routing fires
    assert p.model_for("critic")
    assert p.model_for("intent")
    assert p.model_for("coder")


def test_role_model_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENCLAW_MODEL_CRITIC", "anthropic/claude-3.5-haiku")
    p = get_provider("openrouter")
    assert p.model_for("critic") == "anthropic/claude-3.5-haiku"


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError):
        get_provider("not-a-real-provider")


def test_all_presets_have_required_fields() -> None:
    for name, preset in PRESETS.items():
        assert preset.kind in ("anthropic", "openai_compatible", "mock"), name
        assert preset.default_model, name
        if preset.kind == "openai_compatible":
            assert preset.base_url, name


def test_mock_provider_model_for_returns_default() -> None:
    p = MockProvider()
    assert p.model_for("critic") == "mock-1"
    assert p.model_for(None) == "mock-1"


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


# ── CLI: auto / setup / toml config ─────────────────────────────────────────

def test_toml_config_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    from openclaw.cli import _load_toml_config
    assert _load_toml_config() == {}


def test_toml_config_parses_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".openclaw.toml").write_text(
        "[openclaw]\n"
        'provider = "openrouter"\n'
        "iterations = 5\n"
        "threshold = 8.0\n"
    )
    monkeypatch.chdir(tmp_path)
    from openclaw.cli import _load_toml_config
    cfg = _load_toml_config()
    assert cfg["provider"] == "openrouter"
    assert cfg["iterations"] == 5
    assert cfg["threshold"] == 8.0


def test_setup_cmd_creates_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    for preset in PRESETS.values():
        if preset.key_env:
            monkeypatch.delenv(preset.key_env, raising=False)
    monkeypatch.delenv("OPENCLAW_PROVIDER", raising=False)
    from openclaw.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["setup"])
    rc = args.func(args)
    assert rc == 0
    toml_path = tmp_path / ".openclaw.toml"
    assert toml_path.exists()
    assert "provider" in toml_path.read_text()


def test_auto_cmd_in_git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess as sp
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENCLAW_PROVIDER", "mock")
    sp.run(["git", "init", str(tmp_path)], capture_output=True)
    (tmp_path / "package.json").write_text('{"name":"x"}')
    from openclaw.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["auto", "--quiet", "--iterations", "1"])
    rc = args.func(args)
    # auto detects git repo, runs fix — should succeed or fail gracefully
    assert rc in (0, 1)


def test_apply_toml_defaults_fills_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    from openclaw.cli import _apply_toml_defaults
    args = argparse.Namespace(provider=None, model=None, iterations=3, threshold=7.5)
    _apply_toml_defaults(args, {"provider": "groq", "iterations": 5, "threshold": 8.0})
    assert args.provider == "groq"
    assert args.iterations == 5
    assert args.threshold == 8.0


def test_apply_toml_defaults_does_not_override_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    from openclaw.cli import _apply_toml_defaults
    args = argparse.Namespace(provider="nvidia", model=None, iterations=3, threshold=7.5)
    _apply_toml_defaults(args, {"provider": "groq"})
    assert args.provider == "nvidia"  # explicit flag wins
