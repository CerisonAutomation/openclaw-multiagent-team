"""
FastAPI server with WebSocket, Ollama/Jan routing, skill execution, cron API.

Run:
    python -m tools.inspector.server
  or:
    uvicorn tools.inspector.server:app --host 0.0.0.0 --port 8765 --reload

Then open tools/inspector.html in a browser.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .agents import SkillResult, skill_index
from .config import SKILL_INDEX, get_config
from .relay import RelayAgent
from .scheduler import Event, FileWatcher, bus, heartbeat, scheduler
from .tools import (
    analyze_repo,
    compute_metrics,
    git_summary,
    lint_python,
    run_shell,
    scan_secrets,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── WebSocket broadcast manager ───────────────────────────────────────────────

class WSManager:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._connections = [c for c in self._connections if c is not ws]

    async def broadcast(self, data: dict[str, Any]) -> None:
        payload = json.dumps(data, default=str)
        for ws in list(self._connections):
            try:
                await ws.send_text(payload)
            except Exception:
                self.disconnect(ws)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


ws_manager = WSManager()


async def _bus_to_ws(event: Event) -> None:
    await ws_manager.broadcast({"event": event.kind, "ts": event.ts, **event.payload})


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    bus.subscribe(_bus_to_ws)
    heartbeat.start()
    scheduler.start()
    logger.info("inspector server ready on %s:%d", cfg.server_host, cfg.server_port)
    yield
    heartbeat.stop()
    scheduler.stop()
    bus.unsubscribe(_bus_to_ws)


app = FastAPI(
    title="openclaw inspector",
    description="Autodetect · Compile · Ollama agents · Cron · Real-time",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve inspector.html from the parent tools/ directory
_TOOLS_DIR = Path(__file__).parent.parent
_UI_PATH = _TOOLS_DIR / "inspector.html"


# ── Routes: meta ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    if _UI_PATH.exists():
        return FileResponse(_UI_PATH, media_type="text/html")
    return HTMLResponse("<p>inspector.html not found — copy it to tools/inspector.html</p>")


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "ts": time.time(),
        "ws_connections": ws_manager.connection_count,
        "heartbeat": heartbeat.last_status,
    }


# ── Routes: Ollama / Jan provider ─────────────────────────────────────────────

@app.get("/api/providers")
async def get_providers():
    ollama = skill_index.ollama
    status = await ollama.status()
    jan = await _jan_status()
    return {
        "ollama": {
            "available": status.available,
            "base": get_config().ollama_base,
            "models": status.models,
            "error": status.error,
        },
        "jan": jan,
        "active": "ollama" if status.available else ("jan" if jan["available"] else "none"),
    }


@app.get("/api/providers/models")
async def list_models():
    models = await skill_index.ollama.list_models()
    jan_models = await _jan_models()
    return {"ollama": models, "jan": jan_models}


async def _jan_status() -> dict[str, Any]:
    import httpx as _httpx
    jan_base = get_config().jan_base
    try:
        async with _httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{jan_base}/models")
            if r.status_code == 200:
                data = r.json()
                names = [m["id"] for m in data.get("data", [])]
                return {"available": True, "base": jan_base, "models": names, "error": ""}
    except Exception as e:
        return {"available": False, "base": jan_base, "models": [], "error": str(e)}
    return {"available": False, "base": jan_base, "models": [], "error": "HTTP error"}


async def _jan_models() -> list[dict]:
    import httpx as _httpx
    jan_base = get_config().jan_base
    try:
        async with _httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{jan_base}/models")
            if r.status_code == 200:
                return r.json().get("data", [])
    except Exception:
        pass
    return []


async def _jan_generate(prompt: str, system: str, model: str,
                        temperature: float, max_tokens: int) -> str:
    """OpenAI-compatible chat call to Jan."""
    import httpx as _httpx
    cfg = get_config()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    async with _httpx.AsyncClient(timeout=cfg.ollama_timeout) as c:
        r = await c.post(f"{cfg.jan_base}/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# ── Routes: code analysis ─────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    source: str
    language: str = ""          # leave blank for auto-detect
    filename: str = ""


class AnalyzeResponse(BaseModel):
    language: str
    metrics: dict[str, Any]
    secrets: list[dict]
    lint: dict[str, Any] | None = None


@app.post("/api/analyze/code")
async def analyze_code(req: AnalyzeRequest) -> AnalyzeResponse:
    lang = req.language or _detect_language(req.source)
    metrics = compute_metrics(req.source, lang)
    hits = scan_secrets(req.source)
    lint_result = None
    if lang == "python":
        lr = lint_python(req.source)
        lint_result = lr.data if lr.ok else {"error": lr.error}
    return AnalyzeResponse(
        language=lang,
        metrics=metrics,
        secrets=[{"pattern": h.pattern_name, "line": h.line, "snippet": h.snippet, "severity": h.severity} for h in hits],
        lint=lint_result,
    )


@app.post("/api/analyze/repo")
async def analyze_repo_route(body: dict[str, str]):
    path = body.get("path", ".")
    result = analyze_repo(path)
    if not result.ok:
        raise HTTPException(status_code=400, detail=result.error)
    git = git_summary(path)
    return {"repo": result.data, "git": git.data if git.ok else {"error": git.error}}


@app.post("/api/analyze/shell")
async def run_shell_route(body: dict[str, str]):
    cmd = body.get("cmd", "").strip()
    if not cmd:
        raise HTTPException(status_code=400, detail="cmd is required")
    cwd = body.get("cwd", ".")
    timeout = int(body.get("timeout", "30"))
    res = run_shell(cmd, cwd=cwd, timeout=timeout)
    return {"ok": res.ok, "output": res.output, "error": res.error, "exit_code": res.exit_code}


# ── Routes: skills / agents ───────────────────────────────────────────────────

@app.get("/api/skills")
async def list_skills():
    return {"skills": skill_index.list()}


class RunSkillRequest(BaseModel):
    skill: str
    source: str
    model: str | None = None
    provider: str = "auto"       # "ollama" | "jan" | "auto"
    extra_context: str = ""


@app.post("/api/skills/run")
async def run_skill(req: RunSkillRequest):
    provider = req.provider
    skill_def = skill_index.get(req.skill)
    if skill_def is None:
        raise HTTPException(status_code=404, detail=f"skill not found: {req.skill}")

    # Provider routing: try Ollama, fall back to Jan if needed
    if provider == "auto":
        st = await skill_index.ollama.status()
        provider = "ollama" if st.available else "jan"

    if provider == "jan":
        jan_st = await _jan_status()
        if not jan_st["available"]:
            raise HTTPException(status_code=503, detail="Neither Ollama nor Jan is available")
        # Pick first Jan model or use requested
        model = req.model or (jan_st["models"][0] if jan_st["models"] else "default")
        try:
            raw = await _jan_generate(
                prompt=f"{req.extra_context}\n\n{req.source}".strip() if req.extra_context else req.source,
                system=skill_def.system_prompt,
                model=model,
                temperature=skill_def.temperature,
                max_tokens=skill_def.max_tokens,
            )
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Jan error: {e}")
        from .agents import _extract_json
        output = _extract_json(raw) if skill_def.output_format == "json" else raw.strip()
        result = SkillResult(skill=req.skill, model_used=f"jan:{model}", output=output, raw=raw)
    else:
        result = await skill_index.run(req.skill, req.source, model=req.model,
                                        extra_context=req.extra_context)

    if not result.ok:
        raise HTTPException(status_code=500, detail=result.error)
    return {
        "skill": result.skill,
        "model": result.model_used,
        "output": result.output,
        "provider": provider,
    }


class RunPipelineRequest(BaseModel):
    source: str
    skills: list[str] = Field(default_factory=list)
    model: str | None = None
    provider: str = "auto"


@app.post("/api/skills/pipeline")
async def run_pipeline(req: RunPipelineRequest):
    skills = req.skills or list(SKILL_INDEX.keys())
    out: dict[str, Any] = {}
    for skill_name in skills:
        result = await skill_index.run(skill_name, req.source, model=req.model)
        out[skill_name] = {
            "ok": result.ok,
            "output": result.output,
            "model": result.model_used,
            "error": result.error,
        }
    return {"results": out, "skills_run": len(out)}


# ── Routes: direct Ollama/Jan chat ────────────────────────────────────────────

class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    model: str | None = None
    provider: str = "auto"
    temperature: float = 0.3
    max_tokens: int = 2048


@app.post("/api/chat")
async def chat(req: ChatRequest):
    provider = req.provider
    if provider == "auto":
        st = await skill_index.ollama.status()
        provider = "ollama" if st.available else "jan"

    if provider == "jan":
        jan_st = await _jan_status()
        if not jan_st["available"]:
            raise HTTPException(status_code=503, detail="Jan not available")
        model = req.model or (jan_st["models"][0] if jan_st["models"] else "default")
        sys_msgs = [m for m in req.messages if m.get("role") == "system"]
        system = sys_msgs[0]["content"] if sys_msgs else ""
        user_msgs = [m for m in req.messages if m.get("role") != "system"]
        prompt = user_msgs[-1]["content"] if user_msgs else ""
        content = await _jan_generate(prompt, system, model, req.temperature, req.max_tokens)
        return {"content": content, "model": f"jan:{model}", "provider": "jan"}

    content = await skill_index.ollama.chat(
        messages=req.messages, model=req.model,
        temperature=req.temperature, max_tokens=req.max_tokens,
    )
    return {"content": content, "provider": "ollama"}


# ── Routes: loop planner ──────────────────────────────────────────────────────

class LoopPlanRequest(BaseModel):
    task: str
    criteria: str = ""
    model: str | None = None


@app.post("/api/loop/plan")
async def loop_plan(req: LoopPlanRequest):
    """Run the loop_runner skill to produce a bounded-loop specification."""
    source = req.task
    if req.criteria:
        source = f"Task: {req.task}\n\nAcceptance criteria: {req.criteria}"
    result = await skill_index.run("loop_runner", source, model=req.model)
    if not result.ok:
        raise HTTPException(status_code=500, detail=result.error)
    return {
        "plan": result.output,
        "model": result.model_used,
        "cli_hint": (
            f"python tools/loop.py --task \"{req.task}\" "
            f"--promise {result.output.get('completion_token', 'LOOP_DONE') if isinstance(result.output, dict) else 'LOOP_DONE'} "
            f"--max-iter {result.output.get('recommended_max_iter', 20) if isinstance(result.output, dict) else 20}"
        ),
    }


# ── Routes: relay agent ───────────────────────────────────────────────────────

# One RelayAgent per session_id; sessions are in-memory (evicted on restart).
_relay_sessions: dict[str, RelayAgent] = {}


class RelayRequest(BaseModel):
    message: str                   # natural language input from the user
    source: str = ""               # optional code / text to analyse
    session_id: str = "default"    # keep history across calls with the same id
    model: str | None = None       # override model (None = auto-select)


@app.post("/api/relay")
async def relay(req: RelayRequest):
    """
    Unified entry point — classify intent, auto-select tools, execute, synthesize.

    The relay agent sits between the user and every tool/skill:
      1. Runs the `intent` skill to understand what the user wants.
      2. Selects code-analysis tools (metrics, secrets, lint, AST) and LLM skills.
      3. Executes them concurrently where possible.
      4. Returns a structured response with per-tool outputs and a synthesized message.

    Use `session_id` to maintain conversation history across multiple calls.
    """
    if req.session_id not in _relay_sessions:
        _relay_sessions[req.session_id] = RelayAgent(skill_index)

    agent = _relay_sessions[req.session_id]
    result = await agent.handle(req.message, source=req.source, model=req.model)
    return result.to_dict()


@app.get("/api/relay/history/{session_id}")
async def relay_history(session_id: str):
    """Return the message history for a relay session."""
    agent = _relay_sessions.get(session_id)
    if agent is None:
        return {"session_id": session_id, "history": []}
    return {"session_id": session_id, "history": agent.history()}


@app.delete("/api/relay/history/{session_id}")
async def relay_clear_history(session_id: str):
    """Clear the message history for a relay session."""
    agent = _relay_sessions.pop(session_id, None)
    return {"cleared": session_id, "existed": agent is not None}


# ── Routes: scheduler ─────────────────────────────────────────────────────────

@app.get("/api/schedule")
async def list_schedule():
    return {"jobs": scheduler.list()}


class AddJobRequest(BaseModel):
    name: str
    skill: str
    source: str
    interval_s: int = 300


@app.post("/api/schedule/add")
async def add_job(req: AddJobRequest):
    async def _job() -> str:
        r = await skill_index.run(req.skill, req.source)
        await ws_manager.broadcast({
            "event": "scheduled_result",
            "job": req.name,
            "skill": req.skill,
            "output": r.output,
            "model": r.model_used,
            "ok": r.ok,
        })
        return str(r.output)[:200]

    scheduler.add(req.name, _job, req.interval_s)
    return {"added": req.name, "interval_s": req.interval_s}


@app.delete("/api/schedule/{name}")
async def remove_job(name: str):
    removed = scheduler.remove(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"job not found: {name}")
    return {"removed": name}


class WatchRequest(BaseModel):
    path: str
    interval_s: int = 5


_watchers: dict[str, FileWatcher] = {}


@app.post("/api/watch")
async def start_watch(req: WatchRequest):
    if req.path in _watchers:
        return {"watching": req.path, "status": "already_watching"}
    watcher = FileWatcher(req.path, req.interval_s)
    watcher.start()
    _watchers[req.path] = watcher
    return {"watching": req.path, "interval_s": req.interval_s}


@app.delete("/api/watch")
async def stop_watch(body: dict[str, str]):
    path = body.get("path", "")
    watcher = _watchers.pop(path, None)
    if watcher:
        watcher.stop()
    return {"stopped": path}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    logger.info("WS connected — total: %d", ws_manager.connection_count)
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"error": "invalid JSON"}))
                continue
            # Handle ping
            if msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong", "ts": time.time()}))
            # Handle inline skill run via WS
            elif msg.get("type") == "run_skill":
                result = await skill_index.run(
                    msg["skill"], msg.get("source", ""), model=msg.get("model")
                )
                await ws.send_text(json.dumps({
                    "type": "skill_result",
                    "skill": result.skill,
                    "output": result.output,
                    "model": result.model_used,
                    "ok": result.ok,
                }, default=str))
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
        logger.info("WS disconnected — total: %d", ws_manager.connection_count)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_language(src: str) -> str:
    import re as _re

    signals: dict[str, list[tuple[str, int]]] = {
        "json":       [(_r, w) for _r, w in [
            (r"^\s*[\{\[]", 3), (r"^\s*[\{\[][\s\S]*[\}\]]\s*$", 4)]],
        "python":     [(_r, w) for _r, w in [
            (r"^\s*def\s+\w+\s*\(", 4), (r"^\s*class\s+\w+\s*[:\(]", 3),
            (r"^\s*import\s+\w+", 2), (r"\bself\b", 2)]],
        "typescript": [(_r, w) for _r, w in [
            (r"\binterface\s+\w+\s*\{", 4), (r"\btype\s+\w+\s*=", 3),
            (r":\s*(string|number|boolean|any|void)\b", 2)]],
        "javascript": [(_r, w) for _r, w in [
            (r"\b(const|let|var)\s+\w+\s*=", 2), (r"\bfunction\s*\w*\s*\(", 2),
            (r"=>\s*\{", 2)]],
        "html":       [(_r, w) for _r, w in [
            (r"<!doctype\s+html", 5), (r"<html[\s>]", 4)]],
        "css":        [(_r, w) for _r, w in [
            (r"[.#]?[\w-]+\s*\{[\s\S]*?\}", 3), (r"@(media|keyframes)\b", 3)]],
        "yaml":       [(_r, w) for _r, w in [(r"^---", 4), (r"^\s*[\w-]+:\s", 2)]],
        "shell":      [(_r, w) for _r, w in [
            (r"^#!\/(usr\/)?bin\/(env\s+)?(bash|sh|zsh)", 5),
            (r"\becho\s+", 2), (r"\bexport\s+\w+=", 2)]],
    }
    scores: dict[str, int] = {lang: 0 for lang in signals}
    for lang, pats in signals.items():
        for pat, weight in pats:
            if _re.search(pat, src, _re.MULTILINE):
                scores[lang] += weight
    if scores.get("json", 0) > 0:
        try:
            json.loads(src)
        except Exception:
            scores["json"] = max(0, scores["json"] - 3)
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "unknown"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    cfg = get_config()
    uvicorn.run("tools.inspector.server:app", host=cfg.server_host,
                port=cfg.server_port, reload=True, log_level="info")
