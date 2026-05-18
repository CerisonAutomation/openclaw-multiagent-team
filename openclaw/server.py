"""FastAPI server — same orchestrator, exposed over HTTP.

    POST /api/build      {"task": "...", "deploy": false, ...}
    POST /api/fix        multipart: zip file + form field 'task'
    POST /api/analyze    {"repo_path": "/path/to/repo"}
    GET  /api/audit/<id> fetch a past run's audit log

Start with:
    openclaw serve
or:
    uvicorn openclaw.server:app --reload
"""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from openclaw import __version__
from openclaw.orchestrator import Orchestrator, RunConfig
from openclaw.providers import get_provider
from openclaw.tools import analyze_repo

app = FastAPI(
    title="openclaw",
    version=__version__,
    description="Autonomous multi-agent app builder — POST a prompt or a repo, get a built/fixed/deployed result.",
)

AUDIT_DIR = Path(os.environ.get("OPENCLAW_AUDIT_DIR", "./.openclaw_audits")).resolve()
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


class BuildRequest(BaseModel):
    task: str = Field(..., description="What to build")
    workspace: str = Field("./openclaw_workspace", description="Where to write files")
    deploy: bool = False
    deploy_prod: bool = False
    iterations: int = 3
    threshold: float = 7.5
    provider: str | None = None
    model: str | None = None
    image_assets: list[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    repo_path: str


@app.get("/")
def root() -> dict:
    return {
        "name": "openclaw",
        "version": __version__,
        "endpoints": ["/api/build", "/api/fix", "/api/analyze", "/api/audit/{session_id}"],
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "version": __version__}


@app.post("/api/analyze")
def api_analyze(req: AnalyzeRequest) -> dict:
    summary = analyze_repo(req.repo_path)
    if summary.get("error"):
        raise HTTPException(404, summary["error"])
    return summary


@app.post("/api/build")
def api_build(req: BuildRequest) -> JSONResponse:
    cfg = RunConfig(
        workspace=req.workspace,
        max_critique_loops=req.iterations,
        quality_threshold=req.threshold,
        deploy=req.deploy,
        deploy_prod=req.deploy_prod,
        image_assets=req.image_assets,
        verbose=False,
    )
    orch = Orchestrator(provider=get_provider(req.provider, req.model), config=cfg)
    result = orch.run(req.task)
    _persist_audit(result.audit.session_id, result.audit.to_dict())
    payload = result.to_dict()
    payload["final_output"] = result.final_output
    return JSONResponse(payload, status_code=200 if result.ok else 422)


@app.post("/api/fix")
async def api_fix(
    task: str = Form(""),
    repo_path: str = Form(""),
    file: UploadFile | None = File(None),
    deploy: bool = Form(False),
    iterations: int = Form(3),
    threshold: float = Form(7.5),
    provider: str | None = Form(None),
    model: str | None = Form(None),
) -> JSONResponse:
    workspace: str
    if file is not None:
        # Accept a zip upload of the repo
        tmp = Path(tempfile.mkdtemp(prefix="openclaw_repo_"))
        data = await file.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(tmp)
        # If the zip contained a single top-level dir, descend into it
        entries = [p for p in tmp.iterdir() if not p.name.startswith("__MACOSX")]
        workspace = str(entries[0]) if len(entries) == 1 and entries[0].is_dir() else str(tmp)
    elif repo_path:
        workspace = str(Path(repo_path).expanduser().resolve())
    else:
        raise HTTPException(400, "provide either 'file' (zip upload) or 'repo_path'")

    cfg = RunConfig(
        workspace=workspace,
        max_critique_loops=iterations,
        quality_threshold=threshold,
        deploy=deploy,
        verbose=False,
    )
    orch = Orchestrator(provider=get_provider(provider, model), config=cfg)
    result = orch.run(task or "Analyze this repo and apply the most important fixes.", repo_path=workspace)
    _persist_audit(result.audit.session_id, result.audit.to_dict())
    payload = result.to_dict()
    payload["final_output"] = result.final_output
    return JSONResponse(payload, status_code=200 if result.ok else 422)


@app.get("/api/audit/{session_id}")
def api_audit(session_id: str) -> dict:
    p = AUDIT_DIR / f"{_safe(session_id)}.json"
    if not p.exists():
        raise HTTPException(404, f"no audit for session {session_id}")
    import json
    return json.loads(p.read_text())


# ── helpers ─────────────────────────────────────────────────────────────────

def _persist_audit(session_id: str, data: dict[str, Any]) -> None:
    import json
    p = AUDIT_DIR / f"{_safe(session_id)}.json"
    p.write_text(json.dumps(data, indent=2, default=str))


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:128]
