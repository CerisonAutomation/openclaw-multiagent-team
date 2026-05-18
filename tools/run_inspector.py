"""
openclaw inspector — single-file edition.

All 5 modules consolidated into one runnable file.
See AGENTS.md for full trait/skill/model reference.

Usage:
    python tools/inspector.py          # starts FastAPI on :8765
    # then open tools/inspector.html
"""
from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import ast as _ast
import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine

# ── third-party (all in pyproject.toml) ──────────────────────────────────────
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("inspector")

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIG — model registry, skill definitions, runtime settings
# ══════════════════════════════════════════════════════════════════════════════

OLLAMA_BASE    = os.environ.get("OLLAMA_BASE",    "http://localhost:11434")
JAN_BASE       = os.environ.get("JAN_BASE",       "http://localhost:1337/v1")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))
SERVER_HOST    = os.environ.get("INSPECTOR_HOST",  "0.0.0.0")
SERVER_PORT    = int(os.environ.get("INSPECTOR_PORT", "8765"))
HEARTBEAT_S    = int(os.environ.get("HEARTBEAT_S", "30"))


@dataclass(frozen=True)
class ModelPreset:
    name: str
    ctx_window: int
    roles: tuple[str, ...]
    description: str


OLLAMA_MODEL_PRESETS: dict[str, ModelPreset] = {
    "llama3.2":          ModelPreset("llama3.2",          128_000, ("intent","architect","critic","summarizer"), "Meta Llama 3.2 — fast general-purpose"),
    "llama3.1":          ModelPreset("llama3.1",          128_000, ("intent","architect","critic"),              "Meta Llama 3.1 — strong reasoning"),
    "codellama":         ModelPreset("codellama",          16_384, ("coder","reviewer","fixer","tester"),        "Meta CodeLlama — code specialist"),
    "deepseek-coder-v2": ModelPreset("deepseek-coder-v2", 163_840, ("coder","reviewer","tester"),               "DeepSeek Coder V2 — top code benchmark"),
    "qwen2.5-coder":     ModelPreset("qwen2.5-coder",    131_072, ("coder","reviewer","fixer"),                 "Qwen 2.5 Coder — strong instruction+code"),
    "mistral":           ModelPreset("mistral",            32_768, ("intent","fixer","summarizer"),              "Mistral 7B — fast, low RAM"),
    "phi4":              ModelPreset("phi4",               16_384, ("intent","architect","critic"),              "Microsoft Phi-4 — punches above its weight"),
    "phi3":              ModelPreset("phi3",                4_096, ("intent","critic"),                          "Microsoft Phi-3 — smallest capable"),
    "gemma2":            ModelPreset("gemma2",              8_192, ("intent","summarizer"),                      "Google Gemma 2 — good multilingual"),
    "yi-coder":          ModelPreset("yi-coder",           65_536, ("coder","reviewer","tester"),               "01.AI Yi-Coder — strong code"),
    "starcoder2":        ModelPreset("starcoder2",         16_384, ("coder","reviewer"),                         "BigCode StarCoder2 — code-only training"),
}

ROLE_MODEL_PREFERENCE: dict[str, list[str]] = {
    "intent":     ["llama3.2","llama3.1","phi4","mistral","phi3"],
    "architect":  ["llama3.1","llama3.2","phi4","mistral"],
    "coder":      ["qwen2.5-coder","deepseek-coder-v2","codellama","yi-coder","starcoder2"],
    "reviewer":   ["deepseek-coder-v2","qwen2.5-coder","codellama","llama3.1"],
    "fixer":      ["qwen2.5-coder","codellama","mistral","llama3.2"],
    "tester":     ["deepseek-coder-v2","qwen2.5-coder","codellama","yi-coder"],
    "critic":     ["llama3.1","llama3.2","phi4"],
    "summarizer": ["llama3.2","mistral","gemma2"],
}

BLOCKED_COMMANDS = [
    "rm -rf /", "mkfs", ":(){ :|:& };:", "dd if=/dev/zero", "chmod -R 777 /", "> /dev/sda",
]


@dataclass(frozen=True)
class SkillDef:
    name: str
    description: str
    role: str
    output_format: str      # "json" | "text" | "markdown"
    system_prompt: str
    temperature: float = 0.2
    max_tokens: int = 2048


SKILL_INDEX: dict[str, SkillDef] = {
    "intent": SkillDef("intent", "Classify intent, domain, language, complexity", "intent", "json",
        'Classify the code or text. Respond ONLY with valid JSON:\n'
        '{"primary_intent":"<build|fix|test|deploy|review|explain|data|unknown>",'
        '"domain":"<frontend|backend|fullstack|cli|api|data|ml|devops|config|other>",'
        '"language":"<detected language>","complexity":"<trivial|low|moderate|high|very_high>",'
        '"stated_goal":"<one sentence>","key_patterns":["<pattern>"],'
        '"issues_spotted":["<issue>"],"recommended_next":"<highest-value next action>"}',
        temperature=0.1),

    "arch_review": SkillDef("arch_review", "Architecture review: coupling, abstractions, structural issues", "architect", "json",
        'You are a senior software architect. Analyze the code. Respond ONLY with valid JSON:\n'
        '{"overall_rating":<1.0-10.0>,"architecture_style":"<MVC|layered|event-driven|microservice|monolith|functional|other>",'
        '"strengths":["<strength>"],"weaknesses":["<weakness>"],"coupling_issues":["<issue>"],'
        '"missing_abstractions":["<abstraction>"],"top_priority_fix":"<most important>","estimated_refactor_effort":"<e.g. 2h>"}'),

    "security": SkillDef("security", "Identify vulnerabilities (OWASP, secrets, injection)", "reviewer", "json",
        'You are a security engineer. Find vulnerabilities. Respond ONLY with valid JSON:\n'
        '{"risk_level":"<critical|high|medium|low|none>",'
        '"vulnerabilities":[{"type":"<type>","location":"<where>","severity":"<critical|high|medium|low>","description":"<what>","fix":"<how>"}],'
        '"hardening_suggestions":["<suggestion>"],"owasp_categories":["<category>"]}'),

    "test_gen": SkillDef("test_gen", "Generate a complete test file", "tester", "markdown",
        "Generate comprehensive tests. Output ONE markdown code block with the complete test file.\n"
        "Cover: happy path, edge cases, exceptions, boundary conditions.\n"
        "Same language/framework as the input. No preamble.",
        temperature=0.3, max_tokens=4096),

    "summarize": SkillDef("summarize", "Concise technical summary: purpose, API, notable patterns", "summarizer", "markdown",
        "Concise technical summary. Include: what it does (1-2 sentences), key dependencies, "
        "entry points / public API, notable patterns, anything unusual. Start directly — no intro."),

    "fix": SkillDef("fix", "Diagnose errors and propose the smallest correct patch", "fixer", "json",
        'Diagnose the error and propose the minimal fix. Respond ONLY with valid JSON:\n'
        '{"diagnosis":"<root cause in one sentence>","fix_kind":"<rewrite|patch|config|dependency|environment>",'
        '"patch":"<corrected code or unified diff>","explanation":"<why this fixes it>","confidence":"<low|medium|high>"}'),

    "doc": SkillDef("doc", "Add or improve docstrings and inline documentation", "coder", "markdown",
        "Add/improve docstrings. Return the complete file with documentation added.\n"
        "Use language's native format (Python: Google-style, JS/TS: JSDoc). No preamble.",
        temperature=0.3, max_tokens=4096),

    "critic": SkillDef("critic", "Score 8 quality dimensions with actionable critique", "critic", "json",
        'Score the code on 8 dimensions (0.0-10.0). Be accurate, not generous. '
        'Anchors: <6=needs work, 6-8=acceptable, >8=excellent. Rarely >9.\n'
        'Respond ONLY with valid JSON:\n'
        '{"correctness":<f>,"clarity":<f>,"structure":<f>,"testability":<f>,'
        '"security":<f>,"performance":<f>,"maintainability":<f>,"completeness":<f>,'
        '"lowest_dimension":"<name>","critical_fix":"<most important improvement>","summary":"<2-3 sentence verdict>"}',
        temperature=0.1),

    "loop_runner": SkillDef(
        "loop_runner",
        "Plan a bounded autonomous loop: decompose task, write acceptance criteria and completion token",
        "architect", "json",
        'You are a task planner for a bounded autonomous agent loop (Ralph Wiggum pattern).\n'
        'Given a task description, produce a precise loop specification. Respond ONLY with valid JSON:\n'
        '{"task_summary":"<one sentence>",'
        '"acceptance_criteria":["<verifiable criterion>"],'
        '"completion_token":"<SCREAMING_SNAKE_CASE token Claude writes when done>",'
        '"recommended_max_iter":<int 5-30>,'
        '"sub_tasks":["<concrete step>"],'
        '"verification_commands":["<shell command to verify success>"],'
        '"risk_level":"<low|medium|high>",'
        '"notes":"<anything the agent should watch out for>"}',
        temperature=0.2, max_tokens=1024),
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. TOOLS — shell, AST, secrets, lint, git, repo analysis
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    exit_code: int = 0


def run_shell(cmd: str, cwd: str | None = None, timeout: int = 30) -> ToolResult:
    low = cmd.lower()
    for blocked in BLOCKED_COMMANDS:
        if blocked in low:
            return ToolResult(ok=False, error=f"blocked: {blocked!r}", exit_code=-1)
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, cwd=cwd)
        return ToolResult(ok=p.returncode == 0, output=p.stdout[-4000:],
                          error=p.stderr[-1000:], exit_code=p.returncode)
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, error=f"timed out ({timeout}s)", exit_code=-1)
    except Exception as e:
        return ToolResult(ok=False, error=str(e), exit_code=-1)


@dataclass
class ASTMetrics:
    functions: list[str] = field(default_factory=list)
    async_functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    global_vars: list[str] = field(default_factory=list)
    max_depth: int = 0
    cyclomatic_complexity: int = 1
    parse_error: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class _ASTVisitor(_ast.NodeVisitor):
    _BRANCH = (_ast.If, _ast.For, _ast.While, _ast.ExceptHandler,
               _ast.With, _ast.Assert, _ast.comprehension)

    def __init__(self) -> None:
        self.m = ASTMetrics()
        self._depth = 0

    def _enter(self) -> None:
        self._depth += 1
        self.m.max_depth = max(self.m.max_depth, self._depth)

    def _exit(self) -> None:
        self._depth -= 1

    def visit_FunctionDef(self, node: _ast.FunctionDef) -> None:
        self.m.functions.append(node.name)
        for d in node.decorator_list:
            self.m.decorators.append(_ast.unparse(d) if hasattr(_ast, "unparse") else "?")
        self._enter(); self.generic_visit(node); self._exit()

    def visit_AsyncFunctionDef(self, node: _ast.AsyncFunctionDef) -> None:
        self.m.async_functions.append(node.name)
        self._enter(); self.generic_visit(node); self._exit()

    def visit_ClassDef(self, node: _ast.ClassDef) -> None:
        self.m.classes.append(node.name)
        self._enter(); self.generic_visit(node); self._exit()

    def visit_Import(self, node: _ast.Import) -> None:
        for a in node.names: self.m.imports.append(a.name)

    def visit_ImportFrom(self, node: _ast.ImportFrom) -> None:
        if node.module: self.m.imports.append(node.module)

    def visit_Assign(self, node: _ast.Assign) -> None:
        if self._depth == 0:
            for t in node.targets:
                if isinstance(t, _ast.Name): self.m.global_vars.append(t.id)
        self.generic_visit(node)

    def generic_visit(self, node: _ast.AST) -> None:  # type: ignore[override]
        if isinstance(node, self._BRANCH): self.m.cyclomatic_complexity += 1
        super().generic_visit(node)


def analyze_python_ast(src: str) -> ASTMetrics:
    m = ASTMetrics()
    try:
        tree = _ast.parse(src)
    except SyntaxError as e:
        m.parse_error = str(e); return m
    v = _ASTVisitor(); v.visit(tree)
    return v.m


_LANG_SIGNALS: dict[str, list[tuple[str, int]]] = {
    "json":       [(r"^\s*[\{\[]", 3), (r"^\s*[\{\[][\s\S]*[\}\]]\s*$", 4)],
    "typescript": [(r"\binterface\s+\w+\s*\{", 4), (r"\btype\s+\w+\s*=", 3),
                   (r":\s*(string|number|boolean|any|void)\b", 2)],
    "javascript": [(r"\b(const|let|var)\s+\w+\s*=", 2), (r"\bfunction\s*\w*\s*\(", 2), (r"=>\s*\{", 2)],
    "python":     [(r"^\s*def\s+\w+\s*\(", 4), (r"^\s*class\s+\w+\s*[:\(]", 3),
                   (r"^\s*import\s+\w+", 2), (r"\bself\b", 2)],
    "html":       [(r"<!doctype\s+html", 5), (r"<html[\s>]", 4)],
    "css":        [(r"[.#]?[\w-]+\s*\{[\s\S]*?\}", 3), (r"@(media|keyframes)\b", 3)],
    "yaml":       [(r"^---", 4), (r"^\s*[\w-]+:\s", 2)],
    "shell":      [(r"^#!\/(usr\/)?bin\/(env\s+)?(bash|sh|zsh)", 5), (r"\becho\s+", 2)],
    "sql":        [(r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE)\b", 3), (r"\bFROM\s+\w+", 2)],
}


def detect_language(src: str) -> str:
    if not src.strip(): return "empty"
    scores: dict[str, int] = {}
    for lang, pats in _LANG_SIGNALS.items():
        scores[lang] = sum(w for pat, w in pats if re.search(pat, src, re.MULTILINE))
    if scores.get("json", 0) > 0:
        try: json.loads(src)
        except Exception: scores["json"] = max(0, scores["json"] - 3)
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "unknown"


def compute_metrics(src: str, lang: str) -> dict[str, Any]:
    lines = src.split("\n")
    non_blank = [ln for ln in lines if ln.strip()]
    comment_re = re.compile(r"^\s*#") if lang in ("python","shell","yaml") else re.compile(r"^\s*(//|\*|/\*)")
    comment_lines = sum(1 for ln in lines if comment_re.match(ln))
    todos = len(re.findall(r"\b(TODO|FIXME|XXX|HACK)\b", src))
    branches = len(re.findall(r"\b(if|else|elif|for|while|switch|case|catch|try)\b", src))
    fn_count = len(re.findall(r"\b(function|def|fn)\b|=>", src))
    longest = max((len(ln) for ln in lines), default=0)
    avg = round(sum(len(ln) for ln in non_blank) / len(non_blank), 1) if non_blank else 0
    comment_ratio = round(comment_lines / len(non_blank), 3) if non_blank else 0
    cyclomatic_avg = round(branches / fn_count, 2) if fn_count else branches
    out: dict[str, Any] = {
        "bytes": len(src.encode()), "lines_total": len(lines),
        "lines_code": len(non_blank) - comment_lines, "lines_comment": comment_lines,
        "comment_ratio": comment_ratio, "function_count": fn_count,
        "branch_keywords": branches, "cyclomatic_avg": cyclomatic_avg,
        "longest_line": longest, "avg_line_length": avg, "todo_markers": todos,
        "sha256": hashlib.sha256(src.encode()).hexdigest(),
    }
    if lang == "python":
        out["ast"] = analyze_python_ast(src).to_dict()
    return out


_SECRET_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("AWS Access Key",    re.compile(r"AKIA[0-9A-Z]{16}"),                               "critical"),
    ("AWS Secret Key",    re.compile(r"(?i)aws.{0,20}secret.{0,20}['\"][0-9a-zA-Z/+]{40}['\"]"), "critical"),
    ("Generic API Key",   re.compile(r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"][A-Za-z0-9_\-\.]{16,}['\"]"), "high"),
    ("Bearer Token",      re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{20,}"),            "high"),
    ("JWT",               re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"), "high"),
    ("Private Key",       re.compile(r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----"),   "critical"),
    ("Password in Code",  re.compile(r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"][^'\"]{4,}['\"]"), "high"),
    ("GitHub Token",      re.compile(r"ghp_[A-Za-z0-9]{36}"),                           "critical"),
    ("OpenAI Key",        re.compile(r"sk-[A-Za-z0-9]{48}"),                            "critical"),
    ("Anthropic Key",     re.compile(r"sk-ant-[A-Za-z0-9\-]{90,}"),                    "critical"),
    ("Stripe Key",        re.compile(r"(?i)(sk|pk)_(live|test)_[A-Za-z0-9]{24,}"),     "critical"),
    ("Database URL",      re.compile(r"(?i)(mysql|postgres|mongodb|redis)://[^\s\"']{8,}"), "high"),
    ("Slack Token",       re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),               "high"),
]


def scan_secrets(src: str) -> list[dict[str, Any]]:
    hits = []
    for i, line in enumerate(src.split("\n"), 1):
        for name, pat, sev in _SECRET_PATTERNS:
            m = pat.search(line)
            if m:
                snippet = pat.sub("[REDACTED]", line[max(0, m.start()-20):m.end()].strip())
                hits.append({"pattern": name, "line": i, "snippet": snippet, "severity": sev})
    return hits


def lint_python(src: str) -> ToolResult:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(src); tmp = f.name
    try:
        r = run_shell(f"ruff check --output-format=json {tmp}", timeout=20)
        try:
            issues = json.loads(r.output.strip()) if r.output.strip() else []
        except Exception:
            issues = []
        return ToolResult(ok=True, data={
            "issue_count": len(issues),
            "issues": [{"code": i.get("code","?"), "message": i.get("message",""),
                        "line": i.get("location",{}).get("row","?"),
                        "col": i.get("location",{}).get("column","?")} for i in issues[:50]],
        })
    finally:
        os.unlink(tmp)


def git_summary(path: str) -> ToolResult:
    p = Path(path).resolve()
    if not (p / ".git").exists():
        return ToolResult(ok=False, error=f"not a git repo: {p}")
    def g(cmd: str) -> str:
        r = run_shell(f"git -C {p} {cmd}", timeout=15)
        return r.output.strip() if r.ok else ""
    log = g("log --oneline -20")
    commits = [ln.split(" ", 1) for ln in log.splitlines() if ln]
    return ToolResult(ok=True, data={
        "branch": g("branch --show-current"),
        "recent_commits": [{"hash": c[0], "message": c[1]} for c in commits if len(c) == 2],
        "authors": [ln.strip() for ln in g("shortlog -sn --no-merges -20").splitlines()],
        "last_diff_stat": g("diff --stat HEAD~1 HEAD 2>/dev/null || echo '(no prior commit)'"),
        "stash_count": len(g("stash list").splitlines()),
        "tags": [t.strip() for t in g("tag --sort=-version:refname -l | head -5").splitlines()],
        "uncommitted": [ln.strip() for ln in g("status --short").splitlines()],
    })


def analyze_repo(path: str) -> ToolResult:
    root = Path(path).resolve()
    if not root.exists():
        return ToolResult(ok=False, error=f"not found: {root}")
    SKIP = {".git","node_modules",".venv","__pycache__",".pytest_cache","dist","build"}
    ext_counts: dict[str, int] = {}
    total_files = total_bytes = 0
    for item in root.rglob("*"):
        if any(p in item.parts for p in SKIP): continue
        if item.is_file():
            total_files += 1
            try: total_bytes += item.stat().st_size
            except OSError: pass
            ext = item.suffix.lower() or "no-ext"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
    LANG = {".py":"python",".ts":"typescript",".tsx":"typescript",".js":"javascript",
            ".jsx":"javascript",".json":"json",".yaml":"yaml",".yml":"yaml",
            ".html":"html",".css":"css",".sh":"shell",".rs":"rust",".go":"go"}
    languages = sorted({LANG[e] for e in ext_counts if e in LANG})
    return ToolResult(ok=True, data={
        "root": str(root), "total_files": total_files, "total_bytes": total_bytes,
        "languages": languages,
        "ext_breakdown": dict(sorted(ext_counts.items(), key=lambda x: -x[1])[:20]),
        "top_level_entries": [i.name for i in sorted(root.iterdir()) if not i.name.startswith(".")][:30],
    })


# ══════════════════════════════════════════════════════════════════════════════
# 3. AGENTS — Ollama client, Jan fallback, SkillIndex
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OllamaStatus:
    available: bool
    models: list[str] = field(default_factory=list)
    error: str = ""


def _extract_json(text: str) -> dict | list:
    try: return json.loads(text)
    except Exception: pass
    cleaned = re.sub(r"```json?\s*", "", text).strip().rstrip("`").strip()
    try: return json.loads(cleaned)
    except Exception: pass
    for start_ch, end_ch in (("{","}"), ("[","]")):
        start = text.find(start_ch)
        if start == -1: continue
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == start_ch: depth += 1
            elif ch == end_ch:
                depth -= 1
                if depth == 0:
                    try: return json.loads(text[start:i+1])
                    except Exception: break
    return {"_parse_error": True, "raw": text[:500]}


class OllamaClient:
    async def status(self) -> OllamaStatus:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{OLLAMA_BASE}/api/tags")
                if r.status_code == 200:
                    models = [m["name"].split(":")[0] for m in r.json().get("models", [])]
                    return OllamaStatus(available=True, models=models)
                return OllamaStatus(available=False, error=f"HTTP {r.status_code}")
        except Exception as e:
            return OllamaStatus(available=False, error=str(e))

    async def list_models(self) -> list[dict]:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{OLLAMA_BASE}/api/tags")
                if r.status_code == 200:
                    return [{"name": m["name"], "short": m["name"].split(":")[0],
                             "size_gb": round(m.get("size",0)/1e9,2)} for m in r.json().get("models",[])]
        except Exception: pass
        return []

    async def pick_model(self, role: str) -> str:
        available = {m["short"] for m in await self.list_models()}
        for candidate in ROLE_MODEL_PREFERENCE.get(role, []):
            if candidate in available: return candidate
        if available: return sorted(available, key=len, reverse=True)[0]
        return "llama3.2"

    async def generate(self, prompt: str, model: str, system: str = "",
                       temperature: float = 0.2, max_tokens: int = 2048) -> str:
        payload: dict[str, Any] = {
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system: payload["system"] = system
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as c:
            r = await c.post(f"{OLLAMA_BASE}/api/generate", json=payload)
            r.raise_for_status()
            return r.json().get("response", "")

    async def chat(self, messages: list[dict], model: str,
                   temperature: float = 0.3, max_tokens: int = 2048) -> str:
        payload: dict[str, Any] = {
            "model": model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as c:
            r = await c.post(f"{OLLAMA_BASE}/api/chat", json=payload)
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "")


class JanClient:
    async def status(self) -> OllamaStatus:
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(f"{JAN_BASE}/models")
                if r.status_code == 200:
                    models = [m["id"] for m in r.json().get("data", [])]
                    return OllamaStatus(available=True, models=models)
        except Exception as e:
            return OllamaStatus(available=False, error=str(e))
        return OllamaStatus(available=False, error="no response")

    async def first_model(self) -> str | None:
        st = await self.status()
        return st.models[0] if st.available and st.models else None

    async def generate(self, prompt: str, model: str, system: str = "",
                       temperature: float = 0.2, max_tokens: int = 2048) -> str:
        msgs: list[dict] = []
        if system: msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        payload = {"model": model, "messages": msgs, "temperature": temperature,
                   "max_tokens": max_tokens, "stream": False}
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as c:
            r = await c.post(f"{JAN_BASE}/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    async def chat(self, messages: list[dict], model: str,
                   temperature: float = 0.3, max_tokens: int = 2048) -> str:
        payload = {"model": model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens, "stream": False}
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as c:
            r = await c.post(f"{JAN_BASE}/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]


@dataclass
class SkillResult:
    skill: str
    model_used: str
    provider: str
    output: Any
    raw: str
    ok: bool = True
    error: str = ""


class SkillIndex:
    def __init__(self) -> None:
        self._ollama = OllamaClient()
        self._jan = JanClient()
        self._extra: dict[str, SkillDef] = {}

    def register(self, skill: SkillDef) -> None:
        self._extra[skill.name] = skill

    def list(self) -> list[dict]:
        return [{"name": s.name, "description": s.description,
                 "role": s.role, "output_format": s.output_format}
                for s in {**SKILL_INDEX, **self._extra}.values()]

    def get(self, name: str) -> SkillDef | None:
        return SKILL_INDEX.get(name) or self._extra.get(name)

    async def _pick_provider(self, role: str, preferred: str = "auto") -> tuple[str, str, Any]:
        """Return (provider_name, model, client)."""
        if preferred == "jan":
            model = await self._jan.first_model()
            if model: return ("jan", model, self._jan)
            raise RuntimeError("Jan not available")
        ollama_model = await self._ollama.pick_model(role)
        st = await self._ollama.status()
        if st.available:
            return ("ollama", ollama_model, self._ollama)
        jan_model = await self._jan.first_model()
        if jan_model:
            return ("jan", jan_model, self._jan)
        raise RuntimeError("Neither Ollama nor Jan is available")

    async def run(self, skill_name: str, source: str,
                  model: str | None = None, provider: str = "auto",
                  extra_context: str = "") -> SkillResult:
        skill = self.get(skill_name)
        if skill is None:
            return SkillResult(skill=skill_name, model_used="", provider="", output=None,
                               raw="", ok=False, error=f"skill not found: {skill_name!r}")
        try:
            prov_name, prov_model, client = await self._pick_provider(skill.role, provider)
        except RuntimeError as e:
            return SkillResult(skill=skill_name, model_used="", provider="none", output=None,
                               raw="", ok=False, error=str(e))

        chosen = model or prov_model
        prompt = f"{extra_context}\n\n{source}".strip() if extra_context else source
        try:
            raw = await client.generate(prompt, chosen, skill.system_prompt,
                                        skill.temperature, skill.max_tokens)
        except Exception as e:
            return SkillResult(skill=skill_name, model_used=chosen, provider=prov_name,
                               output=None, raw="", ok=False, error=str(e))

        if skill.output_format == "json":
            parsed = _extract_json(raw)
            return SkillResult(skill=skill_name, model_used=chosen, provider=prov_name,
                               output=parsed, raw=raw, ok="_parse_error" not in parsed)
        return SkillResult(skill=skill_name, model_used=chosen, provider=prov_name,
                           output=raw.strip(), raw=raw, ok=True)


skill_index = SkillIndex()


# ══════════════════════════════════════════════════════════════════════════════
# 4. SCHEDULER — event bus, heartbeat, cron, file watcher
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Event:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


Listener = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    def __init__(self) -> None:
        self._listeners: list[Listener] = []

    def subscribe(self, fn: Listener) -> None:
        self._listeners.append(fn)

    def unsubscribe(self, fn: Listener) -> None:
        self._listeners = [cb for cb in self._listeners if cb is not fn]

    async def emit(self, event: Event) -> None:
        for fn in list(self._listeners):
            try: await fn(event)
            except Exception as e: logger.warning("bus listener error: %s", e)


bus = EventBus()


class HeartbeatMonitor:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._last: dict[str, Any] = {}
        self._last_up: bool | None = None

    @property
    def last(self) -> dict[str, Any]:
        return self._last

    def start(self) -> None:
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="heartbeat")

    def stop(self) -> None:
        if self._task: self._task.cancel()

    async def _loop(self) -> None:
        while True:
            status = await self._collect()
            self._last = status
            await bus.emit(Event("heartbeat", status))
            await asyncio.sleep(HEARTBEAT_S)

    async def _collect(self) -> dict[str, Any]:
        st = await skill_index._ollama.status()
        jan_st = await skill_index._jan.status()
        if self._last_up is True and not st.available:
            await bus.emit(Event("ollama_down", {"error": st.error}))
        elif self._last_up is False and st.available:
            await bus.emit(Event("ollama_up", {"models": st.models}))
        self._last_up = st.available
        return {
            "ts": time.time(),
            "ollama": {"available": st.available, "models": st.models, "error": st.error},
            "jan": {"available": jan_st.available, "models": jan_st.models, "error": jan_st.error},
        }


AsyncTask = Callable[[], Coroutine[Any, Any, Any]]


@dataclass
class ScheduledJob:
    name: str
    fn: AsyncTask
    interval_s: int
    last_run: float = 0.0
    run_count: int = 0
    last_error: str = ""
    enabled: bool = True


class CronScheduler:
    def __init__(self) -> None:
        self._jobs: dict[str, ScheduledJob] = {}
        self._task: asyncio.Task | None = None

    def add(self, name: str, fn: AsyncTask, interval_s: int) -> None:
        self._jobs[name] = ScheduledJob(name=name, fn=fn, interval_s=interval_s)

    def remove(self, name: str) -> bool:
        return bool(self._jobs.pop(name, None))

    def list(self) -> list[dict]:
        now = time.time()
        return [{"name": j.name, "interval_s": j.interval_s, "run_count": j.run_count,
                 "next_run_in_s": max(0, j.interval_s - (now - j.last_run)) if j.last_run else 0,
                 "last_error": j.last_error, "enabled": j.enabled}
                for j in self._jobs.values()]

    def start(self) -> None:
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="cron")

    def stop(self) -> None:
        if self._task: self._task.cancel()

    async def _loop(self) -> None:
        while True:
            now = time.time()
            for job in list(self._jobs.values()):
                if job.enabled and now - job.last_run >= job.interval_s:
                    asyncio.create_task(self._run(job))
            await asyncio.sleep(1)

    async def _run(self, job: ScheduledJob) -> None:
        job.last_run = time.time(); job.run_count += 1
        try:
            result = await job.fn()
            await bus.emit(Event("task_done", {"job": job.name, "run": job.run_count,
                                               "result": str(result)[:500]}))
        except Exception as e:
            job.last_error = str(e)
            await bus.emit(Event("task_error", {"job": job.name, "error": str(e)}))


class FileWatcher:
    def __init__(self, path: str, interval_s: int = 5) -> None:
        self._path = path; self._interval = interval_s
        self._mtimes: dict[str, float] = {}
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._poll(), name="file-watcher")

    def stop(self) -> None:
        if self._task: self._task.cancel()

    async def _poll(self) -> None:
        while True:
            try:
                for root, _, files in os.walk(self._path):
                    if any(s in root for s in (".git","node_modules","__pycache__")): continue
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        try: mtime = os.path.getmtime(fpath)
                        except OSError: continue
                        prev = self._mtimes.get(fpath)
                        if prev is not None and mtime != prev:
                            await bus.emit(Event("file_changed", {"path": fpath}))
                        self._mtimes[fpath] = mtime
            except Exception as e: logger.warning("watcher error: %s", e)
            await asyncio.sleep(self._interval)


heartbeat = HeartbeatMonitor()
scheduler = CronScheduler()


# ══════════════════════════════════════════════════════════════════════════════
# 5. SERVER — FastAPI routes, WebSocket, all endpoints
# ══════════════════════════════════════════════════════════════════════════════

class WSManager:
    def __init__(self) -> None: self._conns: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept(); self._conns.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._conns = [c for c in self._conns if c is not ws]

    async def broadcast(self, data: dict) -> None:
        payload = json.dumps(data, default=str)
        for ws in list(self._conns):
            try: await ws.send_text(payload)
            except Exception: self.disconnect(ws)

    @property
    def count(self) -> int: return len(self._conns)


ws_manager = WSManager()


async def _bus_to_ws(event: Event) -> None:
    await ws_manager.broadcast({"event": event.kind, "ts": event.ts, **event.payload})


_UI_PATH = Path(__file__).parent / "inspector.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    bus.subscribe(_bus_to_ws)
    heartbeat.start()
    scheduler.start()
    logger.info("inspector ready → http://%s:%d/", SERVER_HOST, SERVER_PORT)
    yield
    heartbeat.stop(); scheduler.stop()
    bus.unsubscribe(_bus_to_ws)


app = FastAPI(title="openclaw inspector", version="1.0.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def root():
    if _UI_PATH.exists(): return FileResponse(_UI_PATH, media_type="text/html")
    return HTMLResponse("<p>Open tools/inspector.html in a browser.</p>")


@app.get("/healthz")
async def healthz():
    return {"ok": True, "ts": time.time(), "ws_connections": ws_manager.count,
            "heartbeat": heartbeat.last}


# ── providers ─────────────────────────────────────────────────────────────────

@app.get("/api/providers")
async def get_providers():
    ost, jst = await asyncio.gather(skill_index._ollama.status(), skill_index._jan.status())
    active = "ollama" if ost.available else ("jan" if jst.available else "none")
    return {"ollama": {"available": ost.available, "base": OLLAMA_BASE,
                       "models": ost.models, "error": ost.error},
            "jan": {"available": jst.available, "base": JAN_BASE,
                    "models": jst.models, "error": jst.error},
            "active": active}


@app.get("/api/providers/models")
async def list_models():
    ollama_models, jan_st = await asyncio.gather(
        skill_index._ollama.list_models(), skill_index._jan.status())
    return {"ollama": ollama_models, "jan": jan_st.models}


# ── analysis ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    source: str
    language: str = ""


@app.post("/api/analyze/code")
async def analyze_code(req: AnalyzeRequest):
    lang = req.language or detect_language(req.source)
    metrics = compute_metrics(req.source, lang)
    secrets = scan_secrets(req.source)
    lint = None
    if lang == "python":
        lr = lint_python(req.source)
        lint = lr.data if lr.ok else {"error": lr.error}
    return {"language": lang, "metrics": metrics, "secrets": secrets, "lint": lint}


@app.post("/api/analyze/repo")
async def analyze_repo_route(body: dict[str, str]):
    path = body.get("path", ".")
    result = analyze_repo(path)
    if not result.ok: raise HTTPException(400, result.error)
    git = git_summary(path)
    return {"repo": result.data, "git": git.data if git.ok else {"error": git.error}}


@app.post("/api/analyze/shell")
async def run_shell_route(body: dict[str, Any]):
    cmd = body.get("cmd", "").strip()
    if not cmd: raise HTTPException(400, "cmd is required")
    res = run_shell(cmd, cwd=body.get("cwd", "."), timeout=int(body.get("timeout", 30)))
    return {"ok": res.ok, "output": res.output, "error": res.error, "exit_code": res.exit_code}


# ── skills ────────────────────────────────────────────────────────────────────

@app.get("/api/skills")
async def list_skills():
    return {"skills": skill_index.list()}


class RunSkillRequest(BaseModel):
    skill: str
    source: str
    model: str | None = None
    provider: str = "auto"
    extra_context: str = ""


@app.post("/api/skills/run")
async def run_skill(req: RunSkillRequest):
    r = await skill_index.run(req.skill, req.source, model=req.model,
                               provider=req.provider, extra_context=req.extra_context)
    if not r.ok: raise HTTPException(500, r.error)
    return {"skill": r.skill, "model": r.model_used, "provider": r.provider, "output": r.output}


class PipelineRequest(BaseModel):
    source: str
    skills: list[str] = Field(default_factory=list)
    model: str | None = None
    provider: str = "auto"


@app.post("/api/skills/pipeline")
async def run_pipeline(req: PipelineRequest):
    names = req.skills or list(SKILL_INDEX.keys())
    out: dict[str, Any] = {}
    for name in names:
        r = await skill_index.run(name, req.source, model=req.model, provider=req.provider)
        out[name] = {"ok": r.ok, "output": r.output, "model": r.model_used,
                     "provider": r.provider, "error": r.error}
    return {"results": out, "skills_run": len(out)}


# ── chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    model: str | None = None
    provider: str = "auto"
    temperature: float = 0.3
    max_tokens: int = 2048


@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        prov, model, client = await skill_index._pick_provider("intent", req.provider)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    chosen = req.model or model
    content = await client.chat(req.messages, chosen, req.temperature, req.max_tokens)
    return {"content": content, "model": chosen, "provider": prov}


# ── loop planner ─────────────────────────────────────────────────────────────

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
    token = result.output.get("completion_token", "LOOP_DONE") if isinstance(result.output, dict) else "LOOP_DONE"
    max_iter = result.output.get("recommended_max_iter", 20) if isinstance(result.output, dict) else 20
    return {
        "plan": result.output,
        "model": result.model_used,
        "cli_hint": f'python tools/loop.py --task "{req.task}" --promise {token} --max-iter {max_iter}',
    }


# ── relay agent ──────────────────────────────────────────────────────────────

_INTENT_TOOL_MAP: dict[str, list[tuple[str,str]]] = {
    "review":  [("skill","summarize"),("skill","critic"),("code","metrics")],
    "fix":     [("skill","fix"),("skill","security")],
    "test":    [("skill","test_gen")],
    "build":   [("skill","arch_review"),("skill","summarize")],
    "deploy":  [("skill","arch_review"),("skill","security")],
    "explain": [("skill","summarize"),("code","metrics")],
    "data":    [("skill","critic"),("code","metrics")],
    "unknown": [("skill","summarize"),("code","metrics")],
}
_SECURITY_DOMAINS = {"backend","fullstack","api","devops"}


def _relay_select_tools(intent: dict, has_source: bool) -> list[tuple[str,str]]:
    if not has_source:
        return []
    primary = intent.get("primary_intent","unknown")
    domain  = intent.get("domain","other")
    lang    = intent.get("language","unknown")
    base = list(_INTENT_TOOL_MAP.get(primary, _INTENT_TOOL_MAP["unknown"]))
    base.append(("code","secrets"))
    if domain in _SECURITY_DOMAINS and ("skill","security") not in base:
        base.append(("skill","security"))
    if lang == "python":
        base += [("code","lint"),("code","ast")]
    seen: set = set(); result = []
    for item in base:
        if item not in seen:
            seen.add(item); result.append(item)
    return result


def _relay_code_tool(name: str, source: str, intent: dict) -> Any:
    lang = intent.get("language","unknown")
    if name == "metrics":  return compute_metrics(source, lang)
    if name == "secrets":
        return [{"pattern":h.pattern_name,"line":h.line,"snippet":h.snippet,"severity":h.severity}
                for h in scan_secrets(source)]
    if name == "lint":
        r = lint_python(source); return r.data if r.ok else {"error":r.error}
    if name == "ast":
        return analyze_python_ast(source).to_dict()
    return {"error":f"unknown:{name}"}


def _relay_synthesize(user_input: str, intent: dict, outputs: dict) -> str:
    parts = []
    primary=intent.get("primary_intent","?"); domain=intent.get("domain","?")
    lang=intent.get("language","?"); complexity=intent.get("complexity","?")
    goal=intent.get("stated_goal","")
    parts.append(f"**{primary.upper()} · {domain} · {lang} · {complexity}**" + (f"\n_{goal}_" if goal else ""))
    issues=intent.get("issues_spotted",[])
    if issues: parts.append("**Issues:** " + " · ".join(f"`{i}`" for i in issues[:5]))
    if "metrics" in outputs:
        m=outputs["metrics"]
        parts.append(f"**Metrics:** {m.get('lines_code','?')} lines · {m.get('bytes',0):,}B · CC {m.get('cyclomatic_avg','?')} · SHA {str(m.get('sha256',''))[:8]}")
        if m.get("todo_markers",0): parts.append(f"  ⚠ {m['todo_markers']} TODO/FIXME")
    if "secrets" in outputs:
        hits=outputs["secrets"]
        if hits: parts.append(f"**🔴 Secrets ({len(hits)}):** " + "; ".join(f"`{h['pattern']}` L{h['line']}" for h in hits[:4]))
        else: parts.append("**Secrets:** ✓ clean")
    if "lint" in outputs:
        ln=outputs["lint"]
        if "error" in ln: parts.append(f"**Lint:** error — {ln['error']}")
        elif ln.get("issue_count",0)==0: parts.append("**Lint:** ✓ 0 issues")
        else: parts.append(f"**Lint:** {ln['issue_count']} issue(s) — " + " · ".join(f"`{i['code']}` L{i['line']}" for i in ln.get("issues",[])[:3]))
    if "ast" in outputs:
        a=outputs["ast"]
        if not a.get("parse_error"):
            parts.append(f"**AST:** {len(a.get('functions',[]))} fn · {len(a.get('classes',[]))} cls · CC={a.get('cyclomatic_complexity','?')} · depth={a.get('max_depth','?')}")
    _LABELS={"summarize":"Summary","critic":"Critique","fix":"Fix","security":"Security","test_gen":"Tests","arch_review":"Architecture"}
    for skill,label in _LABELS.items():
        if skill not in outputs: continue
        out=outputs[skill]
        if isinstance(out,dict):
            if "error" in out: parts.append(f"**{label}:** ✗ {out['error'][:120]}")
            elif skill=="critic":
                dims={k:v for k,v in out.items() if isinstance(v,(int,float))}
                avg=round(sum(dims.values())/len(dims),1) if dims else "?"
                parts.append(f"**Critique:** avg={avg}/10 · worst={out.get('lowest_dimension','?')}" + (f"\n  → _{out.get('critical_fix','')}_" if out.get("critical_fix") else ""))
            elif skill=="fix": parts.append(f"**Fix ({out.get('confidence','?')}):** {out.get('diagnosis','')}")
            elif skill=="security":
                risk=out.get("risk_level","?")
                emoji={"critical":"🔴","high":"🟠","medium":"🟡","low":"🟢","none":"✓"}.get(risk,"?")
                parts.append(f"**Security:** {emoji} {risk.upper()}" + (f" · {len(out.get('vulnerabilities',[]))} vuln(s)" if out.get("vulnerabilities") else ""))
            elif skill=="arch_review": parts.append(f"**Architecture:** {out.get('overall_rating','?')}/10" + (f"\n  → _{out.get('top_priority_fix','')}_" if out.get("top_priority_fix") else ""))
            else: parts.append(f"**{label}:** {str(out)[:300]}")
        elif isinstance(out,str): parts.append(f"**{label}:**\n{out.strip()[:600]}")
    next_action=intent.get("recommended_next","")
    if next_action: parts.append(f"**Recommended next:** _{next_action}_")
    return "\n\n".join(parts)


_RELAY_MAX_HISTORY = 10

@dataclass
class _RelayMsg:
    role: str
    content: str
    tools_used: list[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)


class _RelayAgent:
    def __init__(self) -> None:
        self._history: list[_RelayMsg] = []

    async def handle(self, user_input: str, source: str = "", model: str | None = None) -> dict:
        t0 = time.time()
        intent_src = f"{user_input}\n\n---\n{source}" if source else user_input
        intent_result = await skill_index.run("intent", intent_src, model=model)
        intent = intent_result.output if isinstance(intent_result.output, dict) else {}
        selected = _relay_select_tools(intent, bool(source.strip()))
        tool_outputs: dict[str, Any] = {}
        skill_tasks: dict[str, Any] = {}
        for tt, name in selected:
            if tt == "code":
                tool_outputs[name] = await asyncio.to_thread(_relay_code_tool, name, source, intent)
            else:
                skill_tasks[name] = asyncio.create_task(skill_index.run(name, source or user_input, model=model))
        if skill_tasks:
            results = await asyncio.gather(*skill_tasks.values(), return_exceptions=True)
            for sname, r in zip(skill_tasks.keys(), results):
                if isinstance(r, Exception): tool_outputs[sname] = {"error": str(r)}
                else: tool_outputs[sname] = r.output if r.ok else {"error": r.error}
        message = _relay_synthesize(user_input, intent, tool_outputs)
        latency_ms = round((time.time() - t0) * 1000, 1)
        tools_used = [f"{tt}:{n}" for tt, n in selected]
        self._history.append(_RelayMsg("user", user_input))
        self._history.append(_RelayMsg("assistant", message, tools_used))
        if len(self._history) > _RELAY_MAX_HISTORY * 2:
            self._history = self._history[-_RELAY_MAX_HISTORY * 2:]
        return {"message": message, "intent": intent, "tools_used": tools_used,
                "tool_outputs": tool_outputs, "model": intent_result.model_used,
                "provider": "ollama", "latency_ms": latency_ms}

    def history(self) -> list[dict]:
        return [{"role": m.role, "content": m.content, "tools": m.tools_used, "ts": m.ts} for m in self._history]

    def clear(self) -> None:
        self._history.clear()


_relay_sessions: dict[str, _RelayAgent] = {}


class RelayRequest(BaseModel):
    message: str
    source: str = ""
    session_id: str = "default"
    model: str | None = None


@app.post("/api/relay")
async def relay(req: RelayRequest):
    """Classify intent, auto-select tools, execute concurrently, synthesize response."""
    if req.session_id not in _relay_sessions:
        _relay_sessions[req.session_id] = _RelayAgent()
    return await _relay_sessions[req.session_id].handle(req.message, req.source, req.model)


@app.get("/api/relay/history/{session_id}")
async def relay_history(session_id: str):
    agent = _relay_sessions.get(session_id)
    return {"session_id": session_id, "history": agent.history() if agent else []}


@app.delete("/api/relay/history/{session_id}")
async def relay_clear(session_id: str):
    agent = _relay_sessions.pop(session_id, None)
    return {"cleared": session_id, "existed": agent is not None}


# ── scheduler ─────────────────────────────────────────────────────────────────

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
    async def _fn() -> str:
        r = await skill_index.run(req.skill, req.source)
        await ws_manager.broadcast({"event": "scheduled_result", "job": req.name,
                                     "skill": req.skill, "output": r.output,
                                     "model": r.model_used, "ok": r.ok})
        return str(r.output)[:200]
    scheduler.add(req.name, _fn, req.interval_s)
    return {"added": req.name, "interval_s": req.interval_s}


@app.delete("/api/schedule/{name}")
async def remove_job(name: str):
    if not scheduler.remove(name): raise HTTPException(404, f"job not found: {name}")
    return {"removed": name}


_watchers: dict[str, FileWatcher] = {}


@app.post("/api/watch")
async def start_watch(body: dict[str, Any]):
    path = body.get("path", ".")
    if path in _watchers: return {"watching": path, "status": "already_watching"}
    w = FileWatcher(path, int(body.get("interval_s", 5)))
    w.start(); _watchers[path] = w
    return {"watching": path}


@app.delete("/api/watch")
async def stop_watch(body: dict[str, str]):
    path = body.get("path", "")
    w = _watchers.pop(path, None)
    if w: w.stop()
    return {"stopped": path}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    logger.info("WS+ total=%d", ws_manager.count)
    try:
        while True:
            data = await ws.receive_text()
            try: msg = json.loads(data)
            except Exception:
                await ws.send_text(json.dumps({"error": "invalid JSON"})); continue
            if msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong", "ts": time.time()}))
            elif msg.get("type") == "run_skill":
                r = await skill_index.run(msg["skill"], msg.get("source",""), model=msg.get("model"))
                await ws.send_text(json.dumps({"type":"skill_result","skill":r.skill,
                                               "output":r.output,"model":r.model_used,
                                               "provider":r.provider,"ok":r.ok}, default=str))
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
        logger.info("WS- total=%d", ws_manager.count)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Use app object directly so the file runs from any CWD without package import tricks.
    # For hot-reload during development: PYTHONPATH=. uvicorn tools.run_inspector:app --reload
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")
