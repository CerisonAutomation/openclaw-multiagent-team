"""Real tool execution: shell, AST analysis, git, secrets scanning, linting."""
from __future__ import annotations

import ast
import hashlib
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import get_config


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    exit_code: int = 0


# ── Shell execution ───────────────────────────────────────────────────────────

def run_shell(
    cmd: str,
    cwd: str | None = None,
    timeout: int = 30,
    env_extra: dict[str, str] | None = None,
) -> ToolResult:
    cfg = get_config()
    cmd_lower = cmd.lower()
    for blocked in cfg.blocked_commands:
        if blocked in cmd_lower:
            return ToolResult(ok=False, error=f"blocked command pattern: {blocked!r}", exit_code=-1)

    env = {**os.environ, **(env_extra or {})}
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=cwd, env=env,
        )
        return ToolResult(
            ok=proc.returncode == 0,
            output=proc.stdout[-4000:],
            error=proc.stderr[-2000:],
            exit_code=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, error=f"timed out after {timeout}s", exit_code=-1)
    except Exception as e:
        return ToolResult(ok=False, error=str(e), exit_code=-1)


# ── Python AST analysis ───────────────────────────────────────────────────────

@dataclass
class ASTMetrics:
    functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    async_functions: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    global_vars: list[str] = field(default_factory=list)
    max_depth: int = 0
    cyclomatic_complexity: int = 1
    parse_error: str = ""

    def to_dict(self) -> dict:
        return {
            "functions": self.functions,
            "async_functions": self.async_functions,
            "classes": self.classes,
            "imports": self.imports,
            "decorators": self.decorators,
            "global_vars": self.global_vars,
            "max_depth": self.max_depth,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "parse_error": self.parse_error,
        }


class _ASTVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.metrics = ASTMetrics()
        self._depth = 0
        # Branch nodes that add to cyclomatic complexity
        self._BRANCH = (
            ast.If, ast.For, ast.While, ast.ExceptHandler,
            ast.With, ast.Assert, ast.comprehension,
        )

    def _enter(self) -> None:
        self._depth += 1
        self.metrics.max_depth = max(self.metrics.max_depth, self._depth)

    def _exit(self) -> None:
        self._depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.metrics.functions.append(node.name)
        for d in node.decorator_list:
            self.metrics.decorators.append(ast.unparse(d) if hasattr(ast, "unparse") else "?")
        self._enter(); self.generic_visit(node); self._exit()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.metrics.async_functions.append(node.name)
        self._enter(); self.generic_visit(node); self._exit()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.metrics.classes.append(node.name)
        self._enter(); self.generic_visit(node); self._exit()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.metrics.imports.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.metrics.imports.append(node.module)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._depth == 0:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self.metrics.global_vars.append(t.id)
        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:  # type: ignore[override]
        if isinstance(node, self._BRANCH):
            self.metrics.cyclomatic_complexity += 1
        super().generic_visit(node)


def analyze_python_ast(src: str) -> ASTMetrics:
    metrics = ASTMetrics()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        metrics.parse_error = str(e)
        return metrics
    visitor = _ASTVisitor()
    visitor.visit(tree)
    return visitor.metrics


# ── Source metrics (language-agnostic) ───────────────────────────────────────

def compute_metrics(src: str, lang: str) -> dict[str, Any]:
    lines = src.split("\n")
    non_blank = [l for l in lines if l.strip()]
    comment_re = (
        re.compile(r"^\s*#") if lang in ("python", "shell", "yaml")
        else re.compile(r"^\s*(//|\*|/\*)")
    )
    comment_lines = sum(1 for l in lines if comment_re.match(l))
    todos = len(re.findall(r"\b(TODO|FIXME|XXX|HACK)\b", src))
    branches = len(re.findall(r"\b(if|else|elif|for|while|switch|case|catch|try)\b", src))
    fn_count = len(re.findall(r"\b(function|def|fn)\b|=>", src))
    longest_line = max((len(l) for l in lines), default=0)
    avg_line = round(sum(len(l) for l in non_blank) / len(non_blank), 1) if non_blank else 0
    comment_ratio = round(comment_lines / len(non_blank), 3) if non_blank else 0
    cyclomatic_avg = round(branches / fn_count, 2) if fn_count else branches
    byte_count = len(src.encode())

    result: dict[str, Any] = {
        "bytes": byte_count,
        "lines_total": len(lines),
        "lines_code": len(non_blank) - comment_lines,
        "lines_comment": comment_lines,
        "comment_ratio": comment_ratio,
        "function_count": fn_count,
        "branch_keywords": branches,
        "cyclomatic_avg": cyclomatic_avg,
        "longest_line": longest_line,
        "avg_line_length": avg_line,
        "todo_markers": todos,
        "sha256": hashlib.sha256(src.encode()).hexdigest(),
    }

    if lang == "python":
        ast_m = analyze_python_ast(src)
        result["ast"] = ast_m.to_dict()

    return result


# ── Secrets scanner ───────────────────────────────────────────────────────────

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS Access Key",    re.compile(r"(?i)AKIA[0-9A-Z]{16}")),
    ("AWS Secret Key",    re.compile(r"(?i)aws.{0,20}secret.{0,20}['\"][0-9a-zA-Z/+]{40}['\"]")),
    ("Generic API Key",   re.compile(r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"][A-Za-z0-9_\-\.]{16,}['\"]")),
    ("Bearer Token",      re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{20,}")),
    ("JWT",               re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("Private Key",       re.compile(r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----")),
    ("Password in Code",  re.compile(r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"][^'\"]{4,}['\"]")),
    ("GitHub Token",      re.compile(r"ghp_[A-Za-z0-9]{36}")),
    ("OpenAI Key",        re.compile(r"sk-[A-Za-z0-9]{48}")),
    ("Anthropic Key",     re.compile(r"sk-ant-[A-Za-z0-9\-]{90,}")),
    ("Stripe Key",        re.compile(r"(?i)(sk|pk)_(live|test)_[A-Za-z0-9]{24,}")),
    ("Database URL",      re.compile(r"(?i)(mysql|postgres|mongodb|redis)://[^\s\"']{8,}")),
    ("Slack Token",       re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}")),
]


@dataclass
class SecretHit:
    pattern_name: str
    line: int
    snippet: str        # redacted context
    severity: str       # "critical" | "high" | "medium"


def scan_secrets(src: str) -> list[SecretHit]:
    hits: list[SecretHit] = []
    lines = src.split("\n")
    for i, line in enumerate(lines, 1):
        for name, pat in _SECRET_PATTERNS:
            m = pat.search(line)
            if m:
                start, end = m.start(), m.end()
                snippet = line[max(0, start - 20): end].strip()
                # Redact the match itself, show context
                redacted = pat.sub("[REDACTED]", snippet)
                sev = "critical" if any(k in name for k in ("Private Key", "AWS")) else "high"
                hits.append(SecretHit(name, i, redacted, sev))
    return hits


# ── Git analysis ──────────────────────────────────────────────────────────────

def git_summary(repo_path: str) -> ToolResult:
    p = Path(repo_path).resolve()
    if not (p / ".git").exists():
        return ToolResult(ok=False, error=f"not a git repository: {p}")

    def git(cmd: str) -> str:
        r = run_shell(f"git -C {p} {cmd}", timeout=15)
        return r.output.strip() if r.ok else ""

    log_raw = git("log --oneline -20")
    authors = git("shortlog -sn --no-merges -20")
    stat = git("diff --stat HEAD~1 HEAD 2>/dev/null || echo '(no prior commit)'")
    stash = git("stash list")
    branch = git("branch --show-current")
    tags = git("tag --sort=-version:refname -l | head -5")
    uncommitted = git("status --short")

    commits = [l.split(" ", 1) for l in log_raw.splitlines() if l]

    return ToolResult(
        ok=True,
        data={
            "branch": branch,
            "recent_commits": [{"hash": c[0], "message": c[1]} for c in commits if len(c) == 2],
            "authors": [l.strip() for l in authors.splitlines()],
            "last_diff_stat": stat,
            "stash_count": len(stash.splitlines()) if stash else 0,
            "tags": [t.strip() for t in tags.splitlines()],
            "uncommitted_files": [l.strip() for l in uncommitted.splitlines()],
        },
    )


# ── Lint (ruff) ───────────────────────────────────────────────────────────────

def lint_python(src: str) -> ToolResult:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(src)
        tmp = f.name
    try:
        r = run_shell(f"ruff check --output-format=json {tmp}", timeout=20)
        # ruff exits 1 when it finds issues — that's expected, not a hard failure
        raw = r.output.strip()
        try:
            import json
            issues = json.loads(raw) if raw else []
        except Exception:
            issues = []
        return ToolResult(
            ok=True,
            data={
                "issue_count": len(issues),
                "issues": [
                    {
                        "code": i.get("code", "?"),
                        "message": i.get("message", ""),
                        "line": i.get("location", {}).get("row", "?"),
                        "col": i.get("location", {}).get("column", "?"),
                    }
                    for i in issues[:50]
                ],
            },
        )
    finally:
        os.unlink(tmp)


# ── Repository walker ─────────────────────────────────────────────────────────

def analyze_repo(path: str) -> ToolResult:
    root = Path(path).resolve()
    if not root.exists():
        return ToolResult(ok=False, error=f"path not found: {root}")

    ext_counts: dict[str, int] = {}
    total_files = 0
    total_bytes = 0
    top_entries = []

    for item in sorted(root.iterdir()):
        if item.name.startswith("."):
            continue
        top_entries.append(item.name)

    SKIP_DIRS = {".git", "node_modules", ".venv", "__pycache__", ".pytest_cache", "dist", "build"}
    for item in root.rglob("*"):
        if any(p in item.parts for p in SKIP_DIRS):
            continue
        if item.is_file():
            total_files += 1
            try:
                total_bytes += item.stat().st_size
            except OSError:
                pass
            ext = item.suffix.lower() or "no-ext"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

    lang_map = {".py": "python", ".ts": "typescript", ".tsx": "typescript", ".js": "javascript",
                ".jsx": "javascript", ".json": "json", ".yaml": "yaml", ".yml": "yaml",
                ".html": "html", ".css": "css", ".sh": "shell", ".rs": "rust", ".go": "go"}
    languages = sorted({lang_map[e] for e in ext_counts if e in lang_map})

    return ToolResult(
        ok=True,
        data={
            "root": str(root),
            "total_files": total_files,
            "total_bytes": total_bytes,
            "languages": languages,
            "ext_breakdown": dict(sorted(ext_counts.items(), key=lambda x: -x[1])[:20]),
            "top_level_entries": top_entries[:30],
        },
    )
