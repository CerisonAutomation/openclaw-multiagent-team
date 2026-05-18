"""Tools agents can actually invoke: shell, fs, vercel, repo analysis, image-gen.

Every tool returns a ToolResult so the orchestrator can log a uniform audit row.
Resource limits adapted from HORUS_JSON_BROWSER (max_execution_time, allowed
domains, restricted_operations).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


# ── Resource limits (from HORUS_JSON_BROWSER pattern) ───────────────────────

@dataclass
class AgentLimits:
    max_execution_time_s: int = 180
    max_tool_calls: int = 200
    allowed_install_managers: tuple[str, ...] = ("npm", "pnpm", "yarn", "pip", "uv")
    restricted_commands: tuple[str, ...] = ("rm -rf /", "mkfs", "dd if=", "shutdown", ":(){:|:&};:")

    def check(self, cmd: str) -> tuple[bool, str]:
        lowered = cmd.lower()
        for bad in self.restricted_commands:
            if bad.lower() in lowered:
                return False, f"command blocked by safety rule: {bad}"
        return True, "ok"


@dataclass
class ToolResult:
    ok: bool
    output: str
    error: str = ""
    exit_code: int = 0
    extra: dict = field(default_factory=dict)


# ── Shell ───────────────────────────────────────────────────────────────────

def run_shell(cmd: str, cwd: str | Path | None = None, timeout: int = 180, limits: AgentLimits | None = None) -> ToolResult:
    limits = limits or AgentLimits()
    ok, reason = limits.check(cmd)
    if not ok:
        return ToolResult(False, "", reason, exit_code=126)
    try:
        proc = subprocess.run(
            cmd, shell=True, cwd=str(cwd) if cwd else None,
            timeout=timeout, capture_output=True, text=True,
        )
        return ToolResult(
            ok=proc.returncode == 0,
            output=proc.stdout,
            error=proc.stderr,
            exit_code=proc.returncode,
        )
    except subprocess.TimeoutExpired as e:
        return ToolResult(False, "", f"timeout after {timeout}s: {e}", exit_code=124)
    except Exception as e:  # noqa: BLE001
        return ToolResult(False, "", f"shell error: {e}", exit_code=1)


# ── File system ─────────────────────────────────────────────────────────────

def write_file(path: str | Path, content: str, workspace: str | Path) -> ToolResult:
    p = Path(workspace) / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return ToolResult(True, f"wrote {p}", extra={"bytes": len(content)})


def read_file(path: str | Path, workspace: str | Path) -> ToolResult:
    p = Path(workspace) / path
    if not p.exists():
        return ToolResult(False, "", f"file not found: {p}")
    return ToolResult(True, p.read_text())


def ensure_workspace(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Repo analysis ───────────────────────────────────────────────────────────

_LANG_SIGNATURES = {
    "package.json": "node",
    "pyproject.toml": "python",
    "requirements.txt": "python",
    "Cargo.toml": "rust",
    "go.mod": "go",
    "pom.xml": "java",
    "Gemfile": "ruby",
}

_FW_SIGNATURES = {
    "next.config.js": "nextjs",
    "next.config.mjs": "nextjs",
    "next.config.ts": "nextjs",
    "vite.config.js": "vite",
    "vite.config.ts": "vite",
    "remix.config.js": "remix",
    "astro.config.mjs": "astro",
    "nuxt.config.ts": "nuxt",
}


def analyze_repo(path: str | Path) -> dict:
    """Return a small structural summary of a repository."""
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return {"error": f"path does not exist: {root}"}

    files = [p.relative_to(root) for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]
    languages: dict[str, int] = {}
    frameworks: list[str] = []
    has_tests = False
    for rel in files:
        name = rel.name
        if name in _LANG_SIGNATURES:
            languages[_LANG_SIGNATURES[name]] = languages.get(_LANG_SIGNATURES[name], 0) + 1
        if name in _FW_SIGNATURES:
            frameworks.append(_FW_SIGNATURES[name])
        if "test" in str(rel).lower() or "spec" in str(rel).lower():
            has_tests = True

    pkg = root / "package.json"
    deps: list[str] = []
    scripts: dict[str, str] = {}
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text())
            deps = sorted(list((data.get("dependencies") or {}).keys()))[:25]
            scripts = data.get("scripts") or {}
        except json.JSONDecodeError:
            pass

    return {
        "root": str(root),
        "file_count": len(files),
        "languages": languages,
        "frameworks": list(set(frameworks)),
        "has_tests": has_tests,
        "package_scripts": scripts,
        "top_deps": deps,
        "top_level_entries": sorted([p.name for p in root.iterdir() if not p.name.startswith(".")])[:30],
    }


# ── Vercel ──────────────────────────────────────────────────────────────────

_VERCEL_URL_RE = re.compile(r"https?://[^\s]+\.vercel\.app[^\s]*")


def vercel_cli_available() -> bool:
    return shutil.which("vercel") is not None


def vercel_deploy(workspace: str | Path, prod: bool = False, token: str | None = None) -> ToolResult:
    """Run `vercel deploy` non-interactively. Requires VERCEL_TOKEN."""
    token = token or os.environ.get("VERCEL_TOKEN")
    if not vercel_cli_available():
        return ToolResult(False, "", "vercel CLI not installed. Run: npm i -g vercel", exit_code=127)
    if not token:
        return ToolResult(False, "", "VERCEL_TOKEN not set; cannot deploy non-interactively", exit_code=2)

    args = ["vercel", "deploy", "--yes", "--token", token]
    if prod:
        args.append("--prod")
    cmd = " ".join(args)
    res = run_shell(cmd, cwd=workspace, timeout=600)
    if res.ok:
        m = _VERCEL_URL_RE.search(res.output + res.error)
        if m:
            res.extra["deploy_url"] = m.group(0)
    return res


# ── Image generation (provider-agnostic stub) ───────────────────────────────

def generate_image(prompt: str, out_path: str | Path, provider: str | None = None) -> ToolResult:
    """Generate an image. Supports OpenAI (DALL-E) when OPENAI_API_KEY is set;
    falls back to a deterministic SVG placeholder so the pipeline can still run
    without network credentials.
    """
    provider = provider or os.environ.get("OPENCLAW_IMAGE_PROVIDER", "auto")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if provider in ("openai", "auto") and os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.images.generate(model="gpt-image-1", prompt=prompt, size="1024x1024", n=1)
            # The OpenAI SDK returns a b64_json or URL; handle URL case which is the default
            data = resp.data[0]
            if getattr(data, "b64_json", None):
                import base64
                out.write_bytes(base64.b64decode(data.b64_json))
            elif getattr(data, "url", None):
                import httpx
                out.write_bytes(httpx.get(data.url, timeout=30).content)
            else:
                return ToolResult(False, "", "OpenAI image response missing data")
            return ToolResult(True, f"image written: {out}", extra={"provider": "openai"})
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, "", f"openai image failed: {e}")

    # Fallback: deterministic SVG placeholder
    svg = _placeholder_svg(prompt)
    out.with_suffix(".svg").write_text(svg)
    return ToolResult(True, f"placeholder SVG written: {out.with_suffix('.svg')}",
                      extra={"provider": "placeholder", "prompt": prompt})


def _placeholder_svg(prompt: str) -> str:
    # Hash-derived gradient so identical prompts get identical placeholders
    import hashlib
    h = hashlib.sha256(prompt.encode()).hexdigest()
    c1, c2 = f"#{h[:6]}", f"#{h[6:12]}"
    safe = (prompt[:60]).replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">'
        f'<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{c1}"/><stop offset="1" stop-color="{c2}"/>'
        f'</linearGradient></defs>'
        f'<rect width="1024" height="1024" fill="url(#g)"/>'
        f'<text x="512" y="512" font-family="sans-serif" font-size="28" fill="white" '
        f'text-anchor="middle" dominant-baseline="middle">{safe}</text></svg>'
    )
