"""openclaw CLI — drop in a prompt or a repo and let agents take over.

Examples:
    openclaw auto                              # detect and run from cwd
    openclaw setup                             # one-time setup wizard
    openclaw build "a Next.js todo app"        # build from a prompt
    openclaw fix ./my-repo --task "..."        # analyze + fix an existing repo
    openclaw deploy ./my-repo --prod           # deploy to Vercel
    openclaw analyze ./my-repo                 # repo summary only, no LLM calls
    openclaw serve --port 8000                 # start FastAPI server
    openclaw providers                         # list provider presets
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from openclaw import __version__
from openclaw.models import PRESETS
from openclaw.orchestrator import Orchestrator, RunConfig
from openclaw.providers import get_provider
from openclaw.tools import analyze_repo


# ── .openclaw.toml loader ────────────────────────────────────────────────────

def _load_toml_config() -> dict:
    """Load [openclaw] section from .openclaw.toml in cwd. Zero external deps."""
    p = Path(".openclaw.toml")
    if not p.exists():
        return {}
    try:
        import tomllib  # Python 3.11+
        with open(p, "rb") as f:
            return tomllib.load(f).get("openclaw", {})
    except ImportError:
        pass
    # Fallback for Python 3.10: parse key = "value" lines in [openclaw] section
    result: dict = {}
    in_section = False
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "[openclaw]":
            in_section = True
            continue
        if line.startswith("["):
            in_section = False
            continue
        if in_section and "=" in line:
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            try:
                result[k] = int(v)
            except ValueError:
                try:
                    result[k] = float(v)
                except ValueError:
                    result[k] = True if v.lower() == "true" else (False if v.lower() == "false" else v)
    return result


def _apply_toml_defaults(args: argparse.Namespace, toml: dict) -> None:
    """Fill args from .openclaw.toml for any field the user didn't override."""
    if not getattr(args, "provider", None) and toml.get("provider"):
        args.provider = str(toml["provider"])
    if not getattr(args, "model", None) and toml.get("model"):
        args.model = str(toml["model"])
    if getattr(args, "iterations", 3) == 3 and toml.get("iterations"):
        args.iterations = int(toml["iterations"])
    if getattr(args, "threshold", 7.5) == 7.5 and toml.get("threshold"):
        args.threshold = float(toml["threshold"])


# ── command handlers ─────────────────────────────────────────────────────────

def _auto_cmd(args: argparse.Namespace) -> int:
    """Auto-detect intent from cwd: git repo → fix mode, no repo → build mode."""
    toml = _load_toml_config()
    _apply_toml_defaults(args, toml)

    is_repo = subprocess.run(
        ["git", "rev-parse", "--git-dir"], capture_output=True
    ).returncode == 0

    task = (
        getattr(args, "task", None)
        or toml.get("task")
        or ("Analyze this repo, identify the most critical issues, and fix them autonomously."
            if is_repo else None)
    )

    if is_repo:
        args.repo = str(Path(".").resolve())
        args.task = task
        return _fix_cmd(args)

    if not task:
        print("error: not in a git repo and no task provided.", file=sys.stderr)
        print("  Provide a task:  openclaw auto --task 'build a todo app'", file=sys.stderr)
        print("  Or in toml:      task = '...'  in .openclaw.toml", file=sys.stderr)
        return 1

    # Build mode — set build-specific defaults from toml or hardcoded fallbacks
    args.task = task
    args.file = None
    args.workspace = str(toml.get("workspace") or "./openclaw_workspace")
    args.deploy = bool(toml.get("deploy", False))
    args.prod = bool(toml.get("prod", False))
    args.image = None
    return _build_cmd(args)


def _setup_cmd(args: argparse.Namespace) -> int:
    """Detect API keys and create .openclaw.toml if missing."""
    print("openclaw setup\n")

    detected = [
        name for name, preset in PRESETS.items()
        if preset.key_env and os.environ.get(preset.key_env)
    ]

    if detected:
        print("Detected providers:")
        for name in detected:
            preset = PRESETS[name]
            print(f"  ✓ {name:<12} {preset.key_env} is set")
        chosen = detected[0]
    else:
        print("  No API keys found in environment.")
        print("  Set at least one, then rerun setup:")
        print("    export OPENROUTER_API_KEY=sk-or-...")
        print("    export NVIDIA_NIM_API_KEY=nvapi-...")
        print("  Zero-cost test: OPENCLAW_PROVIDER=mock openclaw build 'anything'")
        chosen = "mock"

    toml_path = Path(".openclaw.toml")
    if not toml_path.exists():
        toml_path.write_text(
            "[openclaw]\n"
            f'provider = "{chosen}"\n'
            "iterations = 3\n"
            "threshold = 7.5\n"
            'workspace = "./openclaw_workspace"\n'
            '# task = "keep this app always deployable and well-tested"\n'
        )
        print(f"\nCreated {toml_path}  (provider: {chosen})")
    else:
        print(f"\n{toml_path} already exists — not overwritten.")

    print("\nNext steps:")
    print("  openclaw auto                       # detect + run from cwd")
    print("  openclaw build 'my app'              # build from prompt")
    print("  openclaw fix . --task 'fix tests'    # fix current repo")
    print("  openclaw providers                   # list all providers + models")
    return 0


def _build_cmd(args: argparse.Namespace) -> int:
    toml = _load_toml_config()
    _apply_toml_defaults(args, toml)
    task = _resolve_task(args)
    cfg = RunConfig(
        workspace=getattr(args, "workspace", "./openclaw_workspace"),
        max_critique_loops=getattr(args, "iterations", 3),
        quality_threshold=getattr(args, "threshold", 7.5),
        deploy=getattr(args, "deploy", False),
        deploy_prod=getattr(args, "prod", False),
        image_assets=getattr(args, "image", None) or [],
        verbose=not getattr(args, "quiet", False),
    )
    orch = Orchestrator(provider=get_provider(getattr(args, "provider", None),
                                              getattr(args, "model", None)), config=cfg)
    result = orch.run(task)
    _emit_result(result, getattr(args, "audit_out", "openclaw_audit.json"))
    return 0 if result.ok else 1


def _fix_cmd(args: argparse.Namespace) -> int:
    toml = _load_toml_config()
    _apply_toml_defaults(args, toml)
    repo = Path(args.repo).expanduser().resolve()
    if not repo.exists():
        print(f"error: repo not found: {repo}", file=sys.stderr)
        return 2
    task = getattr(args, "task", None) or "Analyze this repo, identify the most important issues, and fix them."
    cfg = RunConfig(
        workspace=str(repo),
        max_critique_loops=getattr(args, "iterations", 3),
        quality_threshold=getattr(args, "threshold", 7.5),
        deploy=False,
        verbose=not getattr(args, "quiet", False),
    )
    orch = Orchestrator(provider=get_provider(getattr(args, "provider", None),
                                              getattr(args, "model", None)), config=cfg)
    result = orch.run(task, repo_path=str(repo))
    _emit_result(result, getattr(args, "audit_out", "openclaw_audit.json"))
    return 0 if result.ok else 1


def _deploy_cmd(args: argparse.Namespace) -> int:
    repo = Path(args.repo).expanduser().resolve()
    cfg = RunConfig(workspace=str(repo), deploy=True, deploy_prod=args.prod,
                    verbose=not args.quiet, max_critique_loops=1)
    task = f"Deploy this project to Vercel ({'production' if args.prod else 'preview'})."
    orch = Orchestrator(provider=get_provider(args.provider, args.model), config=cfg)
    result = orch.run(task, repo_path=str(repo))
    _emit_result(result, args.audit_out)
    return 0 if result.ok else 1


def _analyze_cmd(args: argparse.Namespace) -> int:
    summary = analyze_repo(args.repo)
    print(json.dumps(summary, indent=2))
    return 0


def _serve_cmd(args: argparse.Namespace) -> int:
    import uvicorn
    uvicorn.run("openclaw.server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


# ── helpers ──────────────────────────────────────────────────────────────────

def _resolve_task(args: argparse.Namespace) -> str:
    if getattr(args, "file", None):
        return Path(args.file).read_text().strip()
    if getattr(args, "task", None):
        return args.task.strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    print("error: no task provided. Pass it as an arg, --file, or via stdin.", file=sys.stderr)
    sys.exit(1)


def _emit_result(result, audit_out: str) -> None:
    print("\n" + "─" * 60)
    print("  FINAL OUTPUT")
    print("─" * 60)
    print(result.final_output)
    print(result.seal)
    if result.deploy_url:
        print(f"DEPLOY_URL: {result.deploy_url}")
    Path(audit_out).write_text(json.dumps(result.audit.to_dict(), indent=2, default=str))
    print(f"\naudit log → {audit_out}", file=sys.stderr)


# ── argparse wiring ──────────────────────────────────────────────────────────

def _providers_cmd(args: argparse.Namespace) -> int:
    print(f"{'name':<12}  {'default model':<40}  description")
    print("-" * 100)
    for name, preset in PRESETS.items():
        roles = f" (+{len(preset.role_models)} role overrides)" if preset.role_models else ""
        print(f"{name:<12}  {preset.default_model:<40}  {preset.description}{roles}")
    print("\nSelect with:  --provider <name>   or   export OPENCLAW_PROVIDER=<name>")
    print("Override a role model:  export OPENCLAW_MODEL_<ROLE>=<model>   "
          "(e.g. OPENCLAW_MODEL_CRITIC=llama-3.1-8b-instant)")
    return 0


def _common_run_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", choices=list(PRESETS.keys()), default=None,
                   help="LLM provider (default: from $OPENCLAW_PROVIDER or auto-detect)")
    p.add_argument("--model", default=None,
                   help="Model name (default: from $OPENCLAW_MODEL or provider default)")
    p.add_argument("--iterations", "-i", type=int, default=3,
                   help="Max critique loops (default: 3)")
    p.add_argument("--threshold", "-t", type=float, default=7.5,
                   help="Quality threshold 0-10 (default: 7.5)")
    p.add_argument("--audit-out", default="openclaw_audit.json", help="Audit log path")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="openclaw",
        description="Autonomous multi-agent app builder — drop in a prompt or a repo.",
    )
    p.add_argument("--version", action="version", version=f"openclaw {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    # auto — zero-effort entry point
    au = sub.add_parser("auto", help="Detect intent from cwd and run (reads .openclaw.toml)")
    au.add_argument("task", nargs="?", help="Override task (default: from .openclaw.toml or auto-detected)")
    _common_run_flags(au)
    au.set_defaults(func=_auto_cmd)

    # setup — one-time wizard
    se = sub.add_parser("setup", help="Detect API keys and create .openclaw.toml")
    se.set_defaults(func=_setup_cmd)

    # build
    b = sub.add_parser("build", help="Build from a prompt")
    b.add_argument("task", nargs="?", help="Prompt describing what to build")
    b.add_argument("--file", "-f", help="Read prompt from file")
    b.add_argument("--workspace", "-w", default="./openclaw_workspace", help="Where to write files")
    b.add_argument("--deploy", action="store_true", help="Deploy to Vercel after building")
    b.add_argument("--prod", action="store_true", help="Use --prod for vercel deploy")
    b.add_argument("--image", action="append", help="Image asset prompt (repeatable)")
    _common_run_flags(b)
    b.set_defaults(func=_build_cmd)

    # fix
    fx = sub.add_parser("fix", help="Analyze and fix an existing repo")
    fx.add_argument("repo", help="Path to repo")
    fx.add_argument("--task", help="What to fix (default: general improvement)")
    _common_run_flags(fx)
    fx.set_defaults(func=_fix_cmd)

    # deploy
    d = sub.add_parser("deploy", help="Deploy a repo to Vercel")
    d.add_argument("repo", help="Path to repo")
    d.add_argument("--prod", action="store_true", help="Deploy to production")
    _common_run_flags(d)
    d.set_defaults(func=_deploy_cmd)

    # analyze
    a = sub.add_parser("analyze", help="Print a JSON summary of a repo (no LLM calls)")
    a.add_argument("repo", help="Path to repo")
    a.set_defaults(func=_analyze_cmd)

    # serve
    s = sub.add_parser("serve", help="Start the FastAPI server")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=_serve_cmd)

    # providers
    pv = sub.add_parser("providers", help="List available LLM provider presets and their models")
    pv.set_defaults(func=_providers_cmd)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
