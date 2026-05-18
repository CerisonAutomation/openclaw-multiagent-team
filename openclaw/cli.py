"""openclaw CLI — drop in a prompt or a repo and let agents take over.

Examples:
    openclaw build "a Next.js todo app with dark mode"
    openclaw fix ./my-repo --task "the build is failing on tailwind"
    openclaw deploy ./my-repo --prod
    openclaw analyze ./my-repo
    openclaw serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openclaw import __version__
from openclaw.audit import AuditLog
from openclaw.models import PRESETS, list_presets
from openclaw.orchestrator import Orchestrator, RunConfig
from openclaw.providers import get_provider
from openclaw.tools import analyze_repo


def _build_cmd(args: argparse.Namespace) -> int:
    task = _resolve_task(args)
    cfg = RunConfig(
        workspace=args.workspace,
        max_critique_loops=args.iterations,
        quality_threshold=args.threshold,
        deploy=args.deploy,
        deploy_prod=args.prod,
        image_assets=args.image or [],
        verbose=not args.quiet,
    )
    orch = Orchestrator(provider=get_provider(args.provider, args.model), config=cfg)
    result = orch.run(task)
    _emit_result(result, args.audit_out)
    return 0 if result.ok else 1


def _fix_cmd(args: argparse.Namespace) -> int:
    repo = Path(args.repo).expanduser().resolve()
    if not repo.exists():
        print(f"error: repo not found: {repo}", file=sys.stderr)
        return 2
    task = args.task or "Analyze this repo, identify the most important issues, and fix them."
    cfg = RunConfig(
        workspace=str(repo),
        max_critique_loops=args.iterations,
        quality_threshold=args.threshold,
        deploy=False,
        verbose=not args.quiet,
    )
    orch = Orchestrator(provider=get_provider(args.provider, args.model), config=cfg)
    result = orch.run(task, repo_path=str(repo))
    _emit_result(result, args.audit_out)
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


# ── helpers ─────────────────────────────────────────────────────────────────

def _resolve_task(args: argparse.Namespace) -> str:
    if args.file:
        return Path(args.file).read_text().strip()
    if args.task:
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


# ── argparse wiring ─────────────────────────────────────────────────────────

def _providers_cmd(args: argparse.Namespace) -> int:
    print(f"{'name':<12}  {'default model':<40}  description")
    print("-" * 100)
    for name, preset in PRESETS.items():
        default = preset.default_model
        roles = f" (+{len(preset.role_models)} role overrides)" if preset.role_models else ""
        print(f"{name:<12}  {default:<40}  {preset.description}{roles}")
    print(f"\nSelect with:  --provider <name>   or   export OPENCLAW_PROVIDER=<name>")
    print(f"Override one role's model:  export OPENCLAW_MODEL_<ROLE>=<model>   "
          f"(e.g. OPENCLAW_MODEL_CRITIC=llama-3.1-8b-instant)")
    return 0


def _common_run_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", choices=list(PRESETS.keys()), default=None,
                   help="LLM provider (default: from $OPENCLAW_PROVIDER or auto-detect)")
    p.add_argument("--model", default=None, help="Model name (default: from $OPENCLAW_MODEL or provider default)")
    p.add_argument("--iterations", "-i", type=int, default=3, help="Max critique loops (default: 3)")
    p.add_argument("--threshold", "-t", type=float, default=7.5, help="Quality threshold 0-10 (default: 7.5)")
    p.add_argument("--audit-out", default="openclaw_audit.json", help="Audit log path")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="openclaw",
        description="Autonomous multi-agent app builder — drop in a prompt or a repo.",
    )
    p.add_argument("--version", action="version", version=f"openclaw {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

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
    pv = sub.add_parser("providers", help="List available LLM provider presets")
    pv.set_defaults(func=_providers_cmd)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
