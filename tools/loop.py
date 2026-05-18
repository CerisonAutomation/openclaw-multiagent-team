#!/usr/bin/env python3
"""
Ralph Wiggum Loop runner for openclaw.

Wraps the `claude` CLI in a bounded autonomous iteration loop.
Claude keeps working until it writes the completion token to the promise file,
or the max-iteration cap is hit.

Usage:
    python tools/loop.py "Fix all failing tests"
    python tools/loop.py --task "Add docstrings to tools/inspector/tools.py" \\
                         --promise "DOCS_DONE" --max-iter 15
    python tools/loop.py --task-file task.md --promise BUILD_OK --max-iter 20 --model opus

Architecture:
  1. Write task + acceptance criteria + instructions to a temp prompt file.
  2. Export LOOP_PROMISE_TOKEN, LOOP_PROMISE_FILE, LOOP_MAX_ITER into env.
  3. The stop hook (.claude/hooks/stop_check.py) reads those vars on every turn end.
  4. If the promise file contains the token → hook exits 0 (done).
  5. If the token is missing → hook exits 2 (block stop, reinject message).
  6. Loop ends when the promise is met OR max iterations are exhausted.

Flow:
    task + criteria
         │
         ▼
    [Claude plans → edits → runs tools]
         │
         ▼
    [Stop hook fires]
         │
    promise present? ──yes──▶ exit cleanly
         │
         no
         │
    max_iter hit? ──yes──▶ exit with warning
         │
         no
         │
    reinject "keep working" message ──▶ back to Claude
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time


PROMISE_FILE = ".loop_done"
ITER_FILE = ".loop_iter"

TASK_TEMPLATE = textwrap.dedent("""\
    ## Task
    {task}

    ## Acceptance criteria
    {criteria}

    ## Completion protocol
    When **all** acceptance criteria are satisfied, write the exact token below (and nothing else)
    into the file `{promise_file}`:

        {token}

    Do NOT write the token until the work is genuinely done and verified.
    Use tools (tests, lint, build) to verify before writing the token.
""")


def _build_prompt(task: str, criteria: str, token: str) -> str:
    return TASK_TEMPLATE.format(
        task=task,
        criteria=criteria or "The task is complete and the code works correctly.",
        promise_file=PROMISE_FILE,
        token=token,
    )


def _cleanup() -> None:
    for f in (PROMISE_FILE, ITER_FILE):
        try:
            os.remove(f)
        except FileNotFoundError:
            pass


def run_loop(
    task: str,
    criteria: str = "",
    token: str = "LOOP_DONE",
    max_iter: int = 20,
    model: str | None = None,
    dry_run: bool = False,
) -> int:
    if not shutil.which("claude"):
        print("ERROR: `claude` CLI not found in PATH. Install Claude Code first:", file=sys.stderr)
        print("  npm install -g @anthropic-ai/claude-code", file=sys.stderr)
        return 1

    _cleanup()

    prompt = _build_prompt(task, criteria, token)

    env = {
        **os.environ,
        "LOOP_PROMISE_TOKEN": token,
        "LOOP_PROMISE_FILE": PROMISE_FILE,
        "LOOP_MAX_ITER": str(max_iter),
    }

    cmd = ["claude", "--print", prompt]
    if model:
        cmd += ["--model", model]

    print(f"╔══ Ralph Wiggum Loop ═══════════════════════════════════════════")
    print(f"║  task       : {task[:72]}")
    print(f"║  token      : {token}")
    print(f"║  max_iter   : {max_iter}")
    print(f"║  model      : {model or 'default'}")
    print(f"╚════════════════════════════════════════════════════════════════")
    print()

    if dry_run:
        print("[dry-run] Would execute:", " ".join(cmd))
        print("[dry-run] Env overrides:", {k: v for k, v in env.items() if k.startswith("LOOP_")})
        return 0

    start = time.time()
    try:
        result = subprocess.run(cmd, env=env)
        elapsed = time.time() - start
        print()
        print(f"Loop finished in {elapsed:.1f}s — exit code: {result.returncode}")

        # Check promise state
        try:
            content = open(PROMISE_FILE).read()
            if token in content:
                print(f"✓ Promise '{token}' fulfilled.")
                _cleanup()
                return 0
        except FileNotFoundError:
            pass

        iter_count = 0
        try:
            iter_count = int(open(ITER_FILE).read().strip())
        except Exception:
            pass

        print(f"✗ Promise not fulfilled after {iter_count} iteration(s).")
        _cleanup()
        return result.returncode or 1

    except KeyboardInterrupt:
        print("\nLoop interrupted by user.")
        _cleanup()
        return 130


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ralph Wiggum Loop — autonomous bounded Claude Code iteration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python tools/loop.py "Fix all failing tests"
              python tools/loop.py --task "Refactor tools.py" --criteria "ruff passes, tests green" \\
                                   --promise REFACTOR_DONE --max-iter 15
              python tools/loop.py --task-file my_task.md --max-iter 10 --dry-run
        """),
    )
    ap.add_argument("task", nargs="?", default="", help="Task description (positional shorthand)")
    ap.add_argument("--task", dest="task_flag", default="", help="Task description")
    ap.add_argument("--task-file", default="", help="Read task from a file")
    ap.add_argument("--criteria", default="", help="Acceptance criteria (plain text)")
    ap.add_argument("--promise", default="LOOP_DONE", help="Completion token Claude must write")
    ap.add_argument("--max-iter", type=int, default=20, help="Hard iteration cap (default: 20)")
    ap.add_argument("--model", default="", help="Claude model override")
    ap.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    args = ap.parse_args()

    task = args.task_flag or args.task or ""
    if args.task_file:
        try:
            task = open(args.task_file).read().strip()
        except OSError as e:
            print(f"ERROR reading task file: {e}", file=sys.stderr)
            sys.exit(1)

    if not task:
        ap.print_help()
        sys.exit(1)

    sys.exit(run_loop(
        task=task,
        criteria=args.criteria,
        token=args.promise,
        max_iter=args.max_iter,
        model=args.model or None,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
