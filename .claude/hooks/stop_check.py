#!/usr/bin/env python3
"""
Ralph Wiggum stop hook — blocks Claude from stopping until the completion promise is met.

Claude Code calls this at the end of every turn when a Stop event fires.
stdin: JSON payload  { "session_id": "...", "transcript_path": "...", "stop_hook_active": bool }
exit 0  → allow stop (task done or no loop active)
exit 2  → block stop (continue working); stdout is fed back to Claude as a user message

Environment variables (set by tools/loop.py or by the user):
  LOOP_PROMISE_TOKEN   – the exact string Claude must write into LOOP_PROMISE_FILE to finish
  LOOP_PROMISE_FILE    – path to the file Claude writes its completion token into (default: .loop_done)
  LOOP_MAX_ITER        – hard cap checked against .loop_iter counter file (default: 20)
"""
import json
import os
import sys


def main() -> None:
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except Exception:
        data = {}

    # --- Idempotency guard ---------------------------------------------------
    # stop_hook_active=true means a Stop hook already ran this turn.
    # Without this guard a failed hook re-triggers itself → infinite loop.
    if data.get("stop_hook_active"):
        sys.exit(0)

    token = os.environ.get("LOOP_PROMISE_TOKEN", "").strip()
    if not token:
        # No active loop — let Claude exit normally.
        sys.exit(0)

    promise_file = os.environ.get("LOOP_PROMISE_FILE", ".loop_done")
    max_iter = int(os.environ.get("LOOP_MAX_ITER", "20"))
    iter_file = ".loop_iter"

    # --- Iteration counter ---------------------------------------------------
    current = 0
    try:
        current = int(open(iter_file).read().strip())
    except Exception:
        pass
    current += 1
    try:
        open(iter_file, "w").write(str(current))
    except Exception:
        pass

    if current >= max_iter:
        print(
            f"[stop_hook] Max iterations reached ({current}/{max_iter}). "
            f"Stopping regardless of promise state.",
            file=sys.stderr,
        )
        _cleanup(iter_file)
        sys.exit(0)

    # --- Promise check -------------------------------------------------------
    try:
        content = open(promise_file).read()
        if token in content:
            print(f"[stop_hook] Promise '{token}' found after {current} iteration(s). Done.", file=sys.stderr)
            _cleanup(iter_file)
            _cleanup(promise_file)
            sys.exit(0)
    except FileNotFoundError:
        pass

    # --- Block stop: tell Claude to keep going -------------------------------
    print(
        f"Loop iteration {current}/{max_iter}. "
        f"Promise token '{token}' not yet written to '{promise_file}'. "
        f"Continue working until the task is complete, then write the token to that file."
    )
    sys.exit(2)


def _cleanup(path: str) -> None:
    try:
        os.remove(path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
