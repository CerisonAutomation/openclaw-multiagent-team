"""openclaw TUI — one command, one agent, everything routed automatically.

    openclaw        # or: make

The Brain classifies your intent and routes to the right system:
  build / deploy   →  7-phase SINGULARITY orchestrator  (Jan or Ollama locally)
  fix / review     →  inspector relay  (code analysis + LLM skills in parallel)
  explain / critic →  relay (local LLM — Jan / Ollama auto-detected)

Paste code on lines after your message; blank line submits.
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env before anything else (JAN_MODEL, JAN_BASE, OLLAMA_BASE …)
def _load_dotenv() -> None:
    for candidate in [Path(os.getcwd()) / ".env", _ROOT / ".env"]:
        if candidate.exists():
            for raw in candidate.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return

_load_dotenv()

from openclaw.brain import Brain  # noqa: E402


# ── ANSI ──────────────────────────────────────────────────────────────────────

_TTY = sys.stdout.isatty()
_C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
}


def c(color: str, text: str) -> str:
    return f"{_C.get(color,'')}{text}{_C['reset']}" if _TTY else text


def _hr(width: int = 58) -> str:
    return c("dim", "─" * width)


# ── Spinner ───────────────────────────────────────────────────────────────────

class _Spinner:
    """Shows a live status line while the brain is working."""
    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self) -> None:
        self._msg = ""
        self._running = False
        self._thread: threading.Thread | None = None

    def update(self, msg: str) -> None:
        self._msg = msg

    def start(self) -> None:
        if not _TTY:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        if _TTY:
            sys.stdout.write("\r" + " " * 70 + "\r")
            sys.stdout.flush()

    def _loop(self) -> None:
        i = 0
        while self._running:
            frame = self._FRAMES[i % len(self._FRAMES)]
            line = f"\r  {c('cyan', frame)} {c('dim', self._msg)}"
            sys.stdout.write(line[:72])
            sys.stdout.flush()
            i += 1
            time.sleep(0.1)


# ── Banner + provider check ───────────────────────────────────────────────────

def _banner() -> None:
    print(c("cyan",  "╔══════════════════════════════════════════════════════════╗"))
    print(c("cyan",  "║") +
          c("bold",  "   openclaw  ·  relay-orchestrator-brain  ·  local LLMs   ") +
          c("cyan",  "║"))
    print(c("cyan",  "╚══════════════════════════════════════════════════════════╝"))
    print(c("dim",   "  ask anything  ·  paste code on the next lines"))
    print(c("dim",   "  blank line = submit  ·  :help for commands  ·  Ctrl-C exits"))
    print()


async def _show_providers(brain: Brain) -> bool:
    st = await brain.status_check()
    ollama = st["ollama"]
    jan    = st["jan"]
    active = st["active"]

    if ollama["ok"]:
        models = ", ".join(ollama["models"][:4]) or "—"
        print(c("green", "  ✓ Ollama") + c("dim", f"  {models}"))
    else:
        print(c("dim",   f"  ✗ Ollama   {ollama['error'] or 'offline'}"))

    if jan["ok"]:
        models = ", ".join(jan["models"][:3]) or "—"
        print(c("green", "  ✓ Jan    ") + c("dim", f"  {models}"))
    else:
        print(c("dim",   f"  ✗ Jan      {jan['error'] or 'offline — open the Jan app'}"))

    if active == "none":
        print()
        print(c("red", "  No LLM provider available."))
        print(c("dim", "  Start Ollama:   ollama serve"))
        print(c("dim", "  Or open Jan and make sure the server is running on :1337"))
        return False

    print()
    return True


# ── Input ─────────────────────────────────────────────────────────────────────

def _read_input(prompt_str: str) -> str:
    """Read one turn. First line = message; following lines = source (blank submits)."""
    try:
        first = input(prompt_str)
    except (EOFError, KeyboardInterrupt):
        return ":quit"

    if not first.strip():
        return ""
    if first.strip().startswith(":"):
        return first.strip()

    lines = [first]
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        lines.append(line)
    return "\n".join(lines)


# ── Render ────────────────────────────────────────────────────────────────────

def _render(result) -> None:  # result: BrainResult
    print()
    print(_hr())

    if result.was_polished:
        print(c("yellow", "  ↳ prompt polished by trait system"))
        print(c("dim",    f"    {result.polished_prompt[:100]}"))
        print()

    print(result.message)
    print()

    # Footer
    pi       = result.intent.get("primary_intent", "?")
    domain   = result.intent.get("domain", "?")
    comp     = result.intent.get("complexity", "?")
    route_c  = c("magenta", result.route)
    prov_c   = (c("green", result.provider)
                if result.provider not in ("none", "", "cloud")
                else c("dim", result.provider or "—"))
    model_c  = c("dim",   result.model or "—")
    lat_c    = c("dim",   f"{result.latency_ms:.0f}ms")
    tools_c  = c("dim",   " · ".join(result.tools_used[:6]) if result.tools_used else "—")

    print(c("dim", f"  {pi} · {domain} · {comp}") +
          f"  {route_c}  {prov_c}  {model_c}  {lat_c}")
    if result.tools_used:
        print(c("dim", f"  {tools_c}"))
    if result.error:
        print(c("red", f"  error: {result.error[:120]}"))
    print()


# ── Main TUI loop ─────────────────────────────────────────────────────────────

async def _run() -> int:
    _banner()
    brain = Brain()

    ok = await _show_providers(brain)
    if not ok:
        return 1

    spinner = _Spinner()

    while True:
        raw = _read_input(c("bold", c("blue", "you")) + c("dim", " ❯ "))

        if not raw.strip():
            continue

        cmd = raw.strip().lower()

        # ── commands ──
        if cmd in (":quit", ":exit", ":q"):
            print(c("dim", "bye."))
            break

        if cmd == ":clear":
            brain.clear_history()
            print(c("dim", "history cleared."))
            continue

        if cmd == ":history":
            hist = brain.history()
            if not hist:
                print(c("dim", "  (no history)"))
            else:
                for t in hist[-12:]:
                    role = c("bold", "you") if t["role"] == "user" else c("cyan", "agent")
                    snippet = t["content"][:100].replace("\n", " ")
                    print(f"  {role}: {snippet}")
            print()
            continue

        if cmd == ":providers":
            await _show_providers(brain)
            continue

        if cmd == ":help":
            print(c("bold", "Commands"))
            print("  :clear      clear conversation history (also deletes .openclaw_history.json)")
            print("  :history    show last 12 turns")
            print("  :providers  live provider health check")
            print("  :quit       exit")
            print()
            print(c("bold", "How to use"))
            print("  Line 1: your request in plain English")
            print("  Lines 2+: paste code or text for analysis")
            print("  Blank line: submit")
            print()
            print(c("bold", "Routing"))
            print("  build / deploy   →  7-phase orchestrator (uses Jan/Ollama locally)")
            print("  fix / review     →  relay: code metrics + secrets + lint + LLM skills")
            print("  explain / critic →  relay: local LLM analysis")
            print()
            print(c("bold", "Prompt polish"))
            print("  Short/vague requests (<10 words) are auto-improved")
            print("  using the 5-trait system before being sent to the LLM.")
            print()
            continue

        # ── normal turn ──
        lines = raw.split("\n")
        user_msg = lines[0].strip()
        source   = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

        # Run brain with spinner showing live status
        spinner.start()
        result_holder: list = []

        def _on_status(msg: str) -> None:
            spinner.update(msg)

        try:
            brain_result = await brain.handle(user_msg, source=source, on_status=_on_status)
            result_holder.append(brain_result)
        except Exception as exc:
            spinner.stop()
            print(c("red", f"  error: {exc}"))
            continue
        finally:
            spinner.stop()

        _render(result_holder[0])

    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        print()
        return 0
