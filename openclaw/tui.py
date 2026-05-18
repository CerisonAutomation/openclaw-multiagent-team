"""openclaw TUI — single entry point. Type a request, the relay does everything.

Flow per turn:
  your text (+ optional pasted code)
      │
      ▼ prompt_polish (if vague — applies trait system)
      │
      ▼ intent classifier  →  tool selection
      │
      ▼ parallel: metrics · secrets · lint · AST · LLM skills (Jan/Ollama)
      │
      ▼ synthesized response in terminal

Usage:
    openclaw          # no args → starts TUI
    openclaw tui      # explicit
    make              # Makefile shortcut
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is importable when running from the installed package.
# Works for both `pip install -e .` (editable) and running from source.
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env from cwd if present — picks up JAN_MODEL, JAN_BASE, OLLAMA_BASE etc.
def _load_dotenv() -> None:
    env_file = Path(os.getcwd()) / ".env"
    if not env_file.exists():
        env_file = _ROOT / ".env"
    if env_file.exists():
        for raw in env_file.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_dotenv()

try:
    from tools.inspector.agents import skill_index
    from tools.inspector.relay import RelayAgent
except ImportError as exc:
    print(f"[openclaw] could not import inspector: {exc}")
    print("  Run from the project root or install with: pip install -e .")
    sys.exit(1)


# ── ANSI helpers ──────────────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty()

_C: dict[str, str] = {
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "dim":     "\033[2m",
    "red":     "\033[31m",
    "green":   "\033[32m",
    "yellow":  "\033[33m",
    "blue":    "\033[34m",
    "magenta": "\033[35m",
    "cyan":    "\033[36m",
    "white":   "\033[37m",
}


def c(color: str, text: str) -> str:
    return f"{_C.get(color,'')}{text}{_C['reset']}" if _USE_COLOR else text


def _hr(ch: str = "─", width: int = 56) -> str:
    return c("dim", ch * width)


# ── Banner + provider check ───────────────────────────────────────────────────

def _banner() -> None:
    print(c("cyan", "╔══════════════════════════════════════════════════════╗"))
    print(c("cyan", "║") + c("bold", "   openclaw  ·  multi-agent relay  ·  local LLMs    ") + c("cyan", "║"))
    print(c("cyan", "╚══════════════════════════════════════════════════════╝"))
    print(c("dim",  "  type your request  ·  paste code on following lines"))
    print(c("dim",  "  blank line = submit  ·  :help  ·  Ctrl-C to exit"))
    print()


async def _check_providers() -> bool:
    """Print provider status, return True if at least one is available."""
    ollama_st = await skill_index.ollama.status()
    jan_st    = await skill_index.jan.status()

    if ollama_st.available:
        models = ", ".join(ollama_st.models[:5]) or "—"
        print(c("green", "  ✓ Ollama") + c("dim", f"  {models}"))
    else:
        print(c("dim",   f"  ✗ Ollama   {ollama_st.error or 'offline'}"))

    if jan_st.available:
        models = ", ".join(jan_st.models[:3]) or "—"
        print(c("green", "  ✓ Jan    ") + c("dim", f"  {models}"))
    else:
        print(c("dim",   f"  ✗ Jan      {jan_st.error or 'offline — start the Jan app'}"))

    if not ollama_st.available and not jan_st.available:
        print()
        print(c("red",  "  No LLM provider available."))
        print(c("dim",  "  Start Ollama:  ollama serve"))
        print(c("dim",  "  Or open Jan and ensure the server is running (:1337)"))
        return False

    print()
    return True


# ── Input reader ──────────────────────────────────────────────────────────────

def _read_input(prompt_str: str) -> str:
    """
    Read one turn of input.  First line is the message; additional lines
    (until a blank line) are appended as source code / extra context.
    Returns ":quit" on EOF.
    """
    try:
        first = input(prompt_str)
    except EOFError:
        return ":quit"
    except KeyboardInterrupt:
        return ":quit"

    if not first.strip():
        return ""
    if first.strip().startswith(":"):
        return first.strip()

    lines = [first]
    # Read continuation lines
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        lines.append(line)

    return "\n".join(lines)


# ── Response renderer ─────────────────────────────────────────────────────────

def _render(result) -> None:  # result: RelayResult
    print()
    print(_hr())
    print(result.message)
    print()

    # Metadata footer
    intent     = result.intent
    pi         = intent.get("primary_intent", "?")
    domain     = intent.get("domain", "?")
    complexity = intent.get("complexity", "?")

    provider_col = c("green", result.provider) if result.provider not in ("none", "") else c("red", "none")
    model_col    = c("dim",   result.model or "—")
    tools_col    = c("dim",   " · ".join(result.tools_used) if result.tools_used else "—")
    lat_col      = c("dim",   f"{result.latency_ms:.0f}ms")

    print(c("dim", f"  {pi} · {domain} · {complexity}") +
          f"  {provider_col} {model_col}  {lat_col}")
    if result.tools_used:
        print(c("dim", f"  tools: {tools_col}"))
    print()


# ── Prompt polish ─────────────────────────────────────────────────────────────

async def _maybe_polish(user_msg: str) -> tuple[str, str]:
    """
    If the message is short/vague, run prompt_polish to apply the trait system.
    Returns (display_note, improved_msg).  display_note is empty if not polished.
    """
    if len(user_msg.split()) >= 12:
        return "", user_msg  # already detailed enough

    result = await skill_index.run("prompt_polish", user_msg)
    if not result.ok or not isinstance(result.output, dict):
        return "", user_msg

    out = result.output
    if not out.get("was_vague", True):
        return "", user_msg

    improved = out.get("improved_prompt", "").strip()
    reason   = out.get("reason", "")
    if not improved or improved == user_msg:
        return "", user_msg

    note = (c("yellow", "  ↳ prompt polished") +
            c("dim",   f"  ({reason})"))
    return note, improved


# ── Main TUI loop ─────────────────────────────────────────────────────────────

async def _run() -> int:
    _banner()
    ok = await _check_providers()
    if not ok:
        return 1

    relay = RelayAgent(skill_index)

    while True:
        raw = _read_input(c("bold", c("blue", "you")) + c("dim", " ❯ "))

        if not raw.strip():
            continue

        cmd = raw.strip().lower()

        # ── built-in commands ──
        if cmd in (":quit", ":exit", ":q"):
            print(c("dim", "bye."))
            break

        if cmd == ":clear":
            relay.clear_history()
            print(c("dim", "history cleared."))
            continue

        if cmd == ":history":
            hist = relay.history()
            if not hist:
                print(c("dim", "  (no history)"))
            for msg in hist:
                role = c("bold", "you") if msg["role"] == "user" else c("cyan", "agent")
                print(f"{role}: {msg['content'][:120]}")
            print()
            continue

        if cmd == ":providers":
            await _check_providers()
            continue

        if cmd == ":help":
            print(c("bold", "Commands"))
            print("  :clear      clear conversation history")
            print("  :history    show last turns")
            print("  :providers  show LLM backend status")
            print("  :quit       exit")
            print()
            print(c("bold", "Input format"))
            print("  Line 1:     your request in plain English")
            print("  Lines 2+:   paste code / text for analysis (blank line to submit)")
            print()
            print(c("bold", "What the agent does automatically"))
            print("  · classifies your intent")
            print("  · polishes vague prompts using the trait system")
            print("  · runs code metrics, secret scanning, linting, AST analysis")
            print("  · runs LLM skills (critic, fix, security, arch_review, test_gen…)")
            print("  · synthesizes everything into one response")
            print("  · uses Jan or Ollama — whichever is available")
            print()
            continue

        # ── normal turn ──
        lines = raw.split("\n")
        user_msg = lines[0].strip()
        source   = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

        # Apply prompt polish to the message (not the source code)
        print(c("dim", "  thinking…"), end="\r", flush=True)
        polish_note, user_msg = await _maybe_polish(user_msg)

        if polish_note:
            print(f"          \r{polish_note}")  # clear "thinking…"

        try:
            result = await relay.handle(user_msg, source=source)
        except Exception as exc:
            print(c("red", f"  error: {exc}"))
            continue

        _render(result)

    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        print()
        return 0
