"""
Relay agent — sits between user input and tools/skills.

Flow per request:
  User input + optional source code
       │
       ▼
  1. Intent classification (intent skill)
       │
       ▼
  2. Tool selection (rule-based on intent fields)
       │
       ▼
  3. Parallel execution:
       ├─ Code analysis tools (metrics, secrets, lint, AST) — synchronous, fast
       └─ LLM skills (critic, fix, test_gen, …) — async Ollama/Jan calls
       │
       ▼
  4. Synthesis (summarize skill or template assembly)
       │
       ▼
  RelayResult {message, intent, tools_used, tool_outputs, model, provider, latency_ms}

The relay maintains a per-session message history (last MAX_HISTORY turns) so
subsequent calls can reference prior context.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from .agents import SkillIndex, SkillResult
from .tools import (
    ASTMetrics,
    analyze_python_ast,
    compute_metrics,
    lint_python,
    scan_secrets,
)


MAX_HISTORY = 10  # turns kept per session (1 turn = 1 user + 1 assistant msg)


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class RelayMessage:
    role: str          # "user" | "assistant"
    content: str
    tools_used: list[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)

    def to_llm_msg(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class RelayResult:
    message: str               # human-readable response
    intent: dict[str, Any]     # raw intent skill output
    tools_used: list[str]      # tool/skill names that were invoked
    tool_outputs: dict[str, Any]  # raw per-tool results
    model: str                 # model that ran intent classification
    provider: str              # "ollama" | "jan" | "none"
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "intent": self.intent,
            "tools_used": self.tools_used,
            "tool_outputs": self.tool_outputs,
            "model": self.model,
            "provider": self.provider,
            "latency_ms": self.latency_ms,
        }


# ── Tool selection rules ──────────────────────────────────────────────────────

# intent.primary_intent → list of (tool_type, name) pairs to run
_INTENT_TOOL_MAP: dict[str, list[tuple[str, str]]] = {
    "review":  [("skill", "summarize"), ("skill", "critic"), ("code", "metrics")],
    "fix":     [("skill", "fix"), ("skill", "security")],
    "test":    [("skill", "test_gen")],
    "build":   [("skill", "arch_review"), ("skill", "summarize")],
    "deploy":  [("skill", "arch_review"), ("skill", "security")],
    "explain": [("skill", "summarize"), ("code", "metrics")],
    "data":    [("skill", "critic"), ("code", "metrics")],
    "unknown": [("skill", "summarize"), ("code", "metrics")],
}

_SECURITY_DOMAINS = {"backend", "fullstack", "api", "devops"}


def _select_tools(
    intent: dict[str, Any],
    has_source: bool,
) -> list[tuple[str, str]]:
    """Return ordered list of (tool_type, name) pairs to execute."""
    if not has_source:
        return []

    primary = intent.get("primary_intent", "unknown")
    domain = intent.get("domain", "other")
    lang = intent.get("language", "unknown")

    base = list(_INTENT_TOOL_MAP.get(primary, _INTENT_TOOL_MAP["unknown"]))

    # Always run secrets scan on code
    base.append(("code", "secrets"))

    # Add security skill for high-risk domains/intents
    if domain in _SECURITY_DOMAINS and ("skill", "security") not in base:
        base.append(("skill", "security"))

    # Python-specific: add lint and AST
    if lang == "python":
        base.append(("code", "lint"))
        base.append(("code", "ast"))

    # Deduplicate while preserving order
    seen: set[tuple[str, str]] = set()
    result = []
    for item in base:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# ── Relay agent ───────────────────────────────────────────────────────────────

class RelayAgent:
    """
    Orchestrates intent → tool selection → execution → synthesis.

    One instance per session; holds message history for context continuity.
    """

    def __init__(self, skill_index: SkillIndex) -> None:
        self._si = skill_index
        self._history: list[RelayMessage] = []

    # ── Public API ────────────────────────────────────────────────────────────

    async def handle(
        self,
        user_input: str,
        source: str = "",
        model: str | None = None,
    ) -> RelayResult:
        """
        Process one turn: classify → select → execute → synthesize.

        Args:
            user_input: Natural language message from the user.
            source:     Optional code/text to analyse (may be empty).
            model:      Force a specific model (overrides auto-selection).
        """
        t0 = time.time()

        # Step 1: Classify intent
        intent_src = f"{user_input}\n\n---\n{source}" if source else user_input
        intent_result = await self._si.run("intent", intent_src, model=model)
        intent = intent_result.output if isinstance(intent_result.output, dict) else {}
        provider = intent_result.provider  # "ollama" | "jan" | "none"

        # Step 2: Select tools
        selected = _select_tools(intent, bool(source.strip()))

        # Step 3: Execute — split into fast sync (code analysis) and async (skills)
        tool_outputs: dict[str, Any] = {}
        skill_tasks: dict[str, asyncio.Task[SkillResult]] = {}

        for tool_type, name in selected:
            if tool_type == "code":
                tool_outputs[name] = await asyncio.to_thread(
                    self._run_code_tool, name, source, intent
                )
            elif tool_type == "skill":
                task = asyncio.create_task(
                    self._si.run(name, source or user_input, model=model)
                )
                skill_tasks[name] = task

        # Await all skill tasks concurrently
        if skill_tasks:
            results = await asyncio.gather(*skill_tasks.values(), return_exceptions=True)
            for skill_name, result in zip(skill_tasks.keys(), results):
                if isinstance(result, Exception):
                    tool_outputs[skill_name] = {"error": str(result)}
                elif isinstance(result, SkillResult):
                    tool_outputs[skill_name] = result.output
                    if not result.ok:
                        tool_outputs[skill_name] = {"error": result.error}

        # Step 4: Synthesize
        message = _synthesize(user_input, intent, tool_outputs)

        latency_ms = round((time.time() - t0) * 1000, 1)
        tools_used = [f"{t}:{n}" for t, n in selected]

        # Update history
        self._history.append(RelayMessage("user", user_input, []))
        self._history.append(RelayMessage("assistant", message, tools_used))
        if len(self._history) > MAX_HISTORY * 2:
            self._history = self._history[-MAX_HISTORY * 2:]

        return RelayResult(
            message=message,
            intent=intent,
            tools_used=tools_used,
            tool_outputs=tool_outputs,
            model=intent_result.model_used,
            provider=provider,
            latency_ms=latency_ms,
        )

    def history(self) -> list[dict[str, Any]]:
        return [
            {"role": m.role, "content": m.content,
             "tools": m.tools_used, "ts": m.ts}
            for m in self._history
        ]

    def clear_history(self) -> None:
        self._history.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _run_code_tool(name: str, source: str, intent: dict[str, Any]) -> Any:
        """Synchronous code-analysis tools (run in thread pool via asyncio.to_thread)."""
        lang = intent.get("language", "unknown")
        if name == "metrics":
            return compute_metrics(source, lang)
        if name == "secrets":
            hits = scan_secrets(source)
            return [
                {"pattern": h.pattern_name, "line": h.line,
                 "snippet": h.snippet, "severity": h.severity}
                for h in hits
            ]
        if name == "lint":
            result = lint_python(source)
            return result.data if result.ok else {"error": result.error}
        if name == "ast":
            m: ASTMetrics = analyze_python_ast(source)
            return m.to_dict()
        return {"error": f"unknown code tool: {name}"}


# ── Response synthesis ────────────────────────────────────────────────────────

def _synthesize(
    user_input: str,
    intent: dict[str, Any],
    outputs: dict[str, Any],
) -> str:
    """Assemble a human-readable response from tool outputs."""
    parts: list[str] = []

    # Intent header
    primary = intent.get("primary_intent", "?")
    domain = intent.get("domain", "?")
    lang = intent.get("language", "?")
    complexity = intent.get("complexity", "?")
    goal = intent.get("stated_goal", "")
    parts.append(
        f"**{primary.upper()} · {domain} · {lang} · {complexity}**"
        + (f"\n_{goal}_" if goal else "")
    )

    # Issues spotted
    issues = intent.get("issues_spotted", [])
    if issues:
        parts.append("**Issues spotted:** " + " · ".join(f"`{i}`" for i in issues[:5]))

    # Code metrics
    if "metrics" in outputs:
        m = outputs["metrics"]
        sha = str(m.get("sha256", ""))[:8]
        parts.append(
            f"**Metrics:** {m.get('lines_code', '?')} lines · "
            f"{m.get('bytes', 0):,} bytes · "
            f"CC avg {m.get('cyclomatic_avg', '?')} · "
            f"SHA {sha}"
        )
        if m.get("todo_markers", 0):
            parts.append(f"  ⚠ {m['todo_markers']} TODO/FIXME marker(s)")

    # Secrets
    if "secrets" in outputs:
        hits = outputs["secrets"]
        if hits:
            crit = [h for h in hits if h.get("severity") == "critical"]
            parts.append(
                f"**🔴 Secrets ({len(hits)} hit{'s' if len(hits) != 1 else ''}):** "
                + "; ".join(f"`{h['pattern']}` line {h['line']}" for h in hits[:4])
            )
            if crit:
                parts.append(f"  ❗ {len(crit)} CRITICAL — rotate credentials immediately")
        else:
            parts.append("**Secrets:** ✓ clean")

    # Lint
    if "lint" in outputs:
        ln = outputs["lint"]
        if "error" in ln:
            parts.append(f"**Lint:** error — {ln['error']}")
        elif ln.get("issue_count", 0) == 0:
            parts.append("**Lint:** ✓ 0 issues")
        else:
            top = ln.get("issues", [])[:3]
            parts.append(
                f"**Lint:** {ln['issue_count']} issue(s) — "
                + " · ".join(f"`{i['code']}` line {i['line']}" for i in top)
            )

    # AST
    if "ast" in outputs:
        a = outputs["ast"]
        if not a.get("parse_error"):
            parts.append(
                f"**AST:** {len(a.get('functions', []))} fn · "
                f"{len(a.get('classes', []))} cls · "
                f"CC={a.get('cyclomatic_complexity', '?')} · "
                f"depth={a.get('max_depth', '?')}"
            )
        else:
            parts.append(f"**AST:** parse error — {a['parse_error']}")

    # LLM skill outputs
    _SKILL_LABELS = {
        "summarize":  "Summary",
        "critic":     "Critique",
        "fix":        "Fix",
        "security":   "Security",
        "test_gen":   "Tests",
        "arch_review":"Architecture",
        "doc":        "Docs",
    }
    for skill, label in _SKILL_LABELS.items():
        if skill not in outputs:
            continue
        out = outputs[skill]
        if isinstance(out, dict):
            if "error" in out:
                parts.append(f"**{label}:** ✗ {out['error'][:120]}")
            elif skill == "critic":
                dims = {k: v for k, v in out.items()
                        if isinstance(v, (int, float)) and k not in ("correctness",)}
                avg = round(sum(dims.values()) / len(dims), 1) if dims else "?"
                worst = out.get("lowest_dimension", "?")
                fix = out.get("critical_fix", "")
                parts.append(
                    f"**Critique:** avg={avg}/10 · worst={worst}"
                    + (f"\n  → _{fix}_" if fix else "")
                )
            elif skill == "fix":
                diag = out.get("diagnosis", "")
                conf = out.get("confidence", "?")
                parts.append(
                    f"**Fix ({conf}):** {diag}"
                    if diag else f"**Fix:** {str(out)[:200]}"
                )
            elif skill == "security":
                risk = out.get("risk_level", "?")
                vulns = out.get("vulnerabilities", [])
                emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡",
                         "low": "🟢", "none": "✓"}.get(risk, "?")
                parts.append(
                    f"**Security:** {emoji} {risk.upper()}"
                    + (f" · {len(vulns)} vulnerability(ies)" if vulns else "")
                )
            elif skill == "arch_review":
                rating = out.get("overall_rating", "?")
                fix_top = out.get("top_priority_fix", "")
                parts.append(
                    f"**Architecture:** {rating}/10"
                    + (f"\n  → _{fix_top}_" if fix_top else "")
                )
            else:
                parts.append(f"**{label}:** {str(out)[:300]}")
        elif isinstance(out, str):
            preview = out.strip()[:600]
            if len(out) > 600:
                preview += "…"
            parts.append(f"**{label}:**\n{preview}")

    # Next action recommendation
    next_action = intent.get("recommended_next", "")
    if next_action:
        parts.append(f"**Recommended next:** _{next_action}_")

    return "\n\n".join(parts)
