"""Brain — single orchestrator that routes every user request to the right system.

Routing logic:
  build / deploy / extend  →  openclaw 7-phase orchestrator
                               (uses Jan/Ollama if local available, else cloud)
  review / fix / analyze   →  inspector relay (always local: Jan → Ollama → none)
  everything else          →  relay (inspect, explain, critique)

One Brain instance per session. Persistent history in .openclaw_history.json.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── History ───────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    role: str
    content: str
    intent: str = ""
    provider: str = ""
    tools: list[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)


# ── Status payload ────────────────────────────────────────────────────────────

@dataclass
class BrainResult:
    message: str
    intent: dict[str, Any]
    tools_used: list[str]
    model: str
    provider: str
    latency_ms: float
    was_polished: bool = False
    polished_prompt: str = ""
    route: str = "relay"        # "relay" | "orchestrator"
    error: str = ""


# ── Brain ─────────────────────────────────────────────────────────────────────

# Intents that route to the full 7-phase orchestrator (build, not just advice)
_ORCHESTRATOR_INTENTS = {"build", "deploy", "extend"}

HISTORY_FILE = Path(".openclaw_history.json")
_MAX_HISTORY = 40


class Brain:
    """
    One instance per session.  Call `handle()` per user turn.
    The relay and orchestrator are lazy-loaded on first use.
    """

    def __init__(self) -> None:
        self._relay: Any = None          # RelayAgent (lazy)
        self._skill_index: Any = None    # SkillIndex (lazy)
        self._history: list[Turn] = _load_history()

    # ── Public API ────────────────────────────────────────────────────────────

    async def handle(
        self,
        user_input: str,
        source: str = "",
        on_status: Callable[[str], None] | None = None,
    ) -> BrainResult:
        t0 = time.time()
        status = on_status or (lambda _: None)

        # 1. Polish vague prompts (< 10 words, not a command)
        polished, was_polished = await self._polish(user_input, status)

        # 2. Classify intent (always local)
        status("intent…")
        intent = await self._classify(polished, source)
        primary = intent.get("primary_intent", "unknown")

        # 3. Route
        if primary in _ORCHESTRATOR_INTENTS and not source.strip():
            status("orchestrator…")
            result = await self._run_orchestrator(polished, primary, status)
        else:
            status("relay…")
            result = await self._run_relay(polished, source)

        result.was_polished = was_polished
        result.polished_prompt = polished if was_polished else ""
        result.latency_ms = round((time.time() - t0) * 1000, 1)

        # 4. Persist history
        self._history.append(Turn(role="user", content=user_input, intent=primary))
        self._history.append(Turn(
            role="assistant", content=result.message[:400],
            intent=primary, provider=result.provider, tools=result.tools_used,
        ))
        _save_history(self._history)

        return result

    async def status_check(self) -> dict[str, Any]:
        """Health-check all providers."""
        si = self._get_skill_index()
        ollama_st = await si.ollama.status()
        jan_st    = await si.jan.status()
        return {
            "ollama": {"ok": ollama_st.available, "models": ollama_st.models,
                       "error": ollama_st.error},
            "jan":    {"ok": jan_st.available,    "models": jan_st.models,
                       "error": jan_st.error},
            "active": "ollama" if ollama_st.available else ("jan" if jan_st.available else "none"),
        }

    def history(self) -> list[dict[str, Any]]:
        return [asdict(t) for t in self._history]

    def clear_history(self) -> None:
        self._history.clear()
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_skill_index(self) -> Any:
        if self._skill_index is None:
            from tools.inspector.agents import skill_index
            self._skill_index = skill_index
        return self._skill_index

    def _get_relay(self) -> Any:
        if self._relay is None:
            from tools.inspector.relay import RelayAgent
            self._relay = RelayAgent(self._get_skill_index())
        return self._relay

    async def _polish(
        self, user_input: str, status: Callable[[str], None]
    ) -> tuple[str, bool]:
        """Apply trait system to vague prompts. Returns (polished, was_changed)."""
        words = user_input.split()
        if len(words) >= 10 or user_input.startswith(":"):
            return user_input, False
        status("polishing…")
        try:
            si = self._get_skill_index()
            r = await si.run("prompt_polish", user_input)
            if r.ok and isinstance(r.output, dict) and r.output.get("was_vague"):
                improved = r.output.get("improved_prompt", "").strip()
                if improved and improved != user_input:
                    return improved, True
        except Exception:
            pass
        return user_input, False

    async def _classify(self, text: str, source: str) -> dict[str, Any]:
        """Run intent classification via local LLM."""
        try:
            si = self._get_skill_index()
            combined = f"{text}\n\n---\n{source}" if source else text
            r = await si.run("intent", combined)
            if r.ok and isinstance(r.output, dict):
                return r.output
        except Exception:
            pass
        return {"primary_intent": "unknown"}

    async def _run_relay(self, user_input: str, source: str) -> BrainResult:
        """Route to inspector relay for code analysis + LLM skills."""
        try:
            relay = self._get_relay()
            rr = await relay.handle(user_input, source=source)
            return BrainResult(
                message=rr.message,
                intent=rr.intent,
                tools_used=rr.tools_used,
                model=rr.model,
                provider=rr.provider,
                latency_ms=rr.latency_ms,
                route="relay",
            )
        except Exception as exc:
            return BrainResult(
                message=f"Relay error: {exc}",
                intent={}, tools_used=[], model="", provider="none",
                latency_ms=0, route="relay", error=str(exc),
            )

    async def _run_orchestrator(
        self, task: str, intent: str, status: Callable[[str], None]
    ) -> BrainResult:
        """Route to the 7-phase SINGULARITY orchestrator.

        Uses Jan or Ollama as the LLM provider if available (no cloud key needed).
        Falls back to whatever cloud provider is configured.
        """
        from openclaw.orchestrator import Orchestrator, RunConfig
        from openclaw.providers import get_provider
        from openclaw.models import PRESETS

        # Prefer local provider so builds work without cloud keys
        provider_name, local_model = await self._pick_local_provider()
        status(f"building via {provider_name}…")

        def _sync_run() -> Any:
            if local_model:
                # Construct a local OpenAI-compatible provider pointing at Jan/Ollama
                from openclaw.providers import OpenAIProvider
                preset = PRESETS.get(provider_name)
                base_url = (preset.base_url if preset else None) or "http://localhost:11434/v1"
                provider = OpenAIProvider(
                    name=provider_name, api_key=provider_name,
                    base_url=base_url, model=local_model,
                )
            else:
                provider = get_provider()  # cloud fallback

            cfg = RunConfig(
                workspace="./openclaw_workspace",
                max_critique_loops=2,
                quality_threshold=7.0,
                deploy=(intent == "deploy"),
                verbose=False,
            )
            return Orchestrator(provider=provider, config=cfg).run(task)

        try:
            result = await asyncio.to_thread(_sync_run)
            msg = result.final_output.strip()
            if result.seal:
                msg += f"\n\n{result.seal}"
            if result.deploy_url:
                msg += f"\n\nDeploy URL: {result.deploy_url}"
            return BrainResult(
                message=msg,
                intent={"primary_intent": intent},
                tools_used=["orchestrator:intent", "orchestrator:architect",
                            "orchestrator:coder", "orchestrator:critic"],
                model=local_model or "cloud",
                provider=provider_name,
                latency_ms=0,
                route="orchestrator",
            )
        except Exception as exc:
            return BrainResult(
                message=f"Build failed: {exc}\n\nTip: check provider config or add source code for local analysis.",
                intent={"primary_intent": intent},
                tools_used=[], model="", provider="none",
                latency_ms=0, route="orchestrator", error=str(exc),
            )

    async def _pick_local_provider(self) -> tuple[str, str]:
        """Return (provider_name, model_id) for the best available local LLM."""
        si = self._get_skill_index()

        # Try Jan first (user's active local server)
        jan_st = await si.jan.status()
        if jan_st.available and jan_st.models:
            model = (os.environ.get("JAN_MODEL") or jan_st.models[0]).strip()
            return "jan", model

        # Try Ollama
        ollama_st = await si.ollama.status()
        if ollama_st.available and ollama_st.models:
            return "ollama", ollama_st.models[0]

        # No local LLM → signal caller to use cloud
        return "cloud", ""


# ── History persistence ───────────────────────────────────────────────────────

def _load_history() -> list[Turn]:
    try:
        if HISTORY_FILE.exists():
            raw = json.loads(HISTORY_FILE.read_text())
            return [Turn(**t) for t in raw[-_MAX_HISTORY:]]
    except Exception:
        pass
    return []


def _save_history(history: list[Turn]) -> None:
    try:
        HISTORY_FILE.write_text(
            json.dumps([asdict(t) for t in history[-_MAX_HISTORY:]],
                       indent=2, default=str)
        )
    except Exception:
        pass
