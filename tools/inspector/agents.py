"""Skill index, Ollama client, and agent runners."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import OLLAMA_MODEL_PRESETS, ROLE_MODEL_PREFERENCE, SKILL_INDEX, SkillDef, get_config

logger = logging.getLogger(__name__)


# ── Ollama client ─────────────────────────────────────────────────────────────

@dataclass
class OllamaStatus:
    available: bool
    models: list[str] = field(default_factory=list)
    error: str = ""


class OllamaClient:
    def __init__(self) -> None:
        self._cfg = get_config()

    @property
    def base(self) -> str:
        return self._cfg.ollama_base

    @property
    def timeout(self) -> int:
        return self._cfg.ollama_timeout

    async def status(self) -> OllamaStatus:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{self.base}/api/tags")
                if r.status_code != 200:
                    return OllamaStatus(available=False, error=f"HTTP {r.status_code}")
                data = r.json()
                models = [m["name"].split(":")[0] for m in data.get("models", [])]
                return OllamaStatus(available=True, models=models)
        except Exception as e:
            return OllamaStatus(available=False, error=str(e))

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{self.base}/api/tags")
                if r.status_code == 200:
                    raw = r.json().get("models", [])
                    return [
                        {
                            "name": m["name"],
                            "short": m["name"].split(":")[0],
                            "size_gb": round(m.get("size", 0) / 1e9, 2),
                            "modified": m.get("modified_at", ""),
                            "preset": OLLAMA_MODEL_PRESETS.get(m["name"].split(":")[0]),
                        }
                        for m in raw
                    ]
        except Exception:
            pass
        return []

    def _pick_model(self, role: str) -> str:
        """Pick best available model for role; falls back to config default."""
        return get_config().default_model  # sync fallback (use pick_model_async for real selection)

    async def pick_model_async(self, role: str) -> str:
        """Pick best Ollama model for role; auto-detects from running instance."""
        available = {m["short"] for m in await self.list_models()}
        for candidate in ROLE_MODEL_PREFERENCE.get(role, []):
            if candidate in available:
                return candidate
        if available:
            # Fall back to largest available model (heuristic: prefer longer names like "llama3.1" over "phi3")
            return sorted(available, key=len, reverse=True)[0]
        return self._cfg.default_model

    async def pick_jan_model(self) -> str | None:
        """Return the first available Jan model, or None if Jan is not running."""
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get("http://localhost:1337/v1/models")
                if r.status_code == 200:
                    data = r.json().get("data", [])
                    return data[0]["id"] if data else None
        except Exception:
            pass
        return None

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        role: str = "intent",
    ) -> str:
        if model is None:
            model = await self.pick_model_async(role)
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(f"{self.base}/api/generate", json=payload)
            r.raise_for_status()
            return r.json().get("response", "")

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        role: str = "intent",
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        if model is None:
            model = await self.pick_model_async(role)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(f"{self.base}/api/chat", json=payload)
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "")


# ── JSON extractor ────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict[str, Any] | list[Any]:
    """Find and parse the first JSON object or array in a string."""
    # Try the whole string first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown fences
    cleaned = re.sub(r"```json?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Find first balanced { ... } or [ ... ]
    for start_ch, end_ch in (("{", "}"), ("[", "]")):
        start = text.find(start_ch)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == start_ch:
                depth += 1
            elif ch == end_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start: i + 1])
                    except json.JSONDecodeError:
                        break
    return {"_parse_error": True, "raw": text[:500]}


# ── Skill index ───────────────────────────────────────────────────────────────

@dataclass
class SkillResult:
    skill: str
    model_used: str
    output: Any          # parsed JSON or str depending on skill output_format
    raw: str             # raw LLM output
    ok: bool = True
    error: str = ""
    provider: str = "ollama"  # "ollama" | "jan" | "none"


class SkillIndex:
    """Registry that maps skill names → SkillDef and runs them via Ollama."""

    def __init__(self) -> None:
        self._ollama = OllamaClient()
        self._extra: dict[str, SkillDef] = {}

    def register(self, skill: SkillDef) -> None:
        self._extra[skill.name] = skill

    def list(self) -> list[dict[str, Any]]:
        all_skills = {**SKILL_INDEX, **self._extra}
        return [
            {
                "name": s.name,
                "description": s.description,
                "role": s.role,
                "output_format": s.output_format,
            }
            for s in all_skills.values()
        ]

    def get(self, name: str) -> SkillDef | None:
        return SKILL_INDEX.get(name) or self._extra.get(name)

    async def run(
        self,
        skill_name: str,
        input_text: str,
        model: str | None = None,
        extra_context: str = "",
    ) -> SkillResult:
        skill = self.get(skill_name)
        if skill is None:
            return SkillResult(skill=skill_name, model_used="", output=None, raw="",
                               ok=False, error=f"skill not found: {skill_name!r}")

        chosen_model = model or await self._ollama.pick_model_async(skill.role)
        prompt = input_text
        if extra_context:
            prompt = f"{extra_context}\n\n---\n\n{input_text}"

        try:
            raw = await self._ollama.generate(
                prompt=prompt,
                model=chosen_model,
                system=skill.system_prompt,
                temperature=skill.temperature,
                max_tokens=skill.max_tokens,
                role=skill.role,
            )
        except Exception as e:
            return SkillResult(skill=skill_name, model_used=chosen_model, output=None,
                               raw="", ok=False, error=str(e))

        if skill.output_format == "json":
            parsed = _extract_json(raw)
            return SkillResult(skill=skill_name, model_used=chosen_model,
                               output=parsed, raw=raw, ok="_parse_error" not in parsed)
        return SkillResult(skill=skill_name, model_used=chosen_model,
                           output=raw.strip(), raw=raw, ok=True)

    async def run_pipeline(
        self,
        input_text: str,
        skill_names: list[str] | None = None,
        model: str | None = None,
    ) -> dict[str, SkillResult]:
        names = skill_names or list(SKILL_INDEX.keys())
        results: dict[str, SkillResult] = {}
        for name in names:
            results[name] = await self.run(name, input_text, model=model)
        return results

    @property
    def ollama(self) -> OllamaClient:
        return self._ollama


# Module-level singleton
skill_index = SkillIndex()
