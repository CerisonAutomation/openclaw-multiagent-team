"""HORUS-style audit log.

Schema adopted from HORUS_XTRILLINITY_GODSUMMIT_REWRITE_FINAL:
  - input_rewrite: cleaned restatement of what the user asked
  - primary_output: the chosen final result
  - optional_alt: alternative if primary failed a gate (optional)
  - audit_results: per-dimension validation booleans + scores
  - execution_trace: stages completed, scenario variants tested
  - active_traits: which agents actually fired (not the registry)
  - validation: real_world_deployable, domain_aligned, structure_locked

Writes one JSON file per session and exposes a tail-appendable in-memory dict.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AuditLog:
    session_id: str = field(default_factory=_now)
    task: str = ""
    input_rewrite: str = ""
    intent: dict = field(default_factory=dict)
    architecture: dict = field(default_factory=dict)
    phases: list[dict] = field(default_factory=list)        # one entry per phase
    active_agents: list[str] = field(default_factory=list)  # agents that actually fired
    tool_calls: list[dict] = field(default_factory=list)    # every shell/fs/vercel call
    critique_history: list[dict] = field(default_factory=list)
    final_scores: dict = field(default_factory=dict)
    final_avg: float = 0.0
    elapsed_seconds: float = 0.0
    primary_output: str = ""
    optional_alt: str | None = None
    validation: dict = field(default_factory=lambda: {
        "real_world_deployable": False,
        "domain_aligned": False,
        "structure_locked": False,
    })
    seal: str = ""

    def add_phase(self, phase: str, result: Any) -> None:
        self.phases.append({"phase": phase, "at": _now(), "result": result})

    def add_tool_call(self, tool: str, args: dict, ok: bool, summary: str = "") -> None:
        self.tool_calls.append({"tool": tool, "args": args, "ok": ok, "summary": summary, "at": _now()})

    def fire(self, agent: str) -> None:
        if agent not in self.active_agents:
            self.active_agents.append(agent)

    def write(self, path: str | Path = "openclaw_audit.json") -> Path:
        p = Path(path)
        p.write_text(json.dumps(asdict(self), indent=2, default=str))
        return p

    def to_dict(self) -> dict:
        return asdict(self)
