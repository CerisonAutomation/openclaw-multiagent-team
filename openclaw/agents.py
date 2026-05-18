"""Agent registry.

Schema mined from HORUS_OMNIFUSION_ONEFLOW_V8: every agent gets
{purpose, traits, skills, checks, metrics, on_fail}. The Orchestrator runs
agents in sequence per the routing table; on_fail actions tell it what to do
when a check trips.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from openclaw.audit import AuditLog
from openclaw.providers import Provider
from openclaw import prompts


@dataclass
class AgentSpec:
    name: str
    purpose: str
    traits: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    on_fail: list[str] = field(default_factory=list)


REGISTRY: dict[str, AgentSpec] = {
    "intent": AgentSpec(
        name="intent",
        purpose="Classify request: intent, complexity, domain, stated/unstated goals.",
        traits=["precise", "fast", "no-prose"],
        skills=["json-output", "intent-routing"],
        checks=["json_parseable", "has_stated_goal"],
        on_fail=["retry_once", "use_defaults"],
    ),
    "architect": AgentSpec(
        name="architect",
        purpose="Choose the minimal stack and propose files + commands.",
        traits=["pragmatic", "minimalist"],
        skills=["framework-selection", "scaffold-design"],
        checks=["framework_chosen", "files_listed", "commands_listed"],
        on_fail=["retry_with_constraints", "fall_back_to_static"],
    ),
    "coder": AgentSpec(
        name="coder",
        purpose="Write one file at a time, verbatim, no fences.",
        traits=["literal", "syntax-correct"],
        skills=["multi-language", "framework-idiom"],
        checks=["non_empty", "no_markdown_fences"],
        on_fail=["retry_strip_fences"],
    ),
    "tester": AgentSpec(
        name="tester",
        purpose="Produce smoke-test commands runnable without external services.",
        traits=["skeptical"],
        skills=["build-tooling", "static-checks"],
        checks=["json_parseable", "has_commands"],
        on_fail=["fall_back_to_syntax_check"],
    ),
    "fixer": AgentSpec(
        name="fixer",
        purpose="Diagnose a failed command, propose smallest patch.",
        traits=["surgical"],
        skills=["error-parsing", "minimal-diff"],
        checks=["has_diagnosis", "has_patch"],
        on_fail=["escalate_to_user"],
    ),
    "deployer": AgentSpec(
        name="deployer",
        purpose="Plan deploy commands; pull env vars; capture URL.",
        traits=["deterministic"],
        skills=["vercel", "preflight"],
        checks=["has_preflight", "has_deploy_commands"],
        on_fail=["skip_deploy_with_warning"],
    ),
    "critic": AgentSpec(
        name="critic",
        purpose="Score output across 10 dimensions; surface worst.",
        traits=["ruthless"],
        skills=["multi-dim-scoring"],
        checks=["scores_present", "all_numeric"],
        on_fail=["use_conservative_defaults"],
    ),
}


# ── Agent invocations ────────────────────────────────────────────────────────

def classify_intent(provider: Provider, task: str, audit: AuditLog | None = None) -> dict:
    if audit:
        audit.fire("intent")
    data = provider.json(prompts.INTENT_CLASSIFIER, task)
    if data.get("_parse_error"):
        data = {
            "primary_intent": "build",
            "complexity": "moderate",
            "domain": "fullstack",
            "stated_goal": task[:200],
            "unstated_goal": "ship something runnable",
            "input_rewrite": task[:200],
            "key_constraints": [],
            "needs_repo_context": False,
            "needs_deployment": False,
        }
    return data


def plan_architecture(provider: Provider, intent: dict, audit: AuditLog | None = None) -> dict:
    if audit:
        audit.fire("architect")
    user = json.dumps(intent, indent=2)
    return provider.json(prompts.ARCHITECT, user, max_tokens=1500)


def write_one_file(provider: Provider, file_spec: dict, architecture: dict, audit: AuditLog | None = None) -> str:
    if audit:
        audit.fire("coder")
    ctx = (
        f"Architecture context:\n{json.dumps(architecture, indent=2)}\n\n"
        f"File to write:\n  path: {file_spec['path']}\n  purpose: {file_spec.get('purpose', '')}"
    )
    raw = provider.chat(prompts.CODER, ctx, max_tokens=2500, temperature=0.3)
    return _strip_fences(raw)


def plan_verification(provider: Provider, architecture: dict, audit: AuditLog | None = None) -> dict:
    if audit:
        audit.fire("tester")
    return provider.json(prompts.TESTER, json.dumps(architecture, indent=2))


def diagnose_failure(provider: Provider, cmd: str, output: str, audit: AuditLog | None = None) -> dict:
    if audit:
        audit.fire("fixer")
    user = f"Failed command:\n  {cmd}\n\nOutput:\n{output[-2000:]}"
    return provider.json(prompts.FIXER, user, max_tokens=2500)


def plan_deploy(provider: Provider, architecture: dict, audit: AuditLog | None = None) -> dict:
    if audit:
        audit.fire("deployer")
    return provider.json(prompts.DEPLOYER, json.dumps(architecture, indent=2))


def critique(provider: Provider, task: str, output: str, audit: AuditLog | None = None) -> dict:
    if audit:
        audit.fire("critic")
    user = f"ORIGINAL TASK:\n{task}\n\nARTIFACT TO EVALUATE:\n{output[:6000]}"
    scores = provider.json(prompts.CRITIC, user)
    if scores.get("_parse_error"):
        scores = {
            "clarity": 6.5, "structure": 6.5, "completeness": 6.0, "accuracy": 6.5,
            "applicability": 6.0, "expertise": 6.0, "originality": 6.0, "compliance": 7.0,
            "efficiency": 6.5, "resonance": 6.0,
            "lowest_dimension": "completeness",
            "critical_fix": "Critic returned unparseable output — defaulting to conservative scores.",
        }
    return scores


# ── Routing (per Omega_Intent_Mesh traits_cascade) ───────────────────────────

ROUTING: dict[str, dict[str, list[str]]] = {
    "build":   {"primary": ["intent", "architect", "coder", "tester"], "secondary": ["deployer"]},
    "fix":     {"primary": ["intent", "fixer", "tester"],              "secondary": ["critic"]},
    "extend":  {"primary": ["intent", "architect", "coder", "tester"], "secondary": ["deployer"]},
    "deploy":  {"primary": ["intent", "deployer"],                      "secondary": []},
    "review":  {"primary": ["intent", "critic"],                        "secondary": []},
    "explain": {"primary": ["intent", "critic"],                        "secondary": []},
}


def route(intent: dict) -> list[str]:
    primary = intent.get("primary_intent", "build")
    table = ROUTING.get(primary, ROUTING["build"])
    return list(table["primary"]) + list(table["secondary"])


# ── Helpers ──────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first = text.find("\n")
        text = text[first + 1:] if first != -1 else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip() + "\n"
