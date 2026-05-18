"""Three-stage gates (pre / mid / post) from Omega_Intent_Mesh.

Each gate is a pure check that returns (passed: bool, reason: str). The
orchestrator chooses an on_failure action based on the gate name. No theatrical
'quantum_seal' nonsense — these are real assertions.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GateResult:
    passed: bool
    reason: str
    score: float = 0.0


# ── PRE-EXECUTION ────────────────────────────────────────────────────────────

def gate_intent_clarity(intent: dict, min_required_keys: Iterable[str] = ("primary_intent", "stated_goal")) -> GateResult:
    """Reject if intent classifier failed or returned garbage."""
    if intent.get("_parse_error"):
        return GateResult(False, "intent classifier returned unparseable JSON", 0.0)
    missing = [k for k in min_required_keys if not intent.get(k)]
    if missing:
        return GateResult(False, f"intent missing required keys: {missing}", 0.3)
    if intent.get("stated_goal") and len(str(intent["stated_goal"])) < 4:
        return GateResult(False, "stated_goal too short — request is ambiguous", 0.5)
    return GateResult(True, "intent clear", 1.0)


def gate_architecture_plausible(arch: dict) -> GateResult:
    if arch.get("_parse_error"):
        return GateResult(False, "architect returned unparseable JSON", 0.0)
    if not arch.get("framework"):
        return GateResult(False, "architect did not choose a framework", 0.4)
    files = arch.get("files_to_create") or []
    if not files:
        return GateResult(False, "no files_to_create proposed", 0.3)
    if not arch.get("commands"):
        return GateResult(False, "no build/test commands proposed", 0.4)
    return GateResult(True, f"architecture ok ({len(files)} files)", 1.0)


# ── MID-EXECUTION ────────────────────────────────────────────────────────────

def gate_files_written(workspace: Path, expected: Iterable[str]) -> GateResult:
    missing = [p for p in expected if not (workspace / p).exists()]
    if missing:
        return GateResult(False, f"expected files missing: {missing}", 0.0)
    return GateResult(True, "all files present", 1.0)


def gate_commands_succeeded(exit_codes: Iterable[int]) -> GateResult:
    codes = list(exit_codes)
    if not codes:
        return GateResult(True, "no commands to verify", 1.0)
    failed = sum(1 for c in codes if c != 0)
    if failed:
        return GateResult(False, f"{failed}/{len(codes)} commands failed", 0.0)
    return GateResult(True, f"{len(codes)} commands ok", 1.0)


# ── POST-EXECUTION ───────────────────────────────────────────────────────────

def gate_quality_threshold(scores: dict, threshold: float = 7.5) -> GateResult:
    if not scores:
        return GateResult(False, "no scores produced", 0.0)
    numeric = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
    if not numeric:
        return GateResult(False, "no numeric scores", 0.0)
    worst_key = min(numeric, key=numeric.get)
    worst = numeric[worst_key]
    avg = sum(numeric.values()) / len(numeric)
    if worst >= threshold:
        return GateResult(True, f"all dims >= {threshold} (avg {avg:.1f})", avg / 10.0)
    return GateResult(False, f"{worst_key}={worst:.1f} below {threshold}", avg / 10.0)


_DRIFT_RE = re.compile(r"\b(TODO|FIXME|placeholder|lorem ipsum|tbd)\b", re.IGNORECASE)


def gate_no_placeholders(output: str) -> GateResult:
    matches = _DRIFT_RE.findall(output)
    if matches:
        return GateResult(False, f"placeholders found: {sorted(set(matches))}", 0.0)
    return GateResult(True, "no placeholders", 1.0)


def gate_deploy_succeeded(stdout: str, url_hint: str = r"https?://[^\s]+\.vercel\.app[^\s]*") -> GateResult:
    m = re.search(url_hint, stdout)
    if not m:
        return GateResult(False, "no deploy URL detected in output", 0.0)
    return GateResult(True, f"deploy URL: {m.group(0)}", 1.0)
