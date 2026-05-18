"""Orchestrator — runs the 7-phase SINGULARITY pipeline with real agents.

    PHASE 1  FREEZE      → intent classifier (agents.classify_intent)
    PHASE 2  DOMAIN      → architect picks framework + files
    PHASE 3  ARCHITECT   → preflight gates, pre-execution
    PHASE 4  EXECUTE     → coder writes files, tester validates, fixer patches
    PHASE 5  CRITIQUE    → critic scores 10 dimensions, loops up to N times
    PHASE 6  REALITY     → mid + post gates verify deployable
    PHASE 7  SEAL        → 4-block output contract + audit log

The orchestrator is callable two ways:
    * `run("prompt text")`              → build from a prompt
    * `run("…", repo_path="./app")`     → analyze/fix an existing repo
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openclaw import agents, gates
from openclaw.audit import AuditLog
from openclaw.providers import Provider, get_provider
from openclaw.tools import (
    AgentLimits,
    analyze_repo,
    ensure_workspace,
    generate_image,
    read_file,
    run_shell,
    vercel_cli_available,
    vercel_deploy,
    write_file,
)


@dataclass
class RunConfig:
    workspace: str = "./openclaw_workspace"
    max_critique_loops: int = 3
    quality_threshold: float = 7.5
    deploy: bool = False
    deploy_prod: bool = False
    image_assets: list[str] = field(default_factory=list)   # prompts → assets/<idx>.svg
    timeout_per_command_s: int = 180
    verbose: bool = True


@dataclass
class RunResult:
    ok: bool
    seal: str
    audit: AuditLog
    workspace: str
    deploy_url: str | None = None
    final_output: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "seal": self.seal,
            "workspace": self.workspace,
            "deploy_url": self.deploy_url,
            "audit": self.audit.to_dict(),
        }


class Orchestrator:
    def __init__(self, provider: Provider | None = None, config: RunConfig | None = None) -> None:
        self.provider = provider or get_provider()
        self.config = config or RunConfig()
        self.limits = AgentLimits(max_execution_time_s=self.config.timeout_per_command_s)
        self.audit = AuditLog()

    # ── public entry ────────────────────────────────────────────────────────

    def run(self, task: str, repo_path: str | None = None) -> RunResult:
        t0 = time.time()
        self.audit.task = task
        workspace = ensure_workspace(self.config.workspace)
        self._log(f"workspace: {workspace}")
        self._log(f"provider:  {self.provider.name} ({self.provider.model})")

        # ── Optional repo context ───────────────────────────────────────────
        repo_summary: dict | None = None
        if repo_path:
            self._log(f"analyzing repo: {repo_path}")
            repo_summary = analyze_repo(repo_path)
            self.audit.add_phase("repo_analysis", repo_summary)

        # ── PHASE 1 · FREEZE ────────────────────────────────────────────────
        self._log("PHASE 1 · FREEZE & CLASSIFY")
        full_prompt = task
        if repo_summary:
            full_prompt = (
                f"REPOSITORY CONTEXT:\n{json.dumps(repo_summary, indent=2)}\n\nUSER REQUEST:\n{task}"
            )
        intent = agents.classify_intent(self.provider, full_prompt, self.audit)
        self.audit.intent = intent
        self.audit.input_rewrite = intent.get("input_rewrite", task[:200])

        pre = gates.gate_intent_clarity(intent)
        self.audit.add_phase("gate_intent_clarity", {"passed": pre.passed, "reason": pre.reason})
        if not pre.passed:
            return self._finalize(t0, False, f"intent gate failed: {pre.reason}", workspace, None, "")

        # ── PHASE 2 · DOMAIN SWEEP / ARCHITECTURE ───────────────────────────
        self._log(f"PHASE 2 · ARCHITECT  (intent: {intent.get('primary_intent')})")
        routed_agents = agents.route(intent)
        self._log(f"routing: {' → '.join(routed_agents)}")

        architecture: dict = {}
        if "architect" in routed_agents:
            architecture = agents.plan_architecture(self.provider, intent, self.audit)
            self.audit.architecture = architecture
            arch_gate = gates.gate_architecture_plausible(architecture)
            self.audit.add_phase("gate_architecture", {"passed": arch_gate.passed, "reason": arch_gate.reason})
            if not arch_gate.passed:
                return self._finalize(t0, False, f"architecture gate failed: {arch_gate.reason}",
                                      workspace, None, "")

        # ── PHASE 3 → 4 · EXECUTE  (coder + tester + fixer) ─────────────────
        exit_codes: list[int] = []
        written: list[str] = []
        if "coder" in routed_agents:
            self._log("PHASE 4 · EXECUTE  (writing files)")
            for fs in (architecture.get("files_to_create") or []):
                contents = agents.write_one_file(self.provider, fs, architecture, self.audit)
                res = write_file(fs["path"], contents, workspace)
                self.audit.add_tool_call("write_file", {"path": fs["path"]}, res.ok, res.output)
                written.append(fs["path"])
                self._log(f"  wrote {fs['path']}  ({len(contents)} bytes)")

        # Optional image generation
        for i, img_prompt in enumerate(self.config.image_assets):
            out = Path(workspace) / "assets" / f"asset_{i}.png"
            img = generate_image(img_prompt, out)
            self.audit.add_tool_call("generate_image", {"prompt": img_prompt}, img.ok, img.output)
            self._log(f"  image: {img.output}")

        if "tester" in routed_agents and architecture.get("commands"):
            self._log("PHASE 4 · TEST   (running verification commands)")
            self.audit.fire("tester")
            for cmd in architecture.get("commands", []):
                res = run_shell(cmd, cwd=workspace, timeout=self.config.timeout_per_command_s, limits=self.limits)
                exit_codes.append(res.exit_code)
                self.audit.add_tool_call("run_shell", {"cmd": cmd}, res.ok,
                                         res.output[-400:] + (res.error[-400:] if res.error else ""))
                self._log(f"  $ {cmd}  → exit {res.exit_code}")

                # On failure: fixer agent
                if not res.ok and "fixer" in agents.REGISTRY:
                    diag = agents.diagnose_failure(self.provider, cmd, res.output + "\n" + res.error, self.audit)
                    self.audit.add_phase("fixer_diagnosis", diag)
                    self._apply_fix(diag, workspace)

        # Mid gates
        if written:
            files_gate = gates.gate_files_written(Path(workspace), written)
            self.audit.add_phase("gate_files_written",
                                 {"passed": files_gate.passed, "reason": files_gate.reason})
        if exit_codes:
            cmd_gate = gates.gate_commands_succeeded(exit_codes)
            self.audit.add_phase("gate_commands", {"passed": cmd_gate.passed, "reason": cmd_gate.reason})

        # ── PHASE 5 · CRITIQUE LOOP (SAFLA adaptive weighting) ──────────────
        final_output = self._compose_summary(written, exit_codes, architecture)
        _persistent_weak: str | None = None  # SAFLA: track dimension that fails repeatedly
        for i in range(1, self.config.max_critique_loops + 1):
            self._log(f"PHASE 5 · CRITIQUE  loop {i}/{self.config.max_critique_loops}")
            # Inject adaptive focus hint after first loop if a weak dimension persists
            critique_input = final_output
            if _persistent_weak and i > 1:
                critique_input += (
                    f"\n\n[ADAPTIVE FOCUS: '{_persistent_weak}' has scored below threshold "
                    f"across {i-1} loop(s) — weight this dimension heavily in your assessment]"
                )
            scores = agents.critique(self.provider, task, critique_input, self.audit)
            numeric = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
            avg = sum(numeric.values()) / len(numeric) if numeric else 0.0
            worst_dim = scores.get("lowest_dimension", "?")
            worst_val = numeric.get(worst_dim, min(numeric.values()) if numeric else 0.0)
            self.audit.critique_history.append({
                "iteration": i, "scores": numeric, "avg": round(avg, 2),
                "worst": worst_dim, "worst_score": worst_val,
                "critical_fix": scores.get("critical_fix", ""),
            })
            qgate = gates.gate_quality_threshold(numeric, self.config.quality_threshold)
            self._log(f"  avg={avg:.1f}  worst={worst_dim}({worst_val:.1f})  → {qgate.reason}")
            # SAFLA: update persistent weak tracker
            _persistent_weak = worst_dim if worst_val < self.config.quality_threshold else None
            if qgate.passed or i == self.config.max_critique_loops:
                self.audit.final_scores = numeric
                self.audit.final_avg = round(avg, 2)
                break

        # ── PHASE 6 · REALITY GATES ─────────────────────────────────────────
        ph_gate = gates.gate_no_placeholders(final_output)
        self.audit.add_phase("gate_no_placeholders", {"passed": ph_gate.passed, "reason": ph_gate.reason})
        self.audit.validation["structure_locked"] = ph_gate.passed
        self.audit.validation["domain_aligned"] = bool(architecture.get("framework"))

        # ── PHASE 6.5 · DEPLOY (optional) ───────────────────────────────────
        deploy_url: str | None = None
        if self.config.deploy and "deployer" in routed_agents:
            self._log("PHASE 6.5 · DEPLOY")
            if not vercel_cli_available():
                self._log("  vercel CLI not installed — skipping deploy")
                self.audit.add_phase("deploy_skipped", {"reason": "vercel CLI not installed"})
            else:
                dep = vercel_deploy(workspace, prod=self.config.deploy_prod)
                self.audit.add_tool_call("vercel_deploy", {"prod": self.config.deploy_prod},
                                         dep.ok, (dep.output + dep.error)[-800:])
                if dep.ok and dep.extra.get("deploy_url"):
                    deploy_url = dep.extra["deploy_url"]
                    self.audit.validation["real_world_deployable"] = True
                    self._log(f"  deployed: {deploy_url}")
                else:
                    self._log(f"  deploy failed: {dep.error[:200]}")

        # ── PHASE 7 · SEAL ──────────────────────────────────────────────────
        ok = bool(written) or bool(repo_summary) or self.config.image_assets or deploy_url
        seal_line = self._build_seal(ok, written, deploy_url, repo_summary)
        self.audit.seal = seal_line
        self.audit.primary_output = final_output
        return self._finalize(t0, ok, seal_line, workspace, deploy_url, final_output)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _apply_fix(self, diag: dict, workspace: Path) -> None:
        kind = diag.get("fix_kind", "")
        target = diag.get("target_path", "")
        patch = diag.get("patch", "")
        if kind in ("rewrite_file", "new_file") and target and patch:
            res = write_file(target, patch, workspace)
            self.audit.add_tool_call("fixer.write_file", {"path": target, "kind": kind}, res.ok, res.output)
        elif kind == "install_dep" and patch:
            res = run_shell(f"npm install {patch}", cwd=workspace, limits=self.limits)
            self.audit.add_tool_call("fixer.install", {"pkg": patch}, res.ok, res.output[-300:])
        elif kind == "change_command" and patch:
            res = run_shell(patch, cwd=workspace, limits=self.limits)
            self.audit.add_tool_call("fixer.cmd", {"cmd": patch}, res.ok, res.output[-300:])
        # else: nothing safe to do automatically

    def _compose_summary(self, written: list[str], exit_codes: list[int], arch: dict) -> str:
        passes = sum(1 for c in exit_codes if c == 0)
        fails = len(exit_codes) - passes
        framework = arch.get("framework", "—")
        deploy_target = arch.get("deploy_target", "—")
        bullet_files = "\n".join(f"  - {p}" for p in written) or "  (none)"
        bullet_cmds = "\n".join(f"  - {c}" for c in (arch.get("commands") or [])) or "  (none)"
        return (
            f"Framework: {framework}\n"
            f"Deploy target: {deploy_target}\n\n"
            f"Files written ({len(written)}):\n{bullet_files}\n\n"
            f"Commands run: {passes} passed, {fails} failed\n{bullet_cmds}\n"
        )

    def _build_seal(self, ok: bool, written: list[str], deploy_url: str | None, repo: dict | None) -> str:
        if deploy_url:
            return f"SEAL: deployed {len(written)} files → live at {deploy_url}"
        if written:
            return f"SEAL: built {len(written)} files in workspace → next: deploy or extend"
        if repo:
            return f"SEAL: analyzed repo at {repo.get('root')} → next: extend or fix"
        if not ok:
            return "SEAL: pipeline halted by a gate → check audit log"
        return "SEAL: completed → check audit log for details"

    def _finalize(self, t0: float, ok: bool, seal: str, workspace: Path,
                  deploy_url: str | None, final_output: str) -> RunResult:
        self.audit.elapsed_seconds = round(time.time() - t0, 2)
        self.audit.seal = seal
        return RunResult(
            ok=ok, seal=seal, audit=self.audit, workspace=str(workspace),
            deploy_url=deploy_url, final_output=final_output,
        )

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            import sys
            print(f"\033[90m· {msg}\033[0m", file=sys.stderr, flush=True)


# ── module-level convenience ─────────────────────────────────────────────────

def run(task: str, repo_path: str | None = None, **config_kwargs: Any) -> RunResult:
    cfg = RunConfig(**config_kwargs)
    orch = Orchestrator(config=cfg)
    return orch.run(task, repo_path=repo_path)
