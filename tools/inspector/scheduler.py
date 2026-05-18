"""Asyncio-native scheduler: heartbeat monitor, cron tasks, and event bus."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from .agents import OllamaClient

logger = logging.getLogger(__name__)


# ── Event bus ─────────────────────────────────────────────────────────────────

@dataclass
class Event:
    kind: str                            # "heartbeat" | "task_done" | "task_error" | "ollama_up" | "ollama_down"
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


Listener = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    def __init__(self) -> None:
        self._listeners: list[Listener] = []

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def unsubscribe(self, listener: Listener) -> None:
        self._listeners = [l for l in self._listeners if l is not listener]

    async def emit(self, event: Event) -> None:
        for listener in list(self._listeners):
            try:
                await listener(event)
            except Exception as e:
                logger.warning("event bus listener error: %s", e)


bus = EventBus()          # module-level singleton


# ── Heartbeat monitor ─────────────────────────────────────────────────────────

class HeartbeatMonitor:
    """Pings Ollama + collects basic system metrics at a fixed interval."""

    def __init__(self, interval_s: int = 30) -> None:
        self._interval = interval_s
        self._ollama = OllamaClient()
        self._task: asyncio.Task | None = None
        self._last_status: dict[str, Any] = {}
        self._last_ollama_up: bool | None = None

    @property
    def last_status(self) -> dict[str, Any]:
        return self._last_status

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="heartbeat")

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        while True:
            status = await self._collect()
            self._last_status = status
            await bus.emit(Event("heartbeat", status))
            await asyncio.sleep(self._interval)

    async def _collect(self) -> dict[str, Any]:
        ollama_st = await self._ollama.status()

        # Fire state-change events
        if self._last_ollama_up is True and not ollama_st.available:
            await bus.emit(Event("ollama_down", {"error": ollama_st.error}))
        elif self._last_ollama_up is False and ollama_st.available:
            await bus.emit(Event("ollama_up", {"models": ollama_st.models}))
        self._last_ollama_up = ollama_st.available

        payload: dict[str, Any] = {
            "ts": time.time(),
            "ollama": {
                "available": ollama_st.available,
                "models": ollama_st.models,
                "error": ollama_st.error,
            },
        }

        # Optional: psutil metrics (graceful if not installed)
        try:
            import psutil  # type: ignore
            payload["system"] = {
                "cpu_pct": psutil.cpu_percent(interval=None),
                "mem_used_gb": round(psutil.virtual_memory().used / 1e9, 2),
                "mem_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
                "disk_free_gb": round(psutil.disk_usage("/").free / 1e9, 2),
            }
        except ImportError:
            pass

        return payload


# ── Cron scheduler ────────────────────────────────────────────────────────────

AsyncTask = Callable[[], Coroutine[Any, Any, Any]]


@dataclass
class ScheduledJob:
    name: str
    fn: AsyncTask
    interval_s: int
    last_run: float = 0.0
    run_count: int = 0
    last_error: str = ""
    enabled: bool = True


class CronScheduler:
    """
    Simple interval-based scheduler built on asyncio.
    For cron-expression support add `croniter` or `apscheduler`.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, ScheduledJob] = {}
        self._task: asyncio.Task | None = None

    def add(self, name: str, fn: AsyncTask, interval_s: int) -> None:
        self._jobs[name] = ScheduledJob(name=name, fn=fn, interval_s=interval_s)
        logger.info("scheduled job added: %s every %ds", name, interval_s)

    def remove(self, name: str) -> bool:
        return bool(self._jobs.pop(name, None))

    def enable(self, name: str) -> None:
        if name in self._jobs:
            self._jobs[name].enabled = True

    def disable(self, name: str) -> None:
        if name in self._jobs:
            self._jobs[name].enabled = False

    def list(self) -> list[dict[str, Any]]:
        now = time.time()
        return [
            {
                "name": j.name,
                "interval_s": j.interval_s,
                "run_count": j.run_count,
                "last_run": j.last_run,
                "next_run_in_s": max(0, j.interval_s - (now - j.last_run)) if j.last_run else 0,
                "last_error": j.last_error,
                "enabled": j.enabled,
            }
            for j in self._jobs.values()
        ]

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="cron-scheduler")

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        while True:
            now = time.time()
            for job in list(self._jobs.values()):
                if not job.enabled:
                    continue
                if now - job.last_run >= job.interval_s:
                    asyncio.create_task(self._run_job(job))
            await asyncio.sleep(1)

    async def _run_job(self, job: ScheduledJob) -> None:
        job.last_run = time.time()
        job.run_count += 1
        try:
            result = await job.fn()
            await bus.emit(Event("task_done", {"job": job.name, "run": job.run_count, "result": str(result)[:500]}))
        except Exception as e:
            job.last_error = str(e)
            logger.exception("scheduled job %s failed", job.name)
            await bus.emit(Event("task_error", {"job": job.name, "error": str(e)}))


# ── File watcher (polling fallback) ──────────────────────────────────────────

class FileWatcher:
    """
    Poll a directory for changed files and emit 'file_changed' events.
    Uses watchdog if installed, otherwise falls back to mtime polling.
    """

    def __init__(self, path: str, interval_s: int = 5) -> None:
        self._path = path
        self._interval = interval_s
        self._mtimes: dict[str, float] = {}
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._poll(), name="file-watcher")

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _poll(self) -> None:
        import os
        while True:
            try:
                for root, _, files in os.walk(self._path):
                    if any(skip in root for skip in (".git", "node_modules", "__pycache__")):
                        continue
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        try:
                            mtime = os.path.getmtime(fpath)
                        except OSError:
                            continue
                        prev = self._mtimes.get(fpath)
                        if prev is not None and mtime != prev:
                            await bus.emit(Event("file_changed", {"path": fpath, "mtime": mtime}))
                        self._mtimes[fpath] = mtime
            except Exception as e:
                logger.warning("file watcher error: %s", e)
            await asyncio.sleep(self._interval)


# ── Module-level singletons ───────────────────────────────────────────────────

heartbeat = HeartbeatMonitor()
scheduler = CronScheduler()
