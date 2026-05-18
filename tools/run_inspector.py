"""Thin launcher — starts the inspector server.

Usage:
    python tools/run_inspector.py
    INSPECTOR_PORT=9000 python tools/run_inspector.py
"""
from __future__ import annotations

import sys
import os

# Ensure the project root is on sys.path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from tools.inspector.config import get_config
from tools.inspector.server import app  # noqa: F401 — registers all routes

if __name__ == "__main__":
    cfg = get_config()
    uvicorn.run(app, host=cfg.server_host, port=cfg.server_port, log_level="info")
