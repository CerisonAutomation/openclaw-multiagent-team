"""python -m tools.inspector  →  starts the inspector server."""
import uvicorn
from .config import get_config

cfg = get_config()
uvicorn.run("tools.inspector.server:app", host=cfg.server_host,
            port=cfg.server_port, reload=True, log_level="info")
