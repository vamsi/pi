"""pi-web-ui: Web UI for AI chat with FastAPI and WebSockets."""

from pi.web.app import create_app
from pi.web.config import Config

__all__ = ["Config", "create_app"]
