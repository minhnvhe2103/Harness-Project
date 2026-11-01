import os
from pathlib import Path


def load_env(env_path: str = ".env"):
    """Simple .env file loader without third-party dependencies."""
    p = Path(env_path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v


# Load environment variables on module import
load_env()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-3.1-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

