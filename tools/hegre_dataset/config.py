"""Database configuration loader.

Reads config/database.yml and returns connection strings for the
active environment (controlled by EIDOLON_ENV, defaults to "development").

For tests, set EIDOLON_ENV=test to use a separate database.
For local development without PG, set EIDOLON_DB_URL=sqlite:///:memory:
to use an in-memory SQLite via SQLAlchemy (same ORM, lighter backend).
"""

import os
from pathlib import Path
from functools import lru_cache

try:
    import yaml
except ImportError:
    yaml = None  # config file not available without pyyaml


@lru_cache(maxsize=1)
def _load_config() -> dict:
    """Load database.yml. Cached — file is read once per process."""
    if yaml is None:
        raise ImportError(
            "pyyaml is required to load config/database.yml. "
            "Install with: pip install pyyaml"
        )
    config_path = Path(__file__).resolve().parent.parent / "config" / "database.yml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Database config not found at {config_path}. "
            "Create config/database.yml with your PostgreSQL connection settings."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


def database_url(env: str | None = None) -> str:
    """Return the SQLAlchemy database URL for the given environment.

    Args:
        env: Environment name (default: EIDOLON_ENV or "development").

    Returns:
        SQLAlchemy connection URL string (e.g. "postgresql+psycopg2://user@host/dbname").

    Raises:
        ImportError: if pyyaml is not installed.
        FileNotFoundError: if config/database.yml does not exist.
        KeyError: if the environment is not defined in config.
    """
    if env is None:
        env = os.environ.get("EIDOLON_ENV", "development")

    # Allow override via environment variable (for CI/test without config file)
    override = os.environ.get("EIDOLON_DB_URL")
    if override:
        return override

    cfg = _load_config()
    if env not in cfg:
        raise KeyError(
            f"Environment '{env}' not found in config/database.yml. "
            f"Available environments: {list(cfg.keys())}"
        )

    db = cfg[env]
    return f"postgresql+psycopg2://{db['user']}@{db['host']}/{db['dbname']}"
