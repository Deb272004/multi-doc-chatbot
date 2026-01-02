from pathlib import Path
import os
import yaml


def _project_root() -> Path:
    """
    Returns the project root directory.

    Assumes this file is located at:
    multi_doc_chat/utils/config_loader.py
    """
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | None = None) -> dict:
    """
    Load YAML configuration file.

    Priority order:
    1. Explicit config_path argument
    2. CONFIG_PATH environment variable
    3. <project_root>/config/config.yaml
    """
    env_path = os.getenv("CONFIG_PATH")

    if config_path is None:
        config_path = env_path or (_project_root() / "config" / "config.yaml")

    path = Path(config_path)

    # If relative path is provided, resolve from project root
    if not path.is_absolute():
        path = _project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
