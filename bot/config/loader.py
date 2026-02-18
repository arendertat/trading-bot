"""Configuration loader with environment variable substitution"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from bot.config.models import BotConfig
from bot.config.validator import validate_config_constraints


def substitute_env_vars(data: Any) -> Any:
    """
    Recursively substitute environment variables in configuration data.

    Supports ${VAR_NAME} syntax in string values.
    """
    if isinstance(data, dict):
        return {key: substitute_env_vars(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [substitute_env_vars(item) for item in data]
    elif isinstance(data, str):
        # Match ${VAR_NAME} pattern
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, data)
        result = data
        for var_name in matches:
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ValueError(
                    f"Environment variable {var_name} not found but required in config"
                )
            result = result.replace(f"${{{var_name}}}", env_value)
        return result
    else:
        return data


def _validate_config_path(config_path: str) -> Path:
    """
    Bulgu 1.1: Validate config path against path traversal attacks.

    Resolves the path and ensures it ends with .json extension.
    Does not restrict to a specific directory since the bot supports
    arbitrary config locations via CLI/env, but blocks obvious traversal patterns.

    Raises:
        ValueError: If path appears to be a traversal attempt or invalid
    """
    # Block obvious traversal patterns in the raw string before resolution
    normalized = config_path.replace("\\", "/")
    if ".." in normalized.split("/"):
        raise ValueError(
            f"Config path contains path traversal sequence: {config_path}"
        )

    config_file = Path(config_path)

    # Must be a .json file
    if config_file.suffix.lower() != ".json":
        raise ValueError(
            f"Config path must point to a .json file, got: {config_path}"
        )

    return config_file


def load_config(config_path: str = "config/config.json", load_env: bool = True) -> BotConfig:
    """
    Load and validate bot configuration from JSON file.

    Args:
        config_path: Path to config JSON file
        load_env: Whether to load .env file first (default: True)

    Returns:
        Validated BotConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails or path is unsafe
        json.JSONDecodeError: If config file is invalid JSON
    """
    # Load environment variables from .env file if requested
    if load_env:
        load_dotenv()

    # Bulgu 1.1: Validate path before opening
    config_file = _validate_config_path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        raw_config = json.load(f)

    # Substitute environment variables
    config_data = substitute_env_vars(raw_config)

    # Parse and validate with Pydantic
    try:
        config = BotConfig(**config_data)
        # Apply additional validation constraints
        validate_config_constraints(config)
    except Exception as e:
        raise ValueError(f"Config validation failed: {e}")

    return config


def validate_config_file(config_path: str = "config/config.json") -> None:
    """
    Validate a config file and print result.

    Args:
        config_path: Path to config file
    """
    try:
        config = load_config(config_path)
        print(f"[OK] Config validation successful")
        print(f"  Mode: {config.mode}")
        print(f"  Exchange: {config.exchange.name}")
        print(f"  Max positions: {config.risk.max_open_positions}")
        print(f"  Risk per trade: {config.risk.risk_per_trade_pct * 100:.2f}%")
    except Exception as e:
        print(f"[FAIL] Config validation failed:")
        print(f"  {e}")
        raise


if __name__ == "__main__":
    import sys

    # Allow passing config path as argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.json"
    validate_config_file(config_path)
