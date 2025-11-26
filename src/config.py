"""Central configuration helpers for the SAE steering playground."""

import os
from pathlib import Path
from typing import Dict, Tuple, TypedDict

import torch
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env", override=False)


class ModelConfig(TypedDict):
    """Metadata bundle that explains how to load and steer one transformer."""
    key: str
    label: str
    model_name: str
    hook_point: str
    sae_release: str
    sae_id: str
    feature_ids: Dict[str, int]
    dtype: torch.dtype
    end_tokens: Tuple[str, ...]
    preload: bool


def _require_env(key: str) -> str:
    """Return the required environment variable or raise a loud error.

    Args:
        key: Name of the variable to fetch from the host environment.

    Returns:
        The resolved non-empty string value.

    Raises:
        RuntimeError: If the variable is unset, so configuration bugs surface fast.
    """

    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "gpt2": {
        "key": "gpt2",
        "label": "GPT-2 Small",
        "model_name": _require_env("MODEL_NAME"),
        "hook_point": _require_env("HOOK_POINT"),
        "sae_release": _require_env("SAE_RELEASE"),
        "sae_id": _require_env("SAE_ID"),
        "feature_ids": {
            "shakespeare": 14599,
            "flavors": 15952,
            "math": 15255,
        },
        "dtype": torch.float32,
        "end_tokens": ("<|endoftext|>",),
        "preload": True,
    },
}

DEFAULT_MODEL_KEY = "gpt2"
AVAILABLE_MODELS = {key: cfg["label"] for key, cfg in MODEL_CONFIGS.items()}
TYPEWRITER_DELAY = 0.02
FEATURE_DETAILS: Dict[str, Dict[str, str]] = {
    "shakespeare": {
        "label": "Shakespeare",
        "hint": "Play-like voice",
        "long": "Adds stage cues, poetic lines, and dramatic beats.",
    },
    "flavors": {
        "label": "Flavors",
        "hint": "Food talk",
        "long": "Talks about taste, smell, and texture like a food friend.",
    },
    "math": {
        "label": "Math",
        "hint": "Logic voice",
        "long": "Brings in steps, numbers, and tidy reasoning words.",
    },
}
