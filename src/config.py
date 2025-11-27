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
    cleanup_tokens: Tuple[str, ...]
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


def _prefixed_env(prefix: str, suffix: str) -> str:
    """Fetch environment variables such as ``GPT2_MODEL_NAME``."""

    return _require_env(f"{prefix}_{suffix}")


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "gpt2": {
        "key": "gpt2",
        "label": "GPT-2 Small",
        "model_name": _prefixed_env("GPT2", "MODEL_NAME"),
        "hook_point": _prefixed_env("GPT2", "HOOK_POINT"),
        "sae_release": _prefixed_env("GPT2", "SAE_RELEASE"),
        "sae_id": _prefixed_env("GPT2", "SAE_ID"),
        "feature_ids": {
            "shakespeare": 14599,
            "flavors": 15952,
            "math": 15255,
        },
        "dtype": torch.float32,
        "cleanup_tokens": ("<|endoftext|>", "<|bos|>", "<bos>", "</s>"),
        "preload": True,
    },
    "gemma": {
        "key": "gemma",
        "label": "Gemma 2B",
        "model_name": _prefixed_env("GEMMA", "MODEL_NAME"),
        "hook_point": _prefixed_env("GEMMA", "HOOK_POINT"),
        "sae_release": _prefixed_env("GEMMA", "SAE_RELEASE"),
        "sae_id": _prefixed_env("GEMMA", "SAE_ID"),
        "feature_ids": {
            "shakespeare": 28765,
            "flavors": 13307,
            "math": 22111,
        },
        "dtype": torch.float16,
        "cleanup_tokens": ("</s>", "<bos>", "<s>", "<|endoftext|>"),
        "preload": False,
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
