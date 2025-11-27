from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable, Dict, Mapping, Optional, Tuple

import torch
from sae_lens import SAE
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from config import AVAILABLE_MODELS, DEFAULT_MODEL_KEY, MODEL_CONFIGS, ModelConfig

HookFn = Callable[[Tensor, HookPoint], Tensor]
ALPHA_EPSILON = 1e-6
HTML_TAG_PATTERN = re.compile(r"</?(?!bos\b)[^>]+>")


def _get_device() -> str:
    """Pick the first available accelerator so tensors move automatically.

    Args:
        None

    Returns:
        Lowercase device string understood by PyTorch (cuda, mps, or cpu).
    """

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _get_device()


def get_device() -> str:
    """Return the resolved execution device so the UI can display it.

    Args:
        None

    Returns:
        Lowercase device string identical to what PyTorch expects.
    """

    return DEVICE


def _resolve_dtype(config: ModelConfig) -> torch.dtype:
    """Upgrade the dtype if the host device cannot handle lightweight floats.

    Args:
        config: Model configuration describing preferred dtype.

    Returns:
        ``torch.dtype`` that is safe to use on the detected device.
    """

    if DEVICE == "cpu" and config["dtype"] == torch.float16:
        return torch.float32
    return config["dtype"]


@lru_cache(maxsize=None)
def _load_model(model_key: str) -> HookedTransformer:
    """Load and cache the transformer weights for one model key.

    Args:
        model_key: Identifier inside ``MODEL_CONFIGS``.

    Returns:
        A ``HookedTransformer`` ready for inference.
    """

    config = MODEL_CONFIGS[model_key]
    dtype = _resolve_dtype(config)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    model = HookedTransformer.from_pretrained(
        config["model_name"],
        device=DEVICE,
        dtype=dtype,
    )
    model.eval()
    return model


@lru_cache(maxsize=None)
def _load_sae(model_key: str) -> SAE:
    """Load the companion sparse autoencoder for the requested model.

    Args:
        model_key: Identifier inside ``MODEL_CONFIGS``.

    Returns:
        The cached ``SAE`` instance placed on the right device.
    """

    config = MODEL_CONFIGS[model_key]
    return SAE.from_pretrained(
        release=config["sae_release"],
        sae_id=config["sae_id"],
        device=DEVICE,
    )


def _generate_text(
    model: HookedTransformer,
    tokens: Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    hook_fn: Optional[HookFn],
    hook_point: Optional[str],
    cleanup_tokens: Tuple[str, ...],
) -> str:
    """Decode text while optionally inserting a steering hook on the fly.

    Args:
        model: Loaded ``HookedTransformer`` instance.
        tokens: Prompt tokens returned by ``model.to_tokens``.
        max_new_tokens: Number of tokens to sample beyond the prompt.
        temperature: Softmax temperature for sampling.
        top_p: Nucleus sampling cut-off.
        hook_fn: Optional function applied at ``hook_point`` each forward pass.
        hook_point: Name of the hook location inside the model graph.
        cleanup_tokens: Tuple of strings that should be stripped from raw output.

    Returns:
        Cleaned text string produced by the model.
    """

    generation_kwargs = dict(
        input=tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        stop_at_eos=False,
        verbose=False,
    )

    with torch.no_grad():
        if hook_fn is None:
            out_tokens = model.generate(**generation_kwargs)
        else:
            if hook_point is None:
                raise ValueError("hook_point must be set when hook_fn is provided.")
            with model.hooks(fwd_hooks=[(hook_point, hook_fn)]):
                out_tokens = model.generate(**generation_kwargs)

    text = model.to_string(out_tokens[0])
    for token in cleanup_tokens:
        if token:
            text = text.replace(token, "")
    text = HTML_TAG_PATTERN.sub("", text)
    return text.strip()


def _build_feature_alpha_map(
    alphas: Mapping[str, float],
    feature_ids: Mapping[str, int],
) -> Dict[int, float]:
    """Translate friendly feature names into SAE neuron ids.

    Args:
        alphas: User-facing feature strengths keyed by nickname.
        feature_ids: Mapping from nickname to actual SAE neuron index.

    Returns:
        Dictionary keyed by neuron id with the requested alpha weights.
    """

    mapped: Dict[int, float] = {}
    for name, alpha in alphas.items():
        if abs(alpha) < ALPHA_EPSILON:
            continue
        feature_idx = feature_ids.get(name)
        if feature_idx is None:
            continue
        mapped[feature_idx] = alpha
    return mapped


def _make_steering_hook(sae: SAE, feature_alphas: Mapping[int, float]) -> HookFn:
    """Create a hook that tweaks SAE features immediately before decoding.

    Args:
        sae: Loaded sparse autoencoder tied to the transformer block.
        feature_alphas: Dictionary of neuron ids to additive strengths.

    Returns:
        Callable that Streamlit passes into ``model.hooks``.
    """

    active = {idx: alpha for idx, alpha in feature_alphas.items() if abs(alpha) > ALPHA_EPSILON}
    if not active:
        def noop(resid_pre: Tensor, hook: HookPoint) -> Tensor:
            return resid_pre

        return noop

    def steering_hook(resid_pre: Tensor, hook: HookPoint) -> Tensor:
        with torch.no_grad():
            resid_last = resid_pre[:, -1:, :]
            feats = sae.encode(resid_last)
            recon = sae.decode(feats)

            feats_steered = feats.clone()
            for feat_idx, alpha in active.items():
                feats_steered[..., feat_idx] += alpha

            recon_steered = sae.decode(feats_steered)
            resid_pre[:, -1:, :] = resid_last + (recon_steered - recon)
        return resid_pre

    return steering_hook


def generate_baseline(
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    model_key: str = DEFAULT_MODEL_KEY,
) -> str:
    """Produce plain text without touching any SAE features.

    Args:
        prompt: Text seed passed straight into the transformer.
        max_new_tokens: Number of tokens to sample after the prompt.
        temperature: Sampling temperature for creativity.
        top_p: Cumulative probability cut-off for nucleus sampling.
        model_key: Which configuration inside ``MODEL_CONFIGS`` to use.

    Returns:
        String continuation exactly as the base model produced it.
    """

    config = MODEL_CONFIGS[model_key]
    model = _load_model(model_key)
    tokens = model.to_tokens(prompt)
    return _generate_text(
        model=model,
        tokens=tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        hook_fn=None,
        hook_point=None,
        cleanup_tokens=config["cleanup_tokens"],
    )


def generate_steered(
    prompt: str,
    alphas: Mapping[str, float],
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    model_key: str = DEFAULT_MODEL_KEY,
) -> str:
    """Generate text while boosting the requested SAE features.

    Args:
        prompt: Text seed passed to the transformer.
        alphas: Mapping of feature names to steering strengths.
        max_new_tokens: Number of tokens to sample after the prompt.
        temperature: Sampling temperature for creativity.
        top_p: Cumulative probability cut-off for nucleus sampling.
        model_key: Which configuration inside ``MODEL_CONFIGS`` to use.

    Returns:
        String continuation after steering nudges are injected.
    """

    config = MODEL_CONFIGS[model_key]
    model = _load_model(model_key)
    sae = _load_sae(model_key)

    feature_alphas_idx = _build_feature_alpha_map(alphas, config["feature_ids"])

    hook_fn = _make_steering_hook(sae, feature_alphas_idx)
    tokens = model.to_tokens(prompt)
    return _generate_text(
        model=model,
        tokens=tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        hook_fn=hook_fn,
        hook_point=config["hook_point"],
        cleanup_tokens=config["cleanup_tokens"],
    )


def run_comparison(
    prompt: str,
    alphas: Mapping[str, float],
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    model_key: str = DEFAULT_MODEL_KEY,
) -> Dict[str, str]:
    """Return baseline and steered continuations for easy side-by-side reading.

    Args:
        prompt: Shared text seed for both runs.
        alphas: Mapping of feature names to steering strengths.
        max_new_tokens: Number of tokens to sample in each run.
        temperature: Sampling temperature for creativity.
        top_p: Cumulative probability cut-off for nucleus sampling.
        model_key: Which configuration inside ``MODEL_CONFIGS`` to use.

    Returns:
        Dictionary with ``baseline`` and ``steered`` continuations.
    """

    baseline = generate_baseline(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        model_key=model_key,
    )
    steered = generate_steered(
        prompt=prompt,
        alphas=alphas,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        model_key=model_key,
    )
    return {"baseline": baseline, "steered": steered}


def warm_start_resources() -> None:
    """Eagerly pull the model and SAE weights into cache for smoother UX.

    Args:
        None

    Returns:
        None
    """

    for key, cfg in MODEL_CONFIGS.items():
        if not cfg["preload"]:
            continue
        _load_model(key)
        _load_sae(key)


__all__ = [
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL_KEY",
    "generate_baseline",
    "generate_steered",
    "get_device",
    "run_comparison",
    "warm_start_resources",
]
