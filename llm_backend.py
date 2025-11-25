"""Utility helpers for running GPT-2 with optional SAE steering."""

from functools import lru_cache
from typing import Callable, Dict, Mapping, Optional

import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE

__all__ = ["generate_baseline", "generate_steered"]

# -------------------------
# Global config
# -------------------------

DTYPE = torch.float32
MODEL_NAME = "gpt2-small"
SAE_RELEASE = "gpt2-small-res-jb"
SAE_HOOK_NAME = "blocks.5.hook_resid_pre"
END_OF_TEXT = "<|endoftext|>"
ALPHA_EPSILON = 1e-6

HookFn = Callable[[Tensor, HookPoint], Tensor]


def _get_device() -> str:
    """Pick the best available device: cuda, mps, or cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


DEVICE = _get_device()
print(f"[llm_backend] Using device: {DEVICE}, dtype: {DTYPE}, and model: {MODEL_NAME}.")

# Neuronpedia feature ids for this SAE
FEATURE_IDS: Dict[str, int] = {
    "shakespeare": 14599,
    "flavors": 15952,
    "math": 15255,
}


# -------------------------
# Model and SAE loading
# -------------------------

@lru_cache(maxsize=1)
def load_model() -> HookedTransformer:
    """Load GPT-2 Small once and reuse it across Streamlit reruns."""

    print(f"[llm_backend] Loading {MODEL_NAME} on {DEVICE} (dtype={DTYPE}).")
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
        dtype=DTYPE,
    )
    model.eval()
    return model


@lru_cache(maxsize=1)
def load_sae() -> SAE:
    """Load the Sparse Autoencoder aligned to ``SAE_HOOK_NAME``."""

    print(f"[llm_backend] Loading SAE {SAE_RELEASE}:{SAE_HOOK_NAME} on {DEVICE}.")
    sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_HOOK_NAME,
        device=DEVICE,
    )
    print(
        f"[llm_backend] SAE loaded (d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae})."
    )
    return sae


# -------------------------
# Shared helpers
# -------------------------


def _clean_text(raw_text: str) -> str:
    """Remove the GPT end token and tidy whitespace."""

    return raw_text.replace(END_OF_TEXT, "").strip()


def _prepare_tokens(model: HookedTransformer, prompt: str) -> Tensor:
    """Tokenize the prompt once so repeated runs stay fast."""

    return model.to_tokens(prompt)


def _sample_text(
    model: HookedTransformer,
    tokens: Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    hook_fn: Optional[HookFn] = None,
) -> str:
    """Run generation, optionally attaching a steering hook."""

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
            with model.hooks(fwd_hooks=[(SAE_HOOK_NAME, hook_fn)]):
                out_tokens = model.generate(**generation_kwargs)

    return _clean_text(model.to_string(out_tokens[0]))


def _build_feature_alpha_map(alphas: Mapping[str, float]) -> Dict[int, float]:
    """Translate friendly feature names to SAE indices while filtering zeros."""

    mapped: Dict[int, float] = {}
    for name, alpha in alphas.items():
        if abs(alpha) < ALPHA_EPSILON:
            continue
        feature_idx = FEATURE_IDS.get(name)
        if feature_idx is None:
            continue
        mapped[feature_idx] = alpha
    return mapped


# -------------------------
# Core generation helpers
# -------------------------

def generate_baseline(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Simple helper that runs GPT-2 without SAE steering."""

    model = load_model()
    tokens = _prepare_tokens(model, prompt)
    return _sample_text(
        model=model,
        tokens=tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def _make_steering_hook(
    sae: SAE,
    feature_alphas: Mapping[int, float],
) -> HookFn:
    """Build a hook that nudges selected SAE features on the final token."""

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


def generate_steered(
    prompt: str,
    alphas: Dict[str, float],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate text with SAE-based steering.

    alphas: dict mapping feature name -> alpha, for example:
        {
            "shakespeare": 2.0,
            "flavors": -1.0,
            "math": 0.0,
        }

    If all alphas are zero (or the dict is empty), this falls back
    to baseline generation.
    """
    model = load_model()
    sae = load_sae()

    feature_alphas_idx = _build_feature_alpha_map(alphas)
    if not feature_alphas_idx:
        return generate_baseline(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    hook_fn = _make_steering_hook(sae, feature_alphas_idx)
    tokens = _prepare_tokens(model, prompt)
    return _sample_text(
        model=model,
        tokens=tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        hook_fn=hook_fn,
    )
