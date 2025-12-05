import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sae_lens import SAE
from tqdm.auto import tqdm
from dotenv import load_dotenv
from functools import lru_cache
from transformer_lens import HookedTransformer


load_dotenv(".env", override=False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _require_env(
    key: str
) -> str:
    """
    Retrieve a required environment variable or raise an error if not set.
    
    Args:
        key (str): The name of the environment variable to retrieve.
    
    Returns:
        str: The value of the environment variable.
    """
    
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


def _prefixed_env(prefix: str, suffix: str) -> str:
    return _require_env(f"{prefix}_{suffix}")


FEATURE_LIBRARY = {
    "shakespeare": {
        "label": "Shakespeare",
        "prompt": "To be or not to be, that is the",
        "description": "Pushes the model toward Shakespearean style and content.",
    },
    "flavors": {
        "label": "Flavors",
        "prompt": "The ice cream shop sold many",
        "description": "Tasting notes, flavor words, sensory descriptions.",
    },
    "math": {
        "label": "Math",
        "prompt": "The solution to the equation is",
        "description": "Math notation, formulas, LaTeX-style symbols.",
    },
    "ethical": {
        "label": "Ethical voice",
        "prompt": "We must consider the consequences because",
        "description": "Frames responses with moral language and responsible action cues.",
    },
    "protection_of_lives": {
        "label": "Protect lives",
        "prompt": "To keep everyone safe, we should",
        "description": "Emphasizes safety, harm reduction, and protective instructions.",
    },
    "violence": {
        "label": "Violence",
        "prompt": "The conflict escalated when",
        "description": "Highlights aggression, conflict, and risk language for diagnostics.",
    },
    "reasoning": {
        "label": "Reasoning",
        "prompt": "Mathmatical proof: 2 + 2 =",
        "description": "Encourages logical analysis, critical thinking, and structured argumentation.",
    },
}


MODEL_CONFIGS = {
    "gpt2": {
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
    },
    "gemma": {
        "label": "Gemma 2B",
        "model_name": _prefixed_env("GEMMA", "MODEL_NAME"),
        "hook_point": _prefixed_env("GEMMA", "HOOK_POINT"),
        "sae_release": _prefixed_env("GEMMA", "SAE_RELEASE"),
        "sae_id": _prefixed_env("GEMMA", "SAE_ID"),
        "feature_ids": {
            "ethical": 3372,
            "protection_of_lives": 4562,
            "violence": 14807,
            "reasoning": 12988,
        },
        "dtype": torch.float16,
        "cleanup_tokens": ("</s>", "<bos>", "<s>", "<|endoftext|>"),
    },
}
AVAILABLE_MODELS = {key: cfg["label"] for key, cfg in MODEL_CONFIGS.items()}
DEFAULT_MODEL_KEY = "gpt2"



def _resolve_dtype(
    config: dict
) -> torch.dtype:
    """
    Determine the appropriate torch dtype for model loading based on config and device.
    
    Args:
        config (dict): The model configuration dictionary.
    
    Returns:
        torch.dtype: The resolved torch dtype.
    """
    
    target_dtype = config.get("dtype", torch.float32)
    if DEVICE == "cpu" and target_dtype == torch.float16:
        return torch.float32
    return target_dtype



@lru_cache(maxsize=None)
def _load_bundle(
    model_key: str
) -> tuple[HookedTransformer, SAE, str]:
    """
    Load and return the transformer model and SAE for the specified model key.
    
    Args:
        model_key (str): The key identifying the model configuration to load.
    
    Returns:
        tuple[HookedTransformer, SAE, str]: The loaded model, SAE, and device string.
    """

    if model_key not in MODEL_CONFIGS:
        raise KeyError(f"Unknown model key: {model_key}")

    config = MODEL_CONFIGS[model_key]
    device = DEVICE
    torch.set_grad_enabled(False)

    sae_kwargs = dict(
        release=config["sae_release"],
        sae_id=config["sae_id"],
        device=device,
    )
    cfg_dict = {}
    try:
        sae, cfg_dict, _ = SAE.from_pretrained_with_cfg_and_sparsity(**sae_kwargs)
    except AttributeError:
        sae = SAE.from_pretrained(**sae_kwargs)
    except Exception as exc:
        raise RuntimeError(f"Failed to load SAE for model '{model_key}'") from exc

    model_kwargs = dict(cfg_dict.get("model_from_pretrained_kwargs", {}) or {})
    if "torch_dtype" in model_kwargs and "dtype" not in model_kwargs:
        model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")

    model_kwargs.setdefault("dtype", _resolve_dtype(config))
    model_kwargs.setdefault("device", device)

    try:
        model = HookedTransformer.from_pretrained(
            config["model_name"],
            **model_kwargs,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load transformer '{config['model_name']}'") from exc

    model.eval()
    return model, sae, device


def load_model_and_sae(
    model_key: str = DEFAULT_MODEL_KEY
) -> tuple[HookedTransformer, SAE, str]:
    """
    Return the cached transformer + SAE bundle for the requested model.
    
    Args:
        model_key (str): The key identifying the model configuration to load.
    
    Returns:
        tuple[HookedTransformer, SAE, str]: The loaded model, SAE, and device string.
    """

    return _load_bundle(model_key)


def get_device() -> str:
    return DEVICE


def get_steering_hook(
    steering_vector: torch.Tensor, 
    coefficient: float
):
    """
    Create a hook that injects the steering vector at the hook point.
    
    Args:
        steering_vector (torch.Tensor): The steering vector to inject.
        coefficient (float): The scaling coefficient for the steering vector.
    
    Returns:
        function: A hook function for use in the model.
    """

    def hook(resid_pre, hook=None):
        v = steering_vector.to(device=resid_pre.device, dtype=resid_pre.dtype)
        resid_pre = resid_pre.clone()
        resid_pre[:, -1, :] += v * coefficient
        return resid_pre

    return hook


def generate_text(
    model: HookedTransformer,
    prompt: str,
    steering_vector: torch.Tensor | None,
    coefficient: float,
    hook_point: str | None,
    *,
    max_new_tokens: int = 80,
    temperature: float = 1.0,
) -> str:
    """
    Generate text from the model with optional steering.
    
    Args:
        model (HookedTransformer): The transformer model to use for generation.
        prompt (str): The input prompt text.
        steering_vector (torch.Tensor | None): The steering vector to apply.
        coefficient (float): The scaling coefficient for the steering vector.
        hook_point (str | None): The hook point in the model to apply the steering.
        max_new_tokens (int): The maximum number of new tokens to generate.
        temperature (float): The sampling temperature.
    
    Returns:
        str: The generated text.
    """
    
    tokens = model.to_tokens(prompt)
    gen_kwargs = dict(
        input=tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=1,
        do_sample=True,
        stop_at_eos=False,
        verbose=False,
    )

    with torch.no_grad():
        if steering_vector is None or coefficient == 0.0 or not hook_point:
            out_tokens = model.generate(**gen_kwargs)
        else:
            steering_hook = get_steering_hook(steering_vector, coefficient)
            with model.hooks(fwd_hooks=[(hook_point, steering_hook)]):
                out_tokens = model.generate(**gen_kwargs)

    text = model.to_string(out_tokens[0])
    return text.replace("\n", "\\n").replace("<|endoftext|>", "").strip()


def get_next_token_logprobs(
    model: HookedTransformer,
    prompt: str,
    steering_vector: torch.Tensor | None,
    coefficient: float,
    hook_point: str | None,
) -> torch.Tensor:
    """
    Get the log-probabilities for the next token given a prompt, with optional steering.
    
    Args:
        model (HookedTransformer): The transformer model to use.
        prompt (str): The input prompt text.
        steering_vector (torch.Tensor | None): The steering vector to apply.
        coefficient (float): The scaling coefficient for the steering vector.
        hook_point (str | None): The hook point in the model to apply the steering.
    
    Returns:
        torch.Tensor: The log-probabilities for the next token.
    """
    
    tokens = model.to_tokens(prompt)

    with torch.no_grad():
        if steering_vector is None or coefficient == 0.0 or not hook_point:
            logits = model(tokens)
        else:
            steering_hook = get_steering_hook(steering_vector, coefficient)
            with model.hooks(fwd_hooks=[(hook_point, steering_hook)]):
                logits = model(tokens)

        last_logits = logits[0, -1, :]
        logprobs = torch.log_softmax(last_logits, dim=-1)

    return logprobs


def build_nll_dataframe(
    model: HookedTransformer,
    sae: SAE,
    feature_idx: int,
    prompt: str,
    coeff: float,
    hook_point: str | None,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Build a DataFrame comparing NLLs for the next token with and without steering.
    
    Args:
        model (HookedTransformer): The transformer model to use.
        sae (SAE): The SAE containing the steering vectors.
        feature_idx (int): The index of the feature to use for steering.
        prompt (str): The input prompt text.
        coeff (float): The scaling coefficient for the steering vector.
        hook_point (str | None): The hook point in the model to apply the steering.
        top_k (int): The number of top tokens to include in the DataFrame.
    
    Returns:
        pd.DataFrame: A DataFrame comparing NLLs for the next token.
    """
    
    steering_vector = sae.W_dec[feature_idx]

    base_lp = get_next_token_logprobs(model, prompt, None, 0.0, hook_point)
    steer_lp = get_next_token_logprobs(model, prompt, steering_vector, coeff, hook_point)

    base_nll = (-base_lp).detach().cpu().numpy()
    steer_nll = (-steer_lp).detach().cpu().numpy()

    top_idx = np.argsort(base_nll)[:top_k]

    rows = []
    for tid in top_idx:
        token_str = model.tokenizer.decode([int(tid)])
        rows.append({"Token": token_str, "Run": "Baseline", "NLL": float(base_nll[tid])})
        rows.append(
            {
                "Token": token_str,
                "Run": f"Steered (α={coeff:.1f})",
                "NLL": float(steer_nll[tid]),
            }
        )

    return pd.DataFrame(rows)


def sample_top_p_from_logprobs(
    logprobs: torch.Tensor, 
    top_p: float = 0.9
) -> int:
    """
    Sample a token ID from the log-probabilities using top-p (nucleus) sampling.
    
    Args:
        logprobs (torch.Tensor): The log-probabilities for the tokens.
        top_p (float): The cumulative probability threshold for top-p sampling.
    
    Returns:
        int: The sampled token ID.
    """
    
    probs = torch.exp(logprobs)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)

    cutoff_mask = cumulative <= top_p
    cutoff_mask[0] = True
    cutoff_idx = cutoff_mask.nonzero(as_tuple=False)[-1].item()

    keep_probs = sorted_probs[: cutoff_idx + 1]
    keep_indices = sorted_indices[: cutoff_idx + 1]
    keep_probs = keep_probs / keep_probs.sum()

    choice = torch.multinomial(keep_probs, 1).item()
    return int(keep_indices[choice])


def generate_stepwise_dists_same_prefix(
    model: HookedTransformer,
    prompt: str,
    steering_vector: torch.Tensor | None,
    coeff: float,
    hook_point: str | None,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
    progress_callback=None,
) -> tuple[list[int], list[torch.Tensor], list[torch.Tensor]]:
    """
    Generate text step-by-step, recording log-probability distributions at each step.
    
    Args:
        model (HookedTransformer): The transformer model to use.
        prompt (str): The input prompt text.
        steering_vector (torch.Tensor | None): The steering vector to apply.
        coeff (float): The scaling coefficient for the steering vector.
        hook_point (str | None): The hook point in the model to apply the steering.
        max_new_tokens (int): The maximum number of new tokens to generate.
        temperature (float): The sampling temperature.
        top_p (float): The cumulative probability threshold for top-p sampling.
        progress_callback (function | None): Optional callback for progress updates.
    
    Returns:
        tuple[list[int], list[torch.Tensor], list[torch.Tensor]]:
        A tuple containing the generated token IDs, baseline log-probability steps,
        and steered log-probability steps.
    """

    tokens = model.to_tokens(prompt)
    base_steps = []
    steer_steps = []

    for step in tqdm(range(max_new_tokens), desc="Per-token NLL steps", leave=False):
        with torch.no_grad():
            logits_base = model(tokens)
            last_logits_base = logits_base[0, -1, :]
            scaled_base = last_logits_base / temperature
            logprobs_base = torch.log_softmax(scaled_base, dim=-1)

            if steering_vector is not None and coeff != 0.0 and hook_point:
                steering_hook = get_steering_hook(steering_vector, coeff)
                with model.hooks(fwd_hooks=[(hook_point, steering_hook)]):
                    logits_steer = model(tokens)
                last_logits_steer = logits_steer[0, -1, :]
                scaled_steer = last_logits_steer / temperature
                logprobs_steer = torch.log_softmax(scaled_steer, dim=-1)
            else:
                logprobs_steer = logprobs_base.clone()

        base_steps.append(logprobs_base.detach().cpu())
        steer_steps.append(logprobs_steer.detach().cpu())

        if progress_callback is not None:
            progress_callback(step + 1, max_new_tokens)

        next_token_id = sample_top_p_from_logprobs(logprobs_base, top_p=top_p)
        next_token_tensor = torch.tensor([[next_token_id]], device=tokens.device)
        tokens = torch.cat([tokens, next_token_tensor], dim=1)

    return tokens[0].cpu().tolist(), base_steps, steer_steps


def build_step_nll_df(
    model: HookedTransformer,
    base_logprobs: torch.Tensor,
    steered_logprobs: torch.Tensor,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Build a DataFrame comparing stepwise NLLs for top tokens.
    
    Args:
        model (HookedTransformer): The transformer model to use.
        base_logprobs (torch.Tensor): The baseline log-probabilities.
        steered_logprobs (torch.Tensor): The steered log-probabilities.
        top_k (int): The number of top tokens to include in the DataFrame.
    
    Returns:
        pd.DataFrame: A DataFrame comparing stepwise NLLs for top tokens.
    """
    
    base_lp = base_logprobs.detach().numpy()
    steer_lp = steered_logprobs.detach().numpy()

    base_nll = -base_lp
    steer_nll = -steer_lp

    top_idx = np.argsort(base_nll)[:top_k]

    rows = []
    for tid in top_idx:
        token_str = model.tokenizer.decode([int(tid)])
        rows.append({"Token": token_str, "Run": "Baseline", "NLL": float(base_nll[tid])})
        rows.append({"Token": token_str, "Run": "Steered", "NLL": float(steer_nll[tid])})

    return pd.DataFrame(rows)


__all__ = [
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL_KEY",
    "FEATURE_LIBRARY",
    "MODEL_CONFIGS",
    "build_nll_dataframe",
    "build_step_nll_df",
    "generate_stepwise_dists_same_prefix",
    "generate_text",
    "get_device",
    "load_model_and_sae",
]
