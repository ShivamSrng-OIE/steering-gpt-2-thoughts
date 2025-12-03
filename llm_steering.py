import torch
import numpy as np
import pandas as pd
from sae_lens import SAE
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer


FEATURES = {
    "Shakespeare": {
        "index": 14599,
        "prompt": "To be or not to be, that is the",
        "description": "Pushes the model toward Shakespearean style and content.",
    },
    "Flavors": {
        "index": 15952,
        "prompt": "The ice cream shop sold many",
        "description": "Tasting notes, flavor words, sensory descriptions.",
    },
    "Math": {
        "index": 15255,
        "prompt": "The solution to the equation is",
        "description": "Math notation, formulas, LaTeX-style symbols.",
    },
    "Programming": {
        "index": 4545,
        "prompt": "def compute_factorial(n):",
        "description": "Code syntax, programming concepts, function definitions.",
    },
}
HOOK_POINT = "blocks.5.hook_resid_pre"


def load_model_and_sae():
    """Load GPT-2 small and the SAE decoder weights."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    try:
        sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
            release="gpt2-small-res-jb",
            sae_id=HOOK_POINT,
            device=device,
        )
    except AttributeError:
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id=HOOK_POINT,
            device=device,
        )

    model_kwargs = dict(cfg_dict.get("model_from_pretrained_kwargs", {}) or {})
    if "torch_dtype" in model_kwargs and "dtype" not in model_kwargs:
        model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")

    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        device=device,
        **model_kwargs,
    )

    return model, sae, device


def get_steering_hook(steering_vector: torch.Tensor, coefficient: float):
    """Create a hook that injects the steering vector at the hook point."""

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
    max_new_tokens: int = 80,
    temperature: float = 1.0,
) -> str:
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
        if steering_vector is None or coefficient == 0.0:
            out_tokens = model.generate(**gen_kwargs)
        else:
            steering_hook = get_steering_hook(steering_vector, coefficient)
            with model.hooks(fwd_hooks=[(HOOK_POINT, steering_hook)]):
                out_tokens = model.generate(**gen_kwargs)

    text = model.to_string(out_tokens[0])
    return text.replace("\n", "\\n").replace("<|endoftext|>", "").strip()


def get_next_token_logprobs(
    model: HookedTransformer,
    prompt: str,
    steering_vector: torch.Tensor | None,
    coefficient: float,
) -> torch.Tensor:
    tokens = model.to_tokens(prompt)

    with torch.no_grad():
        if steering_vector is None or coefficient == 0.0:
            logits = model(tokens)
        else:
            steering_hook = get_steering_hook(steering_vector, coefficient)
            with model.hooks(fwd_hooks=[(HOOK_POINT, steering_hook)]):
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
    top_k: int = 20,
) -> pd.DataFrame:
    steering_vector = sae.W_dec[feature_idx]

    base_lp = get_next_token_logprobs(model, prompt, None, 0.0)
    steer_lp = get_next_token_logprobs(model, prompt, steering_vector, coeff)

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


def sample_top_p_from_logprobs(logprobs: torch.Tensor, top_p: float = 0.9) -> int:
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
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
    progress_callback=None,
):
    tokens = model.to_tokens(prompt)
    base_steps = []
    steer_steps = []

    for step in tqdm(range(max_new_tokens), desc="Per-token NLL steps", leave=False):
        with torch.no_grad():
            logits_base = model(tokens)
            last_logits_base = logits_base[0, -1, :]
            scaled_base = last_logits_base / temperature
            logprobs_base = torch.log_softmax(scaled_base, dim=-1)

            if steering_vector is not None and coeff != 0.0:
                steering_hook = get_steering_hook(steering_vector, coeff)
                with model.hooks(fwd_hooks=[(HOOK_POINT, steering_hook)]):
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
    "FEATURES",
    "HOOK_POINT",
    "load_model_and_sae",
    "generate_text",
    "generate_stepwise_dists_same_prefix",
    "build_step_nll_df",
    "build_nll_dataframe",
]
