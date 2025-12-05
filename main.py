import torch
import altair as alt
import streamlit as st

from llm_steering import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL_KEY,
    FEATURE_LIBRARY,
    MODEL_CONFIGS,
    build_step_nll_df,
    generate_stepwise_dists_same_prefix,
    generate_text,
    load_model_and_sae,
)


@st.cache_resource
def get_model_and_sae(model_key: str):
    """Cache the heavy model + SAE load for the Streamlit session."""
    return load_model_and_sae(model_key)


def main():
    st.set_page_config(
        page_title="SAE Steering Playground - GPT-2 Small",
        layout="wide",
    )
    st.title("SAE Steering Playground for GPT-2 Small")
    st.divider()
    st.sidebar.header("Controls")

    model_options = list(AVAILABLE_MODELS.keys())
    default_model_index = model_options.index(DEFAULT_MODEL_KEY) if DEFAULT_MODEL_KEY in model_options else 0
    model_key = st.sidebar.selectbox(
        "Model",
        options=model_options,
        index=default_model_index,
        format_func=lambda k: AVAILABLE_MODELS[k],
    )
    model_config = MODEL_CONFIGS[model_key]
    model_label = AVAILABLE_MODELS[model_key]
    hook_point = model_config["hook_point"]

    available_feature_keys = [
        key for key in model_config["feature_ids"].keys() if key in FEATURE_LIBRARY
    ]
    if not available_feature_keys:
        st.sidebar.error("No features are configured for this model.")
        st.stop()

    feature_key = st.sidebar.selectbox(
        "Concept feature",
        options=available_feature_keys,
        format_func=lambda k: FEATURE_LIBRARY[k]["label"],
    )
    feature_meta = FEATURE_LIBRARY.get(feature_key, {"label": feature_key.title(), "prompt": "", "description": ""})
    feature_idx = model_config["feature_ids"].get(feature_key)
    if feature_idx is None:
        st.sidebar.error("Selected feature is not available for this model.")
        st.stop()
    feature_idx = int(feature_idx)

    model, sae, device = get_model_and_sae(model_key)
    st.sidebar.success(f"{model_label} ready on **{device.upper()}**")

    manual_seed_enabled = st.sidebar.checkbox(
        "Lock randomness (manual seed 0)",
        value=True,
        help="When enabled, both runs use the same RNG seed so differences come only from steering.",
    )
    
    coeff = st.sidebar.slider(
        "Steering strength α",
        min_value=-120.0,
        max_value=120.0,
        value=0.0,
        step=1.0,
        help="Negative values suppress the feature. Positive values amplify it.",
    )

    max_new_tokens = st.sidebar.slider(
        "Max new tokens",
        min_value=5,
        max_value=100,
        value=20,
        step=1,
    )

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )

    top_k = st.sidebar.slider(
        "Tokens in each NLL chart (top-k)",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.write("**Concept description**")
    st.sidebar.caption(feature_meta.get("description", ""))
    st.sidebar.write(f"Feature index: `{feature_idx}`")
    st.sidebar.write(f"Hook point: `{hook_point}`")

    intro_card = st.container(border=True)
    with intro_card:
        st.caption(
            f"Steer sparse autoencoder features to see how {model_label} responds in text and token probabilities."
        )
        hero_left, hero_right = st.columns([2, 2])

        with hero_left:
            tab_playbook, tab_evidence = st.tabs(["Playbook", "What you'll see"])
            with tab_playbook:
                st.markdown(
                    "1. Pick a model + feature in the sidebar.\n"
                    "2. Drag α left to suppress or right to amplify.\n"
                    "3. Edit the prompt if you want a different context.\n"
                    "4. Run matched generations to lock randomness (optional)."
                )
            with tab_evidence:
                st.markdown(
                    "- **Text comparison** shows before/after shifts side by side.\n"
                    "- **NLL expander** reveals which tokens gain or lose probability.\n"
                    "- Watch for bars swapping order to confirm targeted steering."
                )

        with hero_right:
            stats_card = st.container(border=True)
            stats_card.markdown("**Quick stats**")
            col_model, col_hook, col_feat = stats_card.columns(3)
            hook_parts = hook_point.split(".") if isinstance(hook_point, str) else [hook_point]
            hook_block = hook_parts[1] if hook_point and len(hook_parts) > 1 else hook_point
            col_model.metric("Model", model_label)
            col_hook.metric("Hook block", hook_block or "?")
            col_feat.metric("Features", len(available_feature_keys))

    tab_play, tab_explain = st.tabs(["Playground", "How it works"])

    with tab_play:
        st.subheader("Current setup")
        status_box = st.container(border=True)
        with status_box:
            st.caption("Run snapshot")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Model", model_label)
            m2.metric("Concept", feature_meta["label"])
            m3.metric("α", f"{coeff:+.1f}")
            m4.metric("Max tokens", str(max_new_tokens))
            m5.metric("Temperature", f"{temperature:.1f}")
            coeff_norm = max(0.0, min(1.0, (coeff + 120.0) / 240.0))
            direction = "Amplifying feature" if coeff > 0 else "Suppressing feature" if coeff < 0 else "Neutral"
            st.progress(
                coeff_norm,
                text=f"{direction} (α from −120 to +120)",
            )
            if manual_seed_enabled:
                st.caption("Matched sampling keeps randomness constant between runs.")
            else:
                st.caption("Random seed unlocked — expect natural sampling variation between runs.")

        st.divider()

        st.subheader("Prompt")
        st.caption("Edit the prompt or keep the default for this concept.")

        prompt = st.text_area(
            label="Prompt",
            value=feature_meta.get("prompt", ""),
            height=140,
            label_visibility="collapsed",
        )

        run_button = st.button(
            "Generate and analyze",
            type="primary",
            width="stretch",
        )

        st.divider()

        if run_button and prompt.strip():
            with st.spinner("Running baseline and steered generations..."):
                sae_feature_count = sae.W_dec.shape[0]
                if feature_idx >= sae_feature_count:
                    st.error(
                        f"Feature index {feature_idx} exceeds SAE size ({sae_feature_count}). "
                        "Update the feature mapping for this model or pick a different feature."
                    )
                    st.stop()

                steering_vector = sae.W_dec[feature_idx]
                if manual_seed_enabled:
                    torch.manual_seed(0)
                baseline = generate_text(
                    model,
                    prompt=prompt,
                    steering_vector=None,
                    coefficient=0.0,
                    hook_point=hook_point,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

                if manual_seed_enabled:
                    torch.manual_seed(0)
                steered = generate_text(
                    model,
                    prompt=prompt,
                    steering_vector=steering_vector,
                    coefficient=coeff,
                    hook_point=hook_point,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                st.markdown("### Text comparison")
                col_b, col_s = st.columns(2, gap="large")

                with col_b:
                    base_box = st.container(border=True)
                    base_box.markdown("**Baseline (α = 0)**")
                    base_box.write(baseline)

                with col_s:
                    steer_box = st.container(border=True)
                    steer_box.markdown(f"**Steered (α = {coeff:.1f})**")
                    steer_box.write(steered)

                with st.expander(
                    "Per-token next-token NLL over the generation", expanded=False
                ):
                    st.caption(
                        "For each generated token (sampled from the baseline run), "
                        "we look at the next-token distribution just before it was chosen, "
                        "under both baseline and steered models."
                    )
                    progress_bar = st.progress(
                        0.0, text="Collecting per-token NLL..."
                    )

                    def progress_cb(done, total):
                        progress_bar.progress(
                            done / total,
                            text=f"Collecting per-token NLL... {done}/{total} steps",
                        )

                    tokens_ids, base_steps, steer_steps = generate_stepwise_dists_same_prefix(
                        model=model,
                        prompt=prompt,
                        steering_vector=steering_vector,
                        coeff=coeff,
                        hook_point=hook_point,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=0.9,
                        progress_callback=progress_cb,
                    )
    
                    progress_bar.progress(1.0, text="Per-token NLL collection done!")
                    progress_bar.empty()

                    prompt_len = model.to_tokens(prompt).shape[1]
                    gen_token_ids = tokens_ids[prompt_len:]
                    num_steps = len(gen_token_ids)

                    charts_per_row = 4 
                    for start in range(0, num_steps, charts_per_row):
                        cols = st.columns(charts_per_row, gap="medium")
                        for col_i in range(charts_per_row):
                            step = start + col_i
                            if step >= num_steps:
                                break

                            token_id = gen_token_ids[step]
                            token_str = model.tokenizer.decode([token_id]) or "<unk>"

                            with cols[col_i]:
                                st.markdown(f"**Step {step+1}: `{token_str}`**")

                                df_step = build_step_nll_df(
                                    model=model,
                                    base_logprobs=base_steps[step],
                                    steered_logprobs=steer_steps[step],
                                    top_k=top_k,
                                )

                                chart = (
                                    alt.Chart(df_step)
                                    .mark_bar()
                                    .encode(
                                        x=alt.X(
                                            "NLL:Q",
                                            title="Negative log-probability",
                                            # scale=alt.Scale(reverse=True),
                                        ),
                                        y=alt.Y(
                                            "Token:N",
                                            sort="-x",
                                            title="Token",
                                        ),
                                        color=alt.Color(
                                            "Run:N",
                                            title="Run",
                                        ),
                                        tooltip=["Token", "Run", "NLL"],
                                    )
                                    .properties(
                                        height=max(24 * top_k, 260),
                                    )
                                )
                                st.altair_chart(chart, width="stretch")
        elif not prompt.strip():
            st.info("Please enter a prompt first, then click Generate and analyze.")
        else:
            st.info("Run a comparison to see baseline vs steered text and NLL charts.")

    with tab_explain:
        st.subheader("What this dashboard does")
        st.markdown(
            """
- We load **{model}** and a **sparse autoencoder (SAE)** attached to `{hook_point}`.
- Each SAE feature has a decoder vector, which is a direction in the model's hidden space.
- When you move the slider, we add `α · decoder_vector` to the residual stream at that layer.
- Positive **α** pushes the model toward that concept; negative **α** pushes it away.
            """
        )
        st.markdown(
            """
### How to read the per-token charts

- The baseline run samples tokens step by step.
- For each step, we look at the next-token distribution under:
  - the **baseline** model (no steering),
  - the **steered** model (with your chosen α).
- Each small chart shows negative log-probability (NLL) for the top candidate tokens:
  - Bars further right (lower NLL) are **more likely**.
  - You can scan across tokens to see where steering makes some words more or less likely.
            """
        )


if __name__ == "__main__":
    main()