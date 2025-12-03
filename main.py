import torch
import altair as alt
import streamlit as st
from llm_steering import *


@st.cache_resource
def get_model_and_sae():
    """Cache the heavy model + SAE load for the Streamlit session."""
    return load_model_and_sae()


# -----------------------------------
# Streamlit UI
# -----------------------------------
def main():
    st.set_page_config(
        page_title="SAE Steering Playground - GPT-2 Small",
        layout="wide",
    )

    st.title("SAE Steering Playground for GPT-2 Small")

    intro_card = st.container(border=True)
    with intro_card:
        st.caption(
            "Steer sparse autoencoder features to see how GPT-2 Small's text and "
            "token probabilities react in real time."
        )
        hero_left, hero_right = st.columns([3, 2])

        with hero_left:
            tab_playbook, tab_evidence = st.tabs(["Playbook", "What you'll see"])
            with tab_playbook:
                st.markdown(
                    "1. Pick a concept feature in the sidebar.\n"
                    "2. Drag α left to suppress or right to amplify.\n"
                    "3. Edit the prompt if you want a different context.\n"
                    "4. Run matched generations to lock randomness."
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
            stat_col1, stat_col2 = stats_card.columns(2)
            stat_col1.metric("Features", len(FEATURES))
            hook_parts = HOOK_POINT.split(".")
            hook_block = hook_parts[1] if len(hook_parts) > 1 else HOOK_POINT
            stat_col2.metric("Hook block", hook_block)
            stats_card.markdown("**Session checklist**")
            stats_card.markdown(
                "- Keep α small (±20) for subtle edits.\n"
                "- Increase max tokens to surface long-form effects.\n"
                "- Use the charts to justify any qualitative claims."
            )

    st.divider()

    # Load heavy stuff once
    model, sae, device = get_model_and_sae()
    st.sidebar.success(f"Model loaded on **{device.upper()}**")

    # Sidebar controls
    st.sidebar.header("Controls")

    concept_name = st.sidebar.selectbox(
        "Concept feature",
        options=list(FEATURES.keys()),
        format_func=lambda k: k,
    )
    concept = FEATURES[concept_name]

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
        max_value=100,  # larger range
        value=20,
        step=1,
    )

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=3.0,   # wider range
        value=1.0,
        step=0.1,
    )

    top_k = st.sidebar.slider(
        "Tokens in each NLL chart (top-k)",
        min_value=5,
        max_value=20,   # up to 20 as you wanted
        value=10,
        step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.write("**Concept description**")
    st.sidebar.caption(concept["description"])
    st.sidebar.write(f"Feature index: `{concept['index']}`")
    st.sidebar.write(f"Hook point: `{HOOK_POINT}`")

    tab_play, tab_explain = st.tabs(["Playground", "How it works"])

    with tab_play:
        st.subheader("Current setup")
        status_box = st.container(border=True)
        with status_box:
            st.caption("Run snapshot")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Concept", concept_name)
            m2.metric("α", f"{coeff:+.1f}")
            m3.metric("Max tokens", str(max_new_tokens))
            m4.metric("Temperature", f"{temperature:.1f}")
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
            value=concept["prompt"],
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
                feature_idx = concept["index"]
                steering_vector = sae.W_dec[feature_idx]

                # Fix seed so differences in text are due to steering, not randomness
                if manual_seed_enabled:
                    torch.manual_seed(0)
                baseline = generate_text(
                    model,
                    prompt=prompt,
                    steering_vector=None,
                    coefficient=0.0,
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
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

                # ----------------------
                # Text comparison (2 cols)
                # ----------------------
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

                # ----------------------
                # Per-token NLL charts in an expander
                # ----------------------
                with st.expander(
                    "Per-token next-token NLL over the generation", expanded=False
                ):
                    st.caption(
                        "For each generated token (sampled from the baseline run), "
                        "we look at the next-token distribution just before it was chosen, "
                        "under both baseline and steered models."
                    )

                    # Streamlit progress bar for per-token loop
                    progress_bar = st.progress(
                        0.0, text="Collecting per-token NLL..."
                    )

                    def progress_cb(done, total):
                        progress_bar.progress(
                            done / total,
                            text=f"Collecting per-token NLL... {done}/{total} steps",
                        )

                    # Collect stepwise distributions on a shared prefix
                    tokens_ids, base_steps, steer_steps = generate_stepwise_dists_same_prefix(
                        model=model,
                        prompt=prompt,
                        steering_vector=steering_vector,
                        coeff=coeff,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=0.9,
                        progress_callback=progress_cb,
                    )

                    # Finished
                    progress_bar.progress(1.0, text="Per-token NLL collection done!")
                    progress_bar.empty()

                    # Separate prompt tokens vs generated tokens
                    prompt_len = model.to_tokens(prompt).shape[1]
                    gen_token_ids = tokens_ids[prompt_len:]
                    num_steps = len(gen_token_ids)

                    charts_per_row = 4  # fewer per row, each larger

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
                                            # uncomment if you want right-to-left
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
- We load **GPT-2 small** and a **sparse autoencoder (SAE)** trained on layer 5.
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