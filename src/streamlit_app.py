from __future__ import annotations

import os
import time
from typing import Dict, Generator, Tuple

import streamlit as st
from huggingface_hub import login

from config import FEATURE_DETAILS, TYPEWRITER_DELAY
from generation import AVAILABLE_MODELS, DEFAULT_MODEL_KEY, get_device, run_comparison


def maybe_login_to_hub() -> None:
    """Log into Hugging Face when ``HUGGINGFACE_TOKEN`` is set.

    Args:
        None

    Returns:
        None
    """

    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        login(token=token)


def word_stream(text: str, delay: float = TYPEWRITER_DELAY) -> Generator[str, None, None]:
    """Yield one word at a time so Streamlit can animate the response.

    Args:
        text: Full string returned by the model.
        delay: Pause between words to mimic a typewriter.

    Returns:
        Generator that yields individual words with a trailing space.
    """

    for word in text.split(" "):
        yield word + " "
        if delay > 0:
            time.sleep(delay)


def format_alpha_caption(alpha_map: Dict[str, float]) -> str:
    """Format slider selections into a caption for the steered panel.

    Args:
        alpha_map: Feature keys mapped to the chosen slider values.

    Returns:
        Human friendly summary that highlights each feature change.
    """

    return ", ".join(
        f"{FEATURE_DETAILS.get(key, {'label': key.title()})['label']}: {value:+.1f}"
        for key, value in alpha_map.items()
    )


def render_sidebar() -> Tuple[str, int, float, float]:
    """Collect model sampling controls from the sidebar.

    Args:
        None

    Returns:
        Tuple containing model key, max tokens, temperature, and top-p.
    """

    with st.sidebar:
        st.header("Generation settings")
        st.caption("Pick how long and playful the writing should be.")
        model_key = st.selectbox(
            "Base model",
            options=list(AVAILABLE_MODELS.keys()),
            index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL_KEY),
            format_func=lambda key: AVAILABLE_MODELS[key],
        )
        st.info(
            f"Running {AVAILABLE_MODELS[model_key]} on {get_device().upper()}"
        )

        max_tokens = st.slider(
            "Max new tokens",
            min_value=16,
            max_value=512,
            value=128,
            step=2,
            help="How many extra tokens to add.",
        )

        with st.expander("Advanced sampling", expanded=False):
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="Higher = wilder wording. Lower = steadier wording.",
            )

            top_p = st.slider(
                "Top p",
                min_value=0.1,
                max_value=1.0,
                value=0.9,
                step=0.05,
                help="Only grab from the most likely words.",
            )

        st.divider()
    return model_key, max_tokens, temperature, top_p


def render_prompt_panel() -> Tuple[str, bool]:
    """Render the prompt input alongside the run button.

    Args:
        None

    Returns:
        Tuple of the current prompt text and a boolean indicating a click.
    """

    st.subheader("Prompt")
    st.caption("Type the start of your story or explanation here.")

    prompt = st.text_area(
        label="Prompt",
        key="prompt_text",
        height=220,
        placeholder="Start with a sentence, like ‘Once upon a time…’",
        label_visibility="collapsed",
    )

    run_button = st.button(
        "Generate",
        type="primary",
        use_container_width=True,
        help="Run the baseline and steered generations side by side.",
    )
    return prompt, run_button


def render_steering_panel() -> Dict[str, float]:
    """Render slider controls and quick metrics for each interpretable feature.

    Args:
        None

    Returns:
        Dictionary mapping feature keys to the chosen slider values.
    """

    st.subheader("Style sliders")
    st.caption("Move a slider to add or remove that vibe.")

    sliders: Dict[str, float] = {}
    for key in FEATURE_DETAILS:
        meta = FEATURE_DETAILS[key]
        sliders[key] = st.slider(
            meta["label"],
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            help=meta.get("long", meta.get("hint", "")),
        )

    cols = st.columns(len(sliders))
    for col, (key, value) in zip(cols, sliders.items()):
        meta = FEATURE_DETAILS.get(key, {"label": key.title(), "hint": ""})
        col.metric(
            label=f"{meta['label']} Steer",
            value=f"{value:+.1f}",
            delta=meta.get("hint", ""),
        )
    return sliders


def render_output_comparison(
    model_key: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    alpha_snapshot: Dict[str, float],
    prompt: str,
    run_button_clicked: bool,
) -> None:
    """Stream the baseline and steered generations side by side.

    Args:
        model_key: Which model configuration to use.
        max_tokens: Number of new tokens to sample.
        temperature: Sampling temperature for creativity.
        top_p: Nucleus sampling parameter.
        alpha_snapshot: Slider values captured at run time.
        prompt: User supplied beginning of the text.
        run_button_clicked: Whether the user triggered a comparison run.

    Returns:
        None
    """

    st.subheader("Output comparison")
    st.caption("Left shows the plain run. Right shows the slider mix.")

    metrics_container = st.container(border=True)
    with metrics_container:
        summary_cols = st.columns(4)
        summary_cols[0].metric("Model", AVAILABLE_MODELS[model_key])
        summary_cols[1].metric("Max tokens", max_tokens)
        summary_cols[2].metric("Temperature", f"{temperature:.1f}")
        summary_cols[3].metric("Top p", f"{top_p:.2f}")

    col_base, col_steered = st.columns(2, gap="large")
    with col_base:
        baseline_panel = st.container(border=True)
        baseline_panel.markdown("**Baseline (no steering)**")
        baseline_placeholder = baseline_panel.empty()
    with col_steered:
        steered_panel = st.container(border=True)
        steered_panel.markdown("**Steered (with SAE nudges)**")
        steered_placeholder = steered_panel.empty()

    if run_button_clicked:
        with st.spinner("Writing both versions..."):
            result = run_comparison(
                prompt=prompt,
                alphas=alpha_snapshot,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                model_key=model_key,
            )

        with baseline_placeholder:
            st.write_stream(word_stream(result["baseline"], delay=TYPEWRITER_DELAY))

        with steered_placeholder:
            st.caption(f"Alphas → {format_alpha_caption(alpha_snapshot)}")
            st.write_stream(word_stream(result["steered"], delay=TYPEWRITER_DELAY))

        st.success("Done! Read both columns and pick your favorite tone.")
    else:
        baseline_placeholder.info("The plain writing will show up here once you press Generate.")
        steered_placeholder.info("The slider mix will show up here once you press Generate.")


def render_explanation_tab() -> None:
    """Lay out a friendly explanation of what the playground does.

    Args:
        None

    Returns:
        None
    """

    st.subheader("What is going on?")

    explain_cols = st.columns([2, 1], gap="large")
    with explain_cols[0]:
        st.markdown(
            "- **Baseline**: the model writes text on its own, without any extra changes.\n"
            "- **Steered**: we gently tweak one hidden knob to push the text in a certain direction.\n"
            "- By reading both, you can see what each knob is really doing."
        )
        st.markdown(
            "**How to explore:**\n"
            "1. Type a prompt you care about.\n"
            "2. Move one slider at a time.\n"
            "3. Look at the baseline and steered text and notice how the tone or content changes."
        )

    with st.expander("Why sparse autoencoders help?", expanded=True):
        st.markdown(
            "Inside the model there are many small features, like tiny switches.\n"
            "Sparse autoencoders try to keep most of these switches off, and only turn on a few at a time.\n"
            "This makes it easier to guess what each switch means.\n"
            "When we nudge one feature with a slider, we have a decent idea of the kind of change we should see in the text."
        )


def run_app() -> None:
    """Assemble the full Streamlit experience end to end.

    Args:
        None

    Returns:
        None
    """

    maybe_login_to_hub()

    st.set_page_config(page_title="GPT-2's Thoughts Steering Playground", layout="wide")

    if "prompt_text" not in st.session_state:
        st.session_state.prompt_text = ""
    
    st.title("GPT-2's Thoughts Steering Playground")

    selections = render_sidebar()

    play_tab, explain_tab = st.tabs(["Playground", "How it works"])
    with play_tab:
        col_left, col_right = st.columns([2, 1], gap="large")
        with col_left:
            prompt, run_clicked = render_prompt_panel()
        with col_right:
            alpha_snapshot = render_steering_panel()
        st.divider()
        render_output_comparison(
            model_key=selections[0],
            max_tokens=selections[1],
            temperature=selections[2],
            top_p=selections[3],
            alpha_snapshot=alpha_snapshot,
            prompt=prompt,
            run_button_clicked=run_clicked,
        )
    with explain_tab:
        render_explanation_tab()
