"""Streamlit front-end for steering GPT-2 features via sparse autoencoders."""

import time
from typing import Dict, Generator, Tuple

import streamlit as st
from llm_backend import generate_baseline, generate_steered

DEFAULT_PROMPT = (
    "Write a quick stage exchange in a Shakespeare style where two cousins argue "
    "about who must carry the lantern."
)
TYPEWRITER_DELAY = 0.02

PROMPT_PRESETS: Dict[str, str] = {
    "Shakespeare Mini Scene": DEFAULT_PROMPT,
    "Flavor Snapshot": (
        "Describe a spoonful of warm soup using simple taste and smell words."
    ),
    "Math Walkthrough": (
        "Explain how to divide 12 apples among 3 kids using neat, step by step logic."
    ),
    "Mixed Bag": (
        "Write a note that starts like a play, shifts into food talk, and ends with a small math tip."
    ),
}

FEATURE_DETAILS = {
    "shakespeare": {
        "label": "Shakespeare",
        "hint": "Playwright vibe",
        "long": "Adds stage cues, poetic lines, and dramatic beats.",
    },
    "flavors": {
        "label": "Flavors",
        "hint": "Flavor talk",
        "long": "Talks about taste, smell, and texture like a food friend.",
    },
    "math": {
        "label": "Math",
        "hint": "Logic voice",
        "long": "Brings in steps, numbers, and tidy reasoning words.",
    },
}


# -------------------------
# Helpers
# -------------------------
def split_prompt_and_completion(prompt: str, full_text: str) -> Tuple[str, str]:
    """
    Split off the generated continuation when the model echoes the prompt.

    Args:
        prompt: Text supplied by the user.
        full_text: Model output that may contain the prompt prefix.

    Returns:
        tuple[str, str]: The original prompt and the trimmed completion text.
    """
    if not full_text:
        return prompt, ""

    prompt_clean = prompt.lstrip()
    full_clean = full_text.lstrip()

    if prompt_clean and full_clean.startswith(prompt_clean):
        completion = full_clean[len(prompt_clean):].lstrip("\n ")
    else:
        completion = full_clean

    return prompt, completion


def word_stream(text: str, delay: float = 0.02) -> Generator[str, None, None]:
    """
    Yield words sequentially to create a gentle typewriter animation.

    Args:
        text: Full string that should be streamed in the UI.
        delay: Pause in seconds between yielded words.

    Returns:
        Generator[str, None, None]: Words with trailing spaces ready for Streamlit.
    """
    for word in text.split(" "):
        yield word + " "
        if delay > 0:
            time.sleep(delay)


def render_feature_cards(alpha_map: Dict[str, float]) -> None:
    """
    Display small metric cards summarizing the current alpha per feature.

    Args:
        alpha_map: Dictionary mapping feature keys to slider values.

    Returns:
        None
    """
    cols = st.columns(len(alpha_map))
    for col, (key, value) in zip(cols, alpha_map.items()):
        meta = FEATURE_DETAILS.get(key, {"label": key.title(), "hint": ""})
        col.metric(label=f"{meta['label']} α", value=f"{value:+.1f}", delta=meta.get("hint", ""))


def format_alpha_caption(alpha_map: Dict[str, float]) -> str:
    """
    Build a concise caption string describing the active alphas.

    Args:
        alpha_map: Dictionary mapping feature keys to slider values.

    Returns:
        str: Comma-separated description of each feature and its alpha.
    """
    return ", ".join(
        f"{FEATURE_DETAILS.get(key, {'label': key.title()})['label']}: {value:+.1f}"
        for key, value in alpha_map.items()
    )


def load_selected_preset() -> None:
    """
    Copy the currently selected preset into the prompt text area.

    Returns:
        None
    """
    st.session_state.prompt_text = PROMPT_PRESETS[st.session_state.selected_preset]


def reset_prompt_to_default() -> None:
    """
    Restore the default prompt so the user can restart their experiment quickly.

    Returns:
        None
    """
    st.session_state.prompt_text = DEFAULT_PROMPT


# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="SAE Steering Playground",
    page_icon="🧠",
    layout="wide",
)

if "prompt_text" not in st.session_state:
    st.session_state.prompt_text = DEFAULT_PROMPT

if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = list(PROMPT_PRESETS.keys())[0]

st.title("SAE Steering Playground")
st.caption("See the base run next to a lightly steered run.")

hero_left, hero_right = st.columns([3, 2], gap="large")
with hero_left:
    st.markdown("### Tinker with small nudges")
    st.markdown(
        "- Type any prompt and watch how the base model replies.\n"
        "- Nudge one feature at a time to feel how the mood shifts.\n"
        "- Save the mixes that give you the voice you like."
    )
with hero_right:
    st.markdown("### Quick workflow")
    st.markdown(
        "1. Pick or type a prompt.\n"
        "2. Set the sampler knobs in the sidebar.\n"
        "3. Move one slider, press **Generate**, compare the two runs."
    )

st.divider()


# -------------------------
# Sidebar: generation settings
# -------------------------
with st.sidebar:
    st.header("Generation settings")
    st.caption("Pick how long and how wild the sampler should be based on temperature and top p.")

    max_tokens = st.slider(
        "Max new tokens",
        min_value=16,
        max_value=512,
        value=128,
        step=16,
        help="How many extra tokens to add.",
    )

    with st.expander("Advanced sampling", expanded=False):
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.5,
            value=0.7,
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


# -------------------------
# Tabs: Playground / How it works
# -------------------------
play_tab, explain_tab = st.tabs(["Playground", "How it works"])


# =========================
# PLAYGROUND TAB
# =========================
with play_tab:
    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        st.subheader("Prompt")
        st.caption("Pick a preset or paste your own words.")

        prompt = st.text_area(
            label="Prompt",
            key="prompt_text",
            height=220,
            placeholder="Type any setup you want the model to continue...",
            label_visibility="collapsed",
        )

        preset_col, action_col = st.columns([3, 1])
        with preset_col:
            st.selectbox(
                "Prompt presets",
                options=list(PROMPT_PRESETS.keys()),
                key="selected_preset",
                help="Swap in a ready-to-use sample prompt.",
                on_change=load_selected_preset,
            )
        with action_col:
            st.button(
                "Reset prompt",
                use_container_width=True,
                help="Bring back the starter prompt.",
                on_click=reset_prompt_to_default,
            )

        run_button = st.button(
            "Generate",
            type="primary",
            use_container_width=True,
            help="Run the baseline and steered generations side by side.",
        )

    with col_right:
        st.subheader("Steering sliders")
        st.caption("Each alpha nudges one SAE feature.")

        shakespeare_alpha = st.slider(
            "Shakespeare",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            help="Negative = less stage drama. Positive = more stage drama.",
        )

        flavors_alpha = st.slider(
            "Flavors",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            help="Boost or mute sensory chatter.",
        )

        math_alpha = st.slider(
            "Math",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            help="Tilt toward step-by-step logic or away from it.",
        )

        alpha_snapshot = {
            "shakespeare": shakespeare_alpha,
            "flavors": flavors_alpha,
            "math": math_alpha,
        }
        render_feature_cards(alpha_snapshot)

    st.divider()

    st.subheader("Output comparison")
    st.caption("Left = base run. Right = steered run.")

    metrics_container = st.container(border=True)
    with metrics_container:
        summary_cols = st.columns(3)
        summary_cols[0].metric("Max tokens", max_tokens)
        summary_cols[1].metric("Temperature", f"{temperature:.1f}")
        summary_cols[2].metric("Top p", f"{top_p:.2f}")

    col_base, col_steered = st.columns(2, gap="large")

    with col_base:
        st.markdown("**Baseline (no steering)**")
        base_container = st.container(border=True)
        with base_container:
            baseline_placeholder = st.empty()

    with col_steered:
        st.markdown("**Steered (with SAE)**")
        steered_container = st.container(border=True)
        with steered_container:
            steered_placeholder = st.empty()

    if run_button:
        if not prompt.strip():
            baseline_placeholder.info("Please enter a prompt and click **Generate**.")
            steered_placeholder.info("Please enter a prompt and click **Generate**.")
        else:
            with st.spinner("Writing both versions..."):
                baseline_full = generate_baseline(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                _, base_completion = split_prompt_and_completion(prompt, baseline_full)

                steered_full = generate_steered(
                    prompt=prompt,
                    alphas=alpha_snapshot,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                _, steered_completion = split_prompt_and_completion(prompt, steered_full)

            base_text = base_completion or "[Empty completion]"
            steered_text = steered_completion or "[Empty completion]"

            with baseline_placeholder:
                st.write_stream(word_stream(base_text, delay=TYPEWRITER_DELAY))

            with steered_placeholder:
                st.caption(f"Alphas → {format_alpha_caption(alpha_snapshot)}")
                st.write_stream(word_stream(steered_text, delay=TYPEWRITER_DELAY))

            st.success("All set!")
    else:
        baseline_placeholder.info("Baseline text will show up here after you click **Generate**.")
        steered_placeholder.info("Steered text will show up here after you click **Generate**.")


# =========================
# EXPLANATION TAB
# =========================
with explain_tab:
    st.subheader("What is going on?")

    explain_cols = st.columns([2, 1], gap="large")
    with explain_cols[0]:
        st.markdown(
            "- **Baseline**: the plain model keeps writing after your prompt.\n"
            "- **Steered**: we grab a hidden vector, encode it with an SAE, bump a few features, decode, and keep writing.\n"
            "- Comparing the two shows how one hidden direction changes tone or content."
        )
        st.markdown(
            "**How to explore:**\n"
            "1. Pick a prompt.\n"
            "2. Move one slider slowly and reread the right column.\n"
            "3. Mix small positive and negative values to stack effects."
        )

    with explain_cols[1]:
        st.info("Keep alphas gentle. Huge values can drown the story or add noise.")

    with st.expander("Why sparse autoencoders help?", expanded=True):
        st.markdown(
            "SAEs keep most features off, so the few that light up are easy to point at. When we nudge a feature, we know which idea we touched, and the rest of the model stays steady."
        )
