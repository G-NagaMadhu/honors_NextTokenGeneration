"""Streamlit GUI for the trained MiniGPT Shakespeare generator."""

import html
import sys
from pathlib import Path

import streamlit as st
import torch

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train import MiniGPT  # noqa: E402


st.set_page_config(
    page_title="MiniGPT Shakespeare Lab",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600;700&display=swap');
    .stApp { background: #f4efdf; color: #101417; }
    [data-testid="stSidebar"] { background: #101417; }
    [data-testid="stSidebar"] * { color: #f4efdf !important; }
    h1, h2, h3 { font-family: 'Cormorant Garamond', Georgia, serif !important; color: #101417; }
    .hero { padding: 1.6rem 0 1rem; border-bottom: 1px solid rgba(16,20,23,.16); }
    .eyebrow { color: #ae5e32; font: 700 .72rem 'IBM Plex Mono', monospace; letter-spacing: .14em; }
    .hero h1 { font-size: clamp(3rem, 7vw, 5.8rem); line-height: .85; margin: .7rem 0 1rem; letter-spacing: -.05em; }
    .hero h1 em { color: #ae5e32; }
    .lede { color: #52615d; font: 1rem/1.6 'IBM Plex Sans', sans-serif; max-width: 680px; }
    .output-box { background: #fffdf7; color: #101417 !important; border: 1px solid rgba(16,20,23,.18); border-top: 4px solid #d98752; padding: 2rem; box-shadow: 0 18px 45px rgba(11,17,18,.12); white-space: pre-wrap; font: 1.35rem/1.55 'Cormorant Garamond', Georgia, serif; min-height: 280px; }
    .metric { background: #fffdf7; border: 1px solid rgba(16,20,23,.12); padding: .8rem; }
    .metric-label { color: #70817c; font: .65rem 'IBM Plex Mono', monospace; letter-spacing: .08em; }
    .metric-value { color: #101417; font: 1.2rem 'IBM Plex Mono', monospace; margin-top: .25rem; }
    .notice { background: rgba(155,182,162,.2); border: 1px solid rgba(101,131,110,.35); padding: 1rem; color: #405048; font: .85rem/1.5 'IBM Plex Sans', sans-serif; }
    .stTextArea textarea { background: #fffdf7 !important; color: #101417 !important; caret-color: #ae5e32 !important; border: 1px solid rgba(16,20,23,.25) !important; }
    .stTextArea textarea::placeholder { color: #70817c !important; opacity: 1 !important; }
    .stTextInput input, .stNumberInput input { color: #101417 !important; background: #fffdf7 !important; }
    .stButton > button, .stDownloadButton > button { background: #101417 !important; color: #ffffff !important; border: 0; border-radius: 0; min-height: 3rem; font-family: 'IBM Plex Mono', monospace; }
    .stButton > button p, .stDownloadButton > button p, .stButton > button span, .stDownloadButton > button span { color: #ffffff !important; }
    .stButton > button:hover, .stDownloadButton > button:hover { background: #ae5e32 !important; color: #ffffff !important; }
    .stDownloadButton > button { margin-top: .75rem; }
    code, .mono { font-family: 'IBM Plex Mono', monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained MiniGPT checkpoint…")
def load_model(checkpoint_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    class Config:
        pass

    config = Config()
    for key, value in checkpoint["config"].items():
        setattr(config, key, value)

    model = MiniGPT(checkpoint["vocab_size"], config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    stoi = checkpoint["stoi"]
    itos = {int(key): value for key, value in checkpoint["itos"].items()}
    return model, config, stoi, itos, device


def generate_text(model, prompt, stoi, itos, device, config, tokens, temperature, top_k, top_p, seed):
    unknown = sorted(set(char for char in prompt if char not in stoi))
    if unknown:
        raise ValueError(f"Prompt contains characters outside the vocabulary: {unknown}")
    if not prompt:
        raise ValueError("Please enter a prompt.")

    ids = [stoi[char] for char in prompt][-config.block_size :]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    context = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(
            context,
            max_new_tokens=tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p if top_p is not None else None,
        )
    return "".join(itos[int(token)] for token in output[0].tolist()), len(ids)


st.markdown(
    '<div class="hero"><div class="eyebrow">MINIGPT / SHAKESPEARE LAB</div><h1>Let the model<br><em>choose what comes next.</em></h1><p class="lede">A small decoder-only Transformer trained on Shakespeare corpus. Enter a prompt, tune the sampling rules, and watch it continue one character at a time.</p></div>',
    unsafe_allow_html=True,
)

checkpoint_path = st.sidebar.text_input(
    "Checkpoint path",
    value="checkpoints/shakespeare_char.pt",
    help="Run Streamlit from your project root, or enter an absolute path.",
)

try:
    model, config, stoi, itos, device = load_model(checkpoint_path)
except Exception as error:
    st.error(f"Could not load the checkpoint: {error}")
    st.info("Run Streamlit from the mini_chatbot project folder, or enter the full checkpoint path in the sidebar.")
    st.stop()

with st.sidebar:
    st.markdown("### Model baseline")
    st.caption(f"Device: {device.upper()}")
    st.caption(f"Vocabulary: {len(stoi)} characters")
    st.caption(f"Layers: {config.n_layer}")
    st.caption(f"Embedding dimension: {config.n_embd}")
    st.caption(f"Attention heads: {config.n_head}")
    st.caption(f"Context length: {config.block_size}")
    st.markdown("---")
    st.caption("This is an educational next-character model. It imitates learned text patterns and is not a factual chatbot.")

left, right = st.columns([1.45, 1], gap="large")
with left:
    st.markdown('<div class="eyebrow">01 / PROMPT</div>', unsafe_allow_html=True)
    prompt = st.text_area("Prompt", value=st.session_state.get("prompt_value", "ROMEO:"), height=150, label_visibility="collapsed")
    st.session_state["prompt_value"] = prompt
    st.caption("Prompt: the starting text the Transformer uses to predict the next characters.")
    quick_prompt = st.selectbox("Quick prompt", ["ROMEO:", "JULIET:", "First Citizen:", "To be, or not to be"], label_visibility="collapsed")
    if st.button("Use selected quick prompt"):
        st.session_state["prompt_value"] = quick_prompt
        st.rerun()

with right:
    st.markdown('<div class="eyebrow">02 / SAMPLING CONTROLS</div>', unsafe_allow_html=True)
    tokens = st.slider("New characters", 20, 900, 300, 10)
    st.caption("New characters: the maximum number of characters generated after the prompt.")
    temperature = st.slider("Temperature", 0.2, 1.6, 0.8, 0.1)
    st.caption("Temperature: lower values make output safer and more predictable; higher values make it more random.")
    top_k = st.slider("Top-k", 0, len(stoi), min(20, len(stoi)), 1)
    st.caption("Top-k: limits the next-character choice to the k most likely characters. 0 disables this filter.")
    use_top_p = st.checkbox("Enable top-p sampling")
    top_p = st.slider("Top-p", 0.5, 1.0, 0.9, 0.01, disabled=not use_top_p) if use_top_p else None
    st.caption("Top-p: keeps the smallest group of likely characters whose combined probability reaches p.")
    seed = st.number_input("Random seed", min_value=0, value=1337, step=1)
    st.caption("Random seed: the same seed and settings can reproduce the same sampled output.")

st.markdown('<div class="notice">This interface uses your trained checkpoint directly. Temperature, top-k, and top-p affect sampling during generation; they do not retrain the Transformer.</div>', unsafe_allow_html=True)
st.markdown(
    f"**Current generation settings:** `{tokens}` new characters · "
    f"temperature `{temperature:.1f}` · top-k `{top_k if top_k else 'off'}` · "
    f"top-p `{top_p if top_p is not None else 'off'}` · seed `{int(seed)}`",
)

if "generated_text" not in st.session_state:
    st.session_state["generated_text"] = ""

if st.button("Generate continuation", type="primary", use_container_width=True):
    try:
        text, used_context = generate_text(
            model, prompt, stoi, itos, device, config,
            tokens, temperature, top_k, top_p, int(seed),
        )
        st.session_state["generated_text"] = text
        st.session_state["used_context"] = used_context
    except Exception as error:
        st.error(str(error))

st.markdown('<div class="eyebrow" style="margin-top:2rem">03 / GENERATED PASSAGE</div>', unsafe_allow_html=True)
if st.session_state.get("generated_text"):
    safe_text = html.escape(st.session_state["generated_text"])
    st.markdown(f'<div class="output-box">{safe_text}</div>', unsafe_allow_html=True)
    st.caption(
        f"Context used: {st.session_state.get('used_context', 0)} characters · "
        f"Generated: {tokens} new characters · Temperature: {temperature:.1f} · "
        f"Top-k: {top_k if top_k else 'off'} · Top-p: {top_p if top_p is not None else 'off'}"
    )
    st.download_button(
        "Download generated text",
        st.session_state["generated_text"],
        file_name="minigpt_generated.txt",
        mime="text/plain",
    )
else:
    st.markdown('<div class="output-box" style="color:#9ca49d">Your generated passage will appear here.</div>', unsafe_allow_html=True)

st.markdown('<div class="eyebrow" style="margin-top:2rem">04 / MODEL DETAILS</div>', unsafe_allow_html=True)
metrics = st.columns(5)
for column, label, value in zip(
    metrics,
    ["LAYERS", "EMBEDDING", "HEADS", "CONTEXT", "VOCABULARY"],
    [config.n_layer, config.n_embd, config.n_head, config.block_size, len(stoi)],
):
    column.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
