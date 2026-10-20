"""Gradio demo for the TinyStories JAX GPT (~20M params)."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import jax
import jax.numpy as jnp
import numpy as np
import spaces
import tiktoken
from flax import nnx
from jax.tree_util import tree_flatten_with_path, tree_unflatten

from model import GPT

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
WEIGHTS_PATH = ROOT / "weights.npz"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def path_to_key(path) -> str:
    parts: list[str] = []
    for p in path:
        if hasattr(p, "key"):
            parts.append(str(p.key))
        elif hasattr(p, "idx"):
            parts.append(str(p.idx))
        else:
            parts.append(str(p))
    key = ".".join(parts)
    if key.endswith(".value"):
        key = key[: -len(".value")]
    return key.replace("..", ".")


def build_model(cfg: dict) -> GPT:
    return GPT(
        maxlen=cfg["maxlen"],
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        feed_forward_dim=cfg["feed_forward_dim"],
        num_transformer_blocks=cfg["num_transformer_blocks"],
        rngs=nnx.Rngs(0),
    )


def load_weights(model: GPT, weights_path: Path) -> GPT:
    if not weights_path.exists():
        raise FileNotFoundError(
            f"weights.npz not found at {weights_path}. "
            "Run export_weights.py before deploying."
        )

    arrays = np.load(weights_path)
    params = nnx.state(model, nnx.Param)
    leaves_with_path, treedef = tree_flatten_with_path(params)

    new_leaves = []
    missing = []
    for path, leaf in leaves_with_path:
        key = path_to_key(path)
        if key not in arrays:
            missing.append(key)
            new_leaves.append(leaf)
            continue
        value = jnp.asarray(arrays[key])
        if hasattr(leaf, "value"):
            leaf.value = value
            new_leaves.append(leaf)
        else:
            new_leaves.append(value)

    if missing:
        raise KeyError(f"Missing {len(missing)} weight keys, e.g. {missing[:3]}")

    restored = tree_unflatten(treedef, new_leaves)
    nnx.update(model, restored)
    return model


print("Loading config and tokenizer...")
CFG = load_config()
TOKENIZER = tiktoken.get_encoding(CFG["tokenizer"])
END_TOKEN = TOKENIZER.encode(
    "<|endoftext|>",
    allowed_special={"<|endoftext|>"},
)[0]

print("Building model and loading weights (CPU)...")
MODEL = load_weights(build_model(CFG), WEIGHTS_PATH)
print("Model ready.")


@spaces.GPU(duration=60)
def create_story(
    prompt: str,
    temperature: float = 0.8,
    max_new_tokens: int = 50,
    seed: int = 0,
) -> str:
    """Autoregressive story generation from a text prompt.

    Decorated for ZeroGPU Spaces; inference still runs on JAX CPU.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return "Please enter a story prompt."

    temperature = max(float(temperature), 1e-5)
    max_new_tokens = int(max_new_tokens)
    tokens = TOKENIZER.encode(prompt)
    key = jax.random.key(int(seed))

    for _ in range(max_new_tokens):
        context = tokens[-MODEL.maxlen :]
        x = jnp.asarray(context, dtype=jnp.int32)[None, :]
        logits = MODEL(x)[0, -1, :] / temperature
        key, subkey = jax.random.split(key)
        next_token = int(jax.random.categorical(subkey, logits))
        tokens.append(next_token)
        if next_token == END_TOKEN:
            break

    return TOKENIZER.decode(tokens)


EXAMPLES = [
    ["Once upon a time a big bear ", 0.2, 40, 0],
    ["Lily found a magic key in the garden. ", 0.7, 60, 1],
    ["There was a tiny robot who wanted to be a chef. ", 0.9, 50, 2],
]

demo = gr.Interface(
    fn=create_story,
    inputs=[
        gr.Textbox(
            label="Story prompt",
            placeholder="Once upon a time...",
            lines=2,
        ),
        gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature"),
        gr.Slider(10, 128, value=50, step=1, label="Max new tokens"),
        gr.Number(value=0, precision=0, label="Seed"),
    ],
    outputs=gr.Textbox(label="Generated story", lines=8),
    title="TinyStories GPT — JAX / Flax NNX",
    description=(
        "A ~20M-parameter decoder-only transformer trained on TinyStories, "
        "built from scratch with JAX and Flax NNX. "
        "Source: [llm-with-Jax](https://github.com/JosselinPerret/llm-with-Jax)."
    ),
    examples=EXAMPLES,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
