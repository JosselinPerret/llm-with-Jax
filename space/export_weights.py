"""Export Orbax checkpoint (nnx.List model) to portable weights.npz."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import SingleDeviceSharding
from jax.tree_util import tree_flatten_with_path

ROOT = Path(__file__).resolve().parent


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
    key = key.replace("..", ".")
    # Map List indices to named block_i attributes used in Space model.py
    if key.startswith("transformer_blocks."):
        rest = key[len("transformer_blocks.") :]
        idx, _, tail = rest.partition(".")
        key = f"block_{idx}.{tail}" if tail else f"block_{idx}"
    return key


def build_list_model(cfg: dict) -> nnx.Module:
    """Architecture matching the training notebook (uses nnx.List)."""

    class TokenAndPositionEmbedding(nnx.Module):
        def __init__(self, maxlen, vocab_size, embed_dim, *, rngs):
            self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
            self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

        def __call__(self, x):
            seq_len = x.shape[1]
            positions = jnp.arange(seq_len)[None, :]
            return self.token_emb(x) + self.pos_emb(positions)

    class TransformerBlock(nnx.Module):
        def __init__(self, embed_dim, num_heads, ff_dim, *, rngs):
            self.attention = nnx.MultiHeadAttention(
                num_heads=num_heads,
                in_features=embed_dim,
                qkv_features=embed_dim,
                out_features=embed_dim,
                decode=False,
                rngs=rngs,
            )

        def __call__(self, x, mask=None):
            return x + self.attention(x, mask=mask)

    class GPT(nnx.Module):
        def __init__(
            self,
            maxlen,
            vocab_size,
            embed_dim,
            num_heads,
            feed_forward_dim,
            num_transformer_blocks,
            *,
            rngs,
        ):
            self.maxlen = maxlen
            self.embedding = TokenAndPositionEmbedding(
                maxlen, vocab_size, embed_dim, rngs=rngs
            )
            self.transformer_blocks = nnx.List(
                [
                    TransformerBlock(
                        embed_dim, num_heads, feed_forward_dim, rngs=rngs
                    )
                    for _ in range(num_transformer_blocks)
                ]
            )
            self.output_layer = nnx.Linear(
                embed_dim, vocab_size, use_bias=False, rngs=rngs
            )

    import jax.numpy as jnp  # noqa: F401 — used above once classes resolve

    # re-bind jnp for nested classes
    import jax.numpy as jnp

    globals()["jnp"] = jnp
    return GPT(
        maxlen=cfg["maxlen"],
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        feed_forward_dim=cfg["feed_forward_dim"],
        num_transformer_blocks=cfg["num_transformer_blocks"],
        rngs=nnx.Rngs(0),
    )


def main() -> None:
    import jax.numpy as jnp  # ensure available for nested defs

    # Rebuild nested classes properly (cleaner than above mess)
    from flax import nnx as nnx_mod
    import jax.numpy as jnp_mod

    class TokenAndPositionEmbedding(nnx_mod.Module):
        def __init__(self, maxlen, vocab_size, embed_dim, *, rngs):
            self.token_emb = nnx_mod.Embed(vocab_size, embed_dim, rngs=rngs)
            self.pos_emb = nnx_mod.Embed(maxlen, embed_dim, rngs=rngs)

        def __call__(self, x):
            seq_len = x.shape[1]
            positions = jnp_mod.arange(seq_len)[None, :]
            return self.token_emb(x) + self.pos_emb(positions)

    class TransformerBlock(nnx_mod.Module):
        def __init__(self, embed_dim, num_heads, ff_dim, *, rngs):
            self.attention = nnx_mod.MultiHeadAttention(
                num_heads=num_heads,
                in_features=embed_dim,
                qkv_features=embed_dim,
                out_features=embed_dim,
                decode=False,
                rngs=rngs,
            )

        def __call__(self, x, mask=None):
            return x + self.attention(x, mask=mask)

    class GPT(nnx_mod.Module):
        def __init__(
            self,
            maxlen,
            vocab_size,
            embed_dim,
            num_heads,
            feed_forward_dim,
            num_transformer_blocks,
            *,
            rngs,
        ):
            self.maxlen = maxlen
            self.embedding = TokenAndPositionEmbedding(
                maxlen, vocab_size, embed_dim, rngs=rngs
            )
            self.transformer_blocks = nnx_mod.List(
                [
                    TransformerBlock(
                        embed_dim, num_heads, feed_forward_dim, rngs=rngs
                    )
                    for _ in range(num_transformer_blocks)
                ]
            )
            self.output_layer = nnx_mod.Linear(
                embed_dim, vocab_size, use_bias=False, rngs=rngs
            )

    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    model = GPT(
        maxlen=cfg["maxlen"],
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        feed_forward_dim=cfg["feed_forward_dim"],
        num_transformer_blocks=cfg["num_transformer_blocks"],
        rngs=nnx_mod.Rngs(0),
    )

    cpu = jax.devices("cpu")[0]
    checkpointer = ocp.PyTreeCheckpointer()
    state = nnx_mod.state(model)
    restore_args = jax.tree_util.tree_map(
        lambda _: ocp.ArrayRestoreArgs(sharding=SingleDeviceSharding(cpu)),
        state,
    )
    restored = checkpointer.restore(
        ROOT / "small_gpt.orbax",
        item=state,
        restore_args=restore_args,
    )
    nnx_mod.update(model, restored)

    params = nnx_mod.state(model, nnx_mod.Param)
    flat: dict[str, np.ndarray] = {}
    for path, leaf in tree_flatten_with_path(params)[0]:
        key = path_to_key(path)
        value = leaf.value if hasattr(leaf, "value") else leaf
        flat[key] = np.asarray(value)

    out = ROOT / "weights.npz"
    np.savez_compressed(out, **flat)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB, {len(flat)} tensors)")
    print("sample:", list(flat)[:5])


if __name__ == "__main__":
    main()
