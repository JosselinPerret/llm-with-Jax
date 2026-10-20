"""GPT decoder built with Flax NNX (no nnx.List — Space-compatible)."""

from __future__ import annotations

import flax.nnx as nnx
import jax.numpy as jnp


class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

    def __call__(self, x):
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        return self.token_emb(x) + self.pos_emb(positions)


class TransformerBlock(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, *, rngs: nnx.Rngs):
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
        maxlen: int,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        num_transformer_blocks: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.maxlen = maxlen
        self.num_transformer_blocks = num_transformer_blocks
        self.embedding = TokenAndPositionEmbedding(
            maxlen, vocab_size, embed_dim, rngs=rngs
        )
        # Named attributes instead of nnx.List (missing on older Flax).
        for i in range(num_transformer_blocks):
            setattr(
                self,
                f"block_{i}",
                TransformerBlock(embed_dim, num_heads, feed_forward_dim, rngs=rngs),
            )
        self.output_layer = nnx.Linear(
            embed_dim, vocab_size, use_bias=False, rngs=rngs
        )

    def causal_attention_mask(self, seq_len: int):
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    def __call__(self, token_ids):
        seq_len = token_ids.shape[1]
        mask = self.causal_attention_mask(seq_len)
        x = self.embedding(token_ids)
        for i in range(self.num_transformer_blocks):
            x = getattr(self, f"block_{i}")(x, mask=mask)
        return self.output_layer(x)
