"""Tiny sequence denoiser: flatten P_t, concat t, MLP, softmax (Δ^N per token).

Not the paper's Transformer LM. Joint over the length-L product of simplices
(Appendix C.1) via a fully connected MLP on the concatenated probability
vectors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def init_denoiser(key, vocab: int, seq_len: int, hidden: int = 64):
    k1, k2, k3 = jax.random.split(key, 3)
    in_dim = seq_len * vocab + 1
    out_dim = seq_len * vocab
    scale = 0.2
    params = {
        "W1": jax.random.normal(k1, (in_dim, hidden), dtype=jnp.float32) * scale,
        "b1": jnp.zeros((hidden,), dtype=jnp.float32),
        "W2": jax.random.normal(k2, (hidden, hidden), dtype=jnp.float32) * scale,
        "b2": jnp.zeros((hidden,), dtype=jnp.float32),
        "W3": jax.random.normal(k3, (hidden, out_dim), dtype=jnp.float32) * scale,
        "b3": jnp.zeros((out_dim,), dtype=jnp.float32),
    }
    return params


def apply_denoiser(params, p_t, t):
    """P̂_0(t, P_t) ∈ Δ^N per position. p_t: (B, L, V), t: scalar or (B,)."""
    p_t = jnp.asarray(p_t, dtype=jnp.float32)
    batch, seq_len, vocab = p_t.shape
    t = jnp.asarray(t, dtype=jnp.float32)
    t = jnp.broadcast_to(t, (batch,)).reshape(batch, 1)
    x = jnp.concatenate([p_t.reshape(batch, seq_len * vocab), t], axis=-1)
    h = jax.nn.relu(x @ params["W1"] + params["b1"])
    h = jax.nn.relu(h @ params["W2"] + params["b2"])
    logits = h @ params["W3"] + params["b3"]
    return jax.nn.softmax(logits.reshape(batch, seq_len, vocab), axis=-1)
