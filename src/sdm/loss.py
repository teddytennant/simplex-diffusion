"""Cross-entropy training loss (eq. 11, Algorithm 1).

  L(θ) = E[ − log P̂_{0, x_0}(t, P_t) ]
        = E[ − ∑_i P_{0,i} log P̂_{0,i} ]

The denoiser outputs a distribution on Δ^N; the target is the clean token
(equivalently the one-hot vertex P_0).
"""

from __future__ import annotations

import jax.numpy as jnp

from sdm.forward import sample_forward
from sdm.simplex import one_hot, uniform_prior


def cross_entropy(p_hat, target, from_one_hot: bool = False):
    """−∑ y log p. `target` is integer tokens, or one-hot if `from_one_hot`."""
    p_hat = jnp.asarray(p_hat, dtype=jnp.float32)
    logp = jnp.log(jnp.clip(p_hat, 1e-12, 1.0))
    if from_one_hot:
        y = jnp.asarray(target, dtype=p_hat.dtype)
        nll = -jnp.sum(y * logp, axis=-1)
    else:
        tokens = jnp.asarray(target, dtype=jnp.int32)
        nll = -jnp.take_along_axis(logp, tokens[..., None], axis=-1)[..., 0]
    return nll


def cross_entropy_onehot(p_hat, y):
    """−∑_i y_i log p_i (eq. 11). Zero when p = y = one-hot."""
    p_hat = jnp.asarray(p_hat, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=p_hat.dtype)
    return -jnp.sum(y * jnp.log(jnp.clip(p_hat, 1e-12, 1.0)), axis=-1)


def denoise_loss(p_hat, tokens):
    """Mean token CE of P̂_0 against clean tokens x_0."""
    return jnp.mean(cross_entropy(p_hat, tokens))


def training_loss(params, apply_fn, key, tokens, t, vocab: int, pi=None, eps: float = 1.0):
    """Algorithm 1: P_0 = e_{x_0}, P_t ~ Dir(β_t), CE(apply(t, P_t), x_0)."""
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    p0 = one_hot(tokens, vocab)
    if pi is None:
        pi = uniform_prior(vocab, dtype=p0.dtype)
    pt = sample_forward(key, p0, t, pi=pi, eps=eps)
    p_hat = apply_fn(params, pt, t)
    return denoise_loss(p_hat, tokens)
