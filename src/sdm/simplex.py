"""Probability simplex Δ^N (paper eq. 1) and projection."""

from __future__ import annotations

import jax.numpy as jnp


def uniform_prior(vocab: int, dtype=jnp.float32):
    """π = Unif({1, …, N}), the paper's default noise prior."""
    return jnp.full((vocab,), 1.0 / vocab, dtype=dtype)


def one_hot(tokens, vocab: int, dtype=jnp.float32):
    """Vertices of Δ^N: P_0 = e_{x_0}."""
    return jnp.astype(tokens[..., None] == jnp.arange(vocab), dtype)


def project_simplex(p, axis: int = -1, eps: float = 1e-12):
    """Clip negatives and renormalize (thinning step in eq. 6).

    A zero (or near-zero) vector is mapped to the uniform point of Δ^N.
    """
    p = jnp.maximum(p, 0.0)
    z = jnp.sum(p, axis=axis, keepdims=True)
    n = p.shape[axis]
    uniform = jnp.ones_like(p) / jnp.asarray(n, dtype=p.dtype)
    return jnp.where(z > eps, p / jnp.maximum(z, eps), uniform)


def is_simplex(p, axis: int = -1, atol: float = 1e-5) -> bool:
    """True if p ≥ 0 and ∑_i p_i = 1 along `axis`."""
    p = jnp.asarray(p)
    nonneg = bool(jnp.all(p >= -atol))
    sums = jnp.sum(p, axis=axis)
    unit = bool(jnp.allclose(sums, 1.0, atol=atol))
    return nonneg and unit
