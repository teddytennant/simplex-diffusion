"""Dirichlet / Beta sampling via independent Gammas (paper Appendix, Prop. 3.2-3.3).

JAX `random.gamma` is used instead of the paper's Marsaglia-Tsang kernel.
Zero concentration coordinates are mapped to zero mass (simplex face).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sdm.simplex import project_simplex

GAMMA_MIN = 1e-8


def sample_gamma(key, shape_param):
    """Gamma(shape, rate=1), with shape 0 → 0."""
    shape_param = jnp.asarray(shape_param, dtype=jnp.float32)
    safe = jnp.maximum(shape_param, GAMMA_MIN)
    g = jax.random.gamma(key, safe)
    return jnp.where(shape_param > 0.0, g, 0.0)


def sample_dirichlet(key, alpha, axis: int = -1):
    """P ~ Dir(α). α_i = 0 forces P_i = 0. Last (or given) axis is the simplex."""
    g = sample_gamma(key, alpha)
    z = jnp.sum(g, axis=axis, keepdims=True)
    return g / jnp.maximum(z, 1e-12)


def sample_beta(key, a, b):
    """Beta(a, b) via Gamma ratio. Degenerate: a=0 → 0, b=0 → 1, both 0 → 0.5."""
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    k1, k2 = jax.random.split(key)
    g1 = sample_gamma(k1, jnp.maximum(a, 0.0))
    g2 = sample_gamma(k2, jnp.maximum(b, 0.0))
    raw = g1 / jnp.maximum(g1 + g2, 1e-12)
    a_pos = a > 0.0
    b_pos = b > 0.0
    out = jnp.where(a_pos & b_pos, raw, 0.5)
    out = jnp.where((~a_pos) & b_pos, 0.0, out)
    out = jnp.where(a_pos & (~b_pos), 1.0, out)
    return out


def dirichlet_mean(alpha, axis: int = -1):
    """E[Dir(α)] = α / ∑ α."""
    alpha = jnp.asarray(alpha, dtype=jnp.float32)
    z = jnp.sum(alpha, axis=axis, keepdims=True)
    return alpha / jnp.maximum(z, 1e-12)


def normalize_simplex(p, axis: int = -1):
    return project_simplex(p, axis=axis)
