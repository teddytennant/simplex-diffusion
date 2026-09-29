"""Forward noising on Δ^N (eq. 3-4, Prop. 3.2, Algorithm 1).

  P_0 = e_{x_0}
  P_t ~ Dir(β_t(P_0, π))    with β_t = c_t (α_t P_0 + (1 − α_t) π)

Equivalent interpolation (eq. 4):
  W_t ~ Beta(c_t α_t, c_t (1 − α_t)),  V_t ~ Dir(c_t (1 − α_t) π)
  P_t = W_t P_0 + (1 − W_t) V_t
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sdm.dirichlet import sample_beta, sample_dirichlet
from sdm.schedules import T_EPS, beta_params, expand_t, forward_mean
from sdm.simplex import project_simplex, uniform_prior


def sample_forward(key, p0, t, pi=None, eps: float = 1.0):
    """Sample P_t | P_0 from Dir(β_t). t=0 returns P_0 (degenerate)."""
    p0 = jnp.asarray(p0, dtype=jnp.float32)
    if pi is None:
        pi = uniform_prior(p0.shape[-1], dtype=p0.dtype)
    beta = beta_params(p0, t, pi=pi, eps=eps)
    pt = sample_dirichlet(key, beta)
    t_b = expand_t(t, p0.ndim)
    pt = jnp.where(t_b <= T_EPS, p0, pt)
    return project_simplex(pt)


def sample_forward_interpolation(key, p0, t, pi=None, eps: float = 1.0):
    """Eq. 4 / Prop. 3.2: P_t = W_t P_0 + (1 − W_t) V_t."""
    p0 = jnp.asarray(p0, dtype=jnp.float32)
    if pi is None:
        pi = uniform_prior(p0.shape[-1], dtype=p0.dtype)
    t_b = expand_t(t, p0.ndim)
    alpha = 1.0 - t_b
    c = jnp.asarray(eps, dtype=p0.dtype) / jnp.clip(t_b, T_EPS, None)
    k_w, k_v = jax.random.split(key)
    lead = p0.shape[:-1]
    a_w = jnp.broadcast_to(c * alpha, lead + (1,))
    b_w = jnp.broadcast_to(c * (1.0 - alpha), lead + (1,))
    w = sample_beta(k_w, a_w, b_w)
    v = sample_dirichlet(k_v, jnp.broadcast_to(c * (1.0 - alpha) * pi, p0.shape))
    pt = w * p0 + (1.0 - w) * v
    pt = jnp.where(t_b <= T_EPS, p0, pt)
    return project_simplex(pt)


def noise_prior_params(vocab: int, eps: float = 1.0, pi=None):
    """p_1 = Dir(c_1 π) with α_1 = 0 (after eq. 10)."""
    if pi is None:
        pi = uniform_prior(vocab)
    c1 = jnp.asarray(eps, dtype=jnp.float32)  # c_1 = ε / 1
    return c1 * pi


__all__ = [
    "forward_mean",
    "noise_prior_params",
    "sample_forward",
    "sample_forward_interpolation",
]
