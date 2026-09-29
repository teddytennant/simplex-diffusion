"""Interpolation α_t and concentration c_t (Section 3.1).

Default schedules used in the paper's figures and Algorithm 1/2:
  α_t = 1 − t                         (linear interpolation)
  c_t = ε / (1 − α_t) = ε / t         (inverse-temperature, t > 0)

Dirichlet path parameters (after eq. 3):
  β_t(P_0, π) = c_t (α_t P_0 + (1 − α_t) π)
  a_t = c_t α_t,  b_t = c_t (1 − α_t)
"""

from __future__ import annotations

import jax.numpy as jnp

from sdm.simplex import uniform_prior

T_EPS = 1e-6


def expand_t(t, ndim_p0: int):
    """Broadcast t (scalar or leading-batch) against P with last axis = vocab."""
    t = jnp.asarray(t, dtype=jnp.float32)
    extra = max(ndim_p0 - t.ndim - 1, 0)
    return t.reshape(t.shape + (1,) * extra + (1,))


def alpha_linear(t):
    """α_t = 1 − t, t ∈ [0, 1]. Data at t=0, noise at t=1."""
    return 1.0 - jnp.asarray(t, dtype=jnp.float32)


def concentration(t, eps: float = 1.0):
    """c_t = ε / (1 − α_t) = ε / t. Clipped for t → 0."""
    t = jnp.asarray(t, dtype=jnp.float32)
    return jnp.asarray(eps, dtype=jnp.float32) / jnp.clip(t, T_EPS, None)


def beta_params(p0, t, pi=None, eps: float = 1.0):
    """β_t(P_0, π) = c_t (α_t P_0 + (1 − α_t) π), last axis = categories."""
    p0 = jnp.asarray(p0, dtype=jnp.float32)
    if pi is None:
        pi = uniform_prior(p0.shape[-1], dtype=p0.dtype)
    t_b = expand_t(t, p0.ndim)
    alpha = 1.0 - t_b
    c = jnp.asarray(eps, dtype=p0.dtype) / jnp.clip(t_b, T_EPS, None)
    mean = alpha * p0 + (1.0 - alpha) * pi
    return c * mean


def forward_mean(p0, t, pi=None):
    """E[P_t | P_0] = α_t P_0 + (1 − α_t) π (Dirichlet mean)."""
    p0 = jnp.asarray(p0, dtype=jnp.float32)
    if pi is None:
        pi = uniform_prior(p0.shape[-1], dtype=p0.dtype)
    t_b = expand_t(t, p0.ndim)
    alpha = 1.0 - t_b
    return alpha * p0 + (1.0 - alpha) * pi
