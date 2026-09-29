"""Closed-form reverse transition (Prop. 3.3, eqs. 6-9, Algorithm 2).

For s < t, κ ∈ [0, 1]:
  r_{s,t} = min(1, b_s / b_t),  b_u = c_u (1 − α_u)
  ρ = (1 − κ) r_{s,t}
  B_i ~ Beta(ρ β_{t,i}, (1 − ρ) β_{t,i})
  P^κ_{s,t,i} = B_i P_{t,i} / ∑_j B_j P_{t,j}          (eq. 6)
  W ~ Beta(ρ c_t, c_s − ρ c_t)
  V ~ Dir(β_s(P_0, π) − ρ β_t(P_0, π))                 (eq. 7)
  P_s = W P^κ_{s,t} + (1 − W) V                        (eq. 8)

κ = 0 retains the largest compatible fraction of P_t (eq. 9).
With c_u = ε/(1 − α_u) one has r_{s,t} = 1, and as ε → ∞ the mean of
eq. 9 is the deterministic DDIM rule
  P_s = ((1−α_s)/(1−α_t)) P_t + ((α_s−α_t)/(1−α_t)) e_{x_0}.

`eta` is DDIM-style stochasticity: eta=0 uses that mean (no RNG);
eta>0 samples Prop. 3.3 with churn κ = eta (clipped to [0, 1]).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sdm.dirichlet import dirichlet_mean, sample_beta, sample_dirichlet
from sdm.schedules import T_EPS, alpha_linear, beta_params, concentration
from sdm.simplex import project_simplex, uniform_prior

RHO_EPS = 1e-6


def reverse_coefficients(s, t, eps: float = 1.0, kappa: float = 0.0):
    """(ρ, c_s, c_t, r) for Prop. 3.3. Inverse-temperature schedule ⇒ r = 1."""
    s = jnp.asarray(s, dtype=jnp.float32)
    t = jnp.asarray(t, dtype=jnp.float32)
    alpha_s = alpha_linear(s)
    alpha_t = alpha_linear(t)
    c_s = concentration(s, eps=eps)
    c_t = concentration(t, eps=eps)
    b_s = c_s * (1.0 - alpha_s)
    b_t = c_t * (1.0 - alpha_t)
    r = jnp.minimum(1.0, b_s / jnp.maximum(b_t, T_EPS))
    kappa = jnp.clip(jnp.asarray(kappa, dtype=jnp.float32), 0.0, 1.0)
    rho = (1.0 - kappa) * r
    return rho, c_s, c_t, r


def _innovation_alpha(p0, s, t, pi, eps, rho):
    """Dirichlet parameters of V: β_s − ρ β_t (eq. 7)."""
    beta_s = beta_params(p0, s, pi=pi, eps=eps)
    beta_t = beta_params(p0, t, pi=pi, eps=eps)
    return jnp.maximum(beta_s - rho * beta_t, 0.0)


def reverse_mean(p_t, p0, s, t, pi=None, eps: float = 1.0, kappa: float = 0.0):
    """E[P_s | P_0, P_t] under Prop. 3.3 using E[W], E[V], E[B].

    With the paper's inverse-temperature schedule and κ=0 this is the
    deterministic DDIM update (after eq. 9).
    """
    p_t = jnp.asarray(p_t, dtype=jnp.float32)
    p0 = jnp.asarray(p0, dtype=jnp.float32)
    if pi is None:
        pi = uniform_prior(p_t.shape[-1], dtype=p_t.dtype)
    rho, c_s, c_t, _r = reverse_coefficients(s, t, eps=eps, kappa=kappa)
    a_w = rho * c_t
    b_w = c_s - rho * c_t
    w = a_w / jnp.maximum(a_w + b_w, T_EPS)
    v = dirichlet_mean(_innovation_alpha(p0, s, t, pi, eps, rho))
    # E[B_i] = ρ when both Beta parameters are positive; ρ=1 ⇒ P^κ = P_t.
    p_thin = jnp.where(rho >= 1.0 - RHO_EPS, p_t, project_simplex(rho * p_t))
    p_s = w * p_thin + (1.0 - w) * v
    return project_simplex(p_s)


def reverse_sample(
    key,
    p_t,
    p0,
    s,
    t,
    pi=None,
    eps: float = 1.0,
    kappa: float = 0.0,
):
    """Draw P_s | P_0, P_t from the Prop. 3.3 kernel (eqs. 6-8)."""
    p_t = jnp.asarray(p_t, dtype=jnp.float32)
    p0 = jnp.asarray(p0, dtype=jnp.float32)
    if pi is None:
        pi = uniform_prior(p_t.shape[-1], dtype=p_t.dtype)
    rho, c_s, c_t, _r = reverse_coefficients(s, t, eps=eps, kappa=kappa)
    a_w = rho * c_t
    b_w = jnp.maximum(c_s - rho * c_t, 0.0)
    k_w, k_v, k_b = jax.random.split(key, 3)
    lead = p_t.shape[:-1]
    w = sample_beta(
        k_w,
        jnp.broadcast_to(a_w, lead),
        jnp.broadcast_to(b_w, lead),
    )[..., None]
    v = sample_dirichlet(k_v, _innovation_alpha(p0, s, t, pi, eps, rho))
    beta_t = beta_params(p0, t, pi=pi, eps=eps)
    # ρ ≈ 1: skip thinning (Algorithm 2). ρ ≈ 0: W ≈ 0, thinning unused.
    b_keep = sample_beta(k_b, rho * beta_t, (1.0 - rho) * beta_t)
    p_thin_stoch = project_simplex(b_keep * p_t)
    p_thin = jnp.where(rho >= 1.0 - RHO_EPS, p_t, p_thin_stoch)
    p_s = w * p_thin + (1.0 - w) * v
    return project_simplex(p_s)


def _vertex_from_pred(p0_pred):
    """Plug-in vertex e_{x_0} with x_0 = argmax P̂ (Algorithm 2)."""
    vocab = p0_pred.shape[-1]
    idx = jnp.argmax(p0_pred, axis=-1)
    return jax.nn.one_hot(idx, vocab, dtype=p0_pred.dtype)


def _sample_vertex(key, p0_pred):
    """x_0 ~ Cat(P̂_θ), P_0 = e_{x_0} (Algorithm 2)."""
    logits = jnp.log(jnp.clip(p0_pred, 1e-12, 1.0))
    idx = jax.random.categorical(key, logits, axis=-1)
    return jax.nn.one_hot(idx, p0_pred.shape[-1], dtype=p0_pred.dtype)


def reverse_step(
    key,
    p_t,
    p0_pred,
    s,
    t,
    pi=None,
    eps: float = 1.0,
    eta: float = 0.0,
    kappa=None,
):
    """One DDIM-like reverse step from t → s given the model prediction.

    eta=0: deterministic given P̂ (argmax vertex + reverse mean; key unused).
    eta>0: sample x_0 ~ Cat(P̂) then Prop. 3.3 with churn κ (default κ=eta).
    """
    p_t = jnp.asarray(p_t, dtype=jnp.float32)
    p0_pred = jnp.asarray(p0_pred, dtype=jnp.float32)
    eta = float(eta)
    if kappa is None:
        kappa = eta
    kappa = float(jnp.clip(kappa, 0.0, 1.0))
    if eta <= 0.0:
        p0 = _vertex_from_pred(p0_pred)
        return reverse_mean(p_t, p0, s, t, pi=pi, eps=eps, kappa=0.0)
    k_x, k_rev = jax.random.split(key)
    p0 = _sample_vertex(k_x, p0_pred)
    return reverse_sample(k_rev, p_t, p0, s, t, pi=pi, eps=eps, kappa=kappa)
