"""DDIM-like sampler with churn/stochasticity η (Algorithm 2, Prop. 3.3).

Times 0 = t_0 < ⋯ < t_M = 1. Initialize P_{t_M} ~ Dir(c_1 π). Reverse
t_k → t_{k−1} for k = M, …, 2. Endpoint: x ~ Cat(P̂_θ(t_1, P_{t_1})), or
argmax when η = 0.

eta=0 is deterministic given the current P_t and the model prediction
(key unused in reverse_step). eta>0 samples the reverse kernel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sdm.dirichlet import sample_dirichlet
from sdm.forward import noise_prior_params
from sdm.reverse import reverse_step
from sdm.simplex import uniform_prior


def time_grid(n_steps: int):
    """t_0 = 0 < ⋯ < t_M = 1 with M = n_steps."""
    return jnp.linspace(0.0, 1.0, n_steps + 1, dtype=jnp.float32)


def sample(
    apply_fn,
    params,
    key,
    *,
    batch: int,
    seq_len: int,
    vocab: int,
    n_steps: int = 8,
    eta: float = 0.0,
    eps: float = 1.0,
    pi=None,
):
    """Generate token sequences. Returns (tokens, last P̂_0)."""
    if pi is None:
        pi = uniform_prior(vocab)
    times = time_grid(n_steps)
    key, k_init = jax.random.split(key)
    beta1 = noise_prior_params(vocab, eps=eps, pi=pi)
    p = sample_dirichlet(k_init, jnp.broadcast_to(beta1, (batch, seq_len, vocab)))
    # Algorithm 2: k = M, M−1, …, 2
    for k in range(n_steps, 1, -1):
        t = float(times[k])
        s = float(times[k - 1])
        t_b = jnp.full((batch,), t, dtype=jnp.float32)
        p_hat = apply_fn(params, p, t_b)
        key, k_step = jax.random.split(key)
        p = reverse_step(k_step, p, p_hat, s, t, pi=pi, eps=eps, eta=eta)
    t1 = float(times[1]) if n_steps >= 1 else 1.0
    p_hat = apply_fn(params, p, jnp.full((batch,), t1, dtype=jnp.float32))
    if eta <= 0.0:
        tokens = jnp.argmax(p_hat, axis=-1)
    else:
        key, k_end = jax.random.split(key)
        logits = jnp.log(jnp.clip(p_hat, 1e-12, 1.0))
        tokens = jax.random.categorical(k_end, logits, axis=-1)
    return tokens, p_hat
