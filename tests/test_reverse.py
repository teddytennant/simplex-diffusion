"""Closed-form reverse (Prop. 3.3, eqs. 6-9)."""

import jax
import jax.numpy as jnp
import numpy as np

from sdm.forward import sample_forward
from sdm.reverse import reverse_mean, reverse_sample, reverse_step
from sdm.simplex import is_simplex, one_hot, uniform_prior


VOCAB = 8
EPS = 1.0


def test_reverse_mean_recovers_clean_as_t_to_0():
    p0 = one_hot(jnp.array(3), VOCAB)
    pi = uniform_prior(VOCAB)
    key = jax.random.PRNGKey(0)
    pt = sample_forward(key, p0, 0.6, pi=pi, eps=EPS)
    ps = reverse_mean(pt, p0, s=1e-3, t=0.6, pi=pi, eps=EPS, kappa=0.0)
    assert is_simplex(ps)
    np.testing.assert_allclose(np.asarray(ps), np.asarray(p0), atol=0.05)
    assert int(jnp.argmax(ps)) == 3


def test_reverse_sample_stays_on_simplex():
    p0 = one_hot(jnp.array([1, 4]), VOCAB)
    pi = uniform_prior(VOCAB)
    key = jax.random.PRNGKey(4)
    k1, k2 = jax.random.split(key)
    pt = sample_forward(k1, p0, 0.5, pi=pi, eps=EPS)
    ps = reverse_sample(k2, pt, p0, s=0.2, t=0.5, pi=pi, eps=EPS, kappa=0.3)
    assert is_simplex(ps)
    np.testing.assert_allclose(np.asarray(ps.sum(-1)), 1.0, atol=1e-5)


def test_reverse_mean_is_convex_combination_on_simplex():
    p0 = one_hot(jnp.array(0), VOCAB)
    pt = jnp.full((VOCAB,), 1.0 / VOCAB)
    ps = reverse_mean(pt, p0, s=0.25, t=0.8, kappa=0.0, eps=EPS)
    assert is_simplex(ps)
    assert float(ps.min()) >= -1e-6


def test_probability_conservation_project():
    """Eq. 6 thinning + eq. 8 convex combination stay on Δ^N."""
    p0 = one_hot(jnp.arange(VOCAB) % VOCAB, VOCAB)
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    pt = sample_forward(k1, p0, 0.9, eps=EPS)
    ps = reverse_sample(k2, pt, p0, s=0.4, t=0.9, kappa=0.5, eps=EPS)
    assert is_simplex(ps)
    # mass is conserved per simplex
    np.testing.assert_allclose(np.asarray(ps.sum(-1)), 1.0, atol=1e-5)


def test_kappa_one_independent_of_state_mean():
    """κ=1 ⇒ P_s ~ Dir(β_s) independent of P_t; mean is α_s P_0 + (1-α_s)π."""
    p0 = one_hot(jnp.zeros((256,), dtype=jnp.int32), VOCAB)
    pi = uniform_prior(VOCAB)
    pt = jnp.broadcast_to(pi, p0.shape)  # dummy, unused
    key = jax.random.PRNGKey(9)
    ps = reverse_sample(key, pt, p0, s=0.3, t=0.9, pi=pi, eps=EPS, kappa=1.0)
    expected = 0.7 * np.asarray(p0[0]) + 0.3 * np.asarray(pi)
    np.testing.assert_allclose(np.asarray(ps.mean(0)), expected, atol=0.05)
