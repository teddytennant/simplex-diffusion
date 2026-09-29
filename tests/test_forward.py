"""Forward Dirichlet path (eq. 3-4, Prop. 3.1-3.2)."""

import jax
import jax.numpy as jnp
import numpy as np

from sdm.forward import (
    forward_mean,
    noise_prior_params,
    sample_forward,
    sample_forward_interpolation,
)
from sdm.simplex import is_simplex, one_hot, uniform_prior


VOCAB = 8
EPS = 1.0


def _many_forward(t, n=512, interpolation=False):
    key = jax.random.PRNGKey(0)
    tokens = jnp.zeros((n,), dtype=jnp.int32)  # all mass on class 0
    p0 = one_hot(tokens, VOCAB)
    pi = uniform_prior(VOCAB)
    fn = sample_forward_interpolation if interpolation else sample_forward
    pt = fn(key, p0, t, pi=pi, eps=EPS)
    return np.asarray(pt), np.asarray(p0), np.asarray(pi)


def test_forward_samples_stay_on_simplex():
    for t in (0.0, 0.2, 0.7, 1.0):
        pt, _, _ = _many_forward(t, n=64)
        assert pt.min() >= -1e-6
        np.testing.assert_allclose(pt.sum(-1), 1.0, atol=1e-5)


def test_forward_mean_matches_interpolation():
    t = 0.4
    pt, p0, pi = _many_forward(t, n=2048)
    expected = np.asarray(forward_mean(p0[:1], t, pi=pi)[0])
    np.testing.assert_allclose(pt.mean(axis=0), expected, atol=0.04)


def test_t_to_T_marginal_is_noise_prior():
    """α_1 = 0 ⇒ P_1 ~ Dir(c_1 π) independent of x_0 (after eq. 10)."""
    key = jax.random.PRNGKey(1)
    n = 1024
    tokens = jax.random.randint(key, (n,), 0, VOCAB)
    p0 = one_hot(tokens, VOCAB)
    pi = uniform_prior(VOCAB)
    pt = sample_forward(jax.random.PRNGKey(2), p0, 1.0, pi=pi, eps=EPS)
    pt = np.asarray(pt)
    assert is_simplex(pt)
    np.testing.assert_allclose(pt.mean(axis=0), np.asarray(pi), atol=0.04)
    # prior parameters Dir(c_1 π) with c_1 = ε
    beta1 = np.asarray(noise_prior_params(VOCAB, eps=EPS, pi=pi))
    np.testing.assert_allclose(beta1, np.asarray(pi) * EPS, atol=1e-6)


def test_interpolation_path_on_simplex_and_mean():
    pt, p0, pi = _many_forward(0.5, n=1024, interpolation=True)
    assert pt.min() >= -1e-6
    np.testing.assert_allclose(pt.sum(-1), 1.0, atol=1e-5)
    expected = np.asarray(forward_mean(p0[:1], 0.5, pi=pi)[0])
    np.testing.assert_allclose(pt.mean(axis=0), expected, atol=0.05)


def test_t_zero_is_clean_one_hot():
    pt, p0, _ = _many_forward(0.0, n=8)
    np.testing.assert_allclose(pt, p0, atol=1e-6)
