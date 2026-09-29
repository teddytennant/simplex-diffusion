"""Simplex constraints (eq. 1) and projection (eq. 6 normalization)."""

import jax.numpy as jnp

from sdm.simplex import is_simplex, one_hot, project_simplex, uniform_prior


def test_uniform_prior_on_simplex():
    pi = uniform_prior(8)
    assert is_simplex(pi)
    assert jnp.allclose(pi, 1.0 / 8)


def test_one_hot_vertices():
    p0 = one_hot(jnp.array([0, 3, 7]), vocab=8)
    assert p0.shape == (3, 8)
    assert is_simplex(p0)
    assert jnp.allclose(p0.sum(-1), 1.0)
    assert float(p0[1, 3]) == 1.0


def test_project_simplex_restores_sum_and_nonneg():
    raw = jnp.array([[1.5, -0.2, 0.3], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    p = project_simplex(raw)
    assert is_simplex(p)
    assert jnp.all(p >= 0.0)
    # zero vector → uniform-ish via eps; still sums to 1
    assert jnp.allclose(p.sum(-1), 1.0)
