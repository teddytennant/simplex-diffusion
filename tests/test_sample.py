"""DDIM-like sampler: η=0 deterministic, η>0 stochastic (Algorithm 2)."""

import jax
import jax.numpy as jnp
import numpy as np

from sdm.reverse import reverse_step
from sdm.sample import sample
from sdm.simplex import one_hot, uniform_prior


VOCAB = 8


def _oracle_apply(params, p_t, t):
    """Model that always predicts a fixed one-hot (class 2)."""
    del params, t
    batch, seq, vocab = p_t.shape
    return jnp.broadcast_to(one_hot(jnp.array(2), vocab), (batch, seq, vocab))


def test_eta_zero_deterministic_given_prediction():
    p0_pred = one_hot(jnp.array(4), VOCAB)
    p_t = uniform_prior(VOCAB)
    s, t = 0.2, 0.7
    k1, k2 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
    a = reverse_step(k1, p_t, p0_pred, s, t, eta=0.0)
    b = reverse_step(k2, p_t, p0_pred, s, t, eta=0.0)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-6)


def test_eta_positive_changes_samples():
    p0_pred = one_hot(jnp.array(4), VOCAB)
    p_t = uniform_prior(VOCAB)
    s, t = 0.2, 0.7
    k1, k2 = jax.random.PRNGKey(0), jax.random.PRNGKey(99)
    a = reverse_step(k1, p_t, p0_pred, s, t, eta=1.0)
    b = reverse_step(k2, p_t, p0_pred, s, t, eta=1.0)
    assert not np.allclose(np.asarray(a), np.asarray(b), atol=1e-5)


def test_eta_zero_full_sampler_same_key_same_tokens():
    params = {}
    key = jax.random.PRNGKey(3)
    tok_a, _ = sample(
        _oracle_apply, params, key, batch=2, seq_len=4, vocab=VOCAB, n_steps=4, eta=0.0
    )
    tok_b, _ = sample(
        _oracle_apply, params, key, batch=2, seq_len=4, vocab=VOCAB, n_steps=4, eta=0.0
    )
    np.testing.assert_array_equal(np.asarray(tok_a), np.asarray(tok_b))
    # oracle always predicts class 2
    np.testing.assert_array_equal(np.asarray(tok_a), 2)


def _identity_apply(params, p_t, t):
    """Pass the current simplex state through as P̂_0 (so reverse noise is visible)."""
    del params, t
    return p_t


def test_eta_positive_full_sampler_differs_across_keys():
    params = {}
    a, _ = sample(
        _identity_apply,
        params,
        jax.random.PRNGKey(1),
        batch=8,
        seq_len=4,
        vocab=VOCAB,
        n_steps=4,
        eta=1.0,
    )
    b, _ = sample(
        _identity_apply,
        params,
        jax.random.PRNGKey(2),
        batch=8,
        seq_len=4,
        vocab=VOCAB,
        n_steps=4,
        eta=1.0,
    )
    # stochastic reverse + categorical endpoint; extremely unlikely to match
    assert not np.array_equal(np.asarray(a), np.asarray(b))
