"""Cross-entropy loss (eq. 11)."""

import jax.numpy as jnp
import numpy as np

from sdm.loss import cross_entropy, cross_entropy_onehot, denoise_loss
from sdm.simplex import one_hot


def test_ce_zero_when_model_is_one_hot_true_token():
    y = one_hot(jnp.array([2, 0, 5]), vocab=8)
    ce = cross_entropy_onehot(y, y)
    np.testing.assert_allclose(np.asarray(ce), 0.0, atol=1e-6)
    ce2 = cross_entropy(y, jnp.array([2, 0, 5]))
    np.testing.assert_allclose(np.asarray(ce2), 0.0, atol=1e-6)


def test_ce_is_neg_sum_y_log_p():
    p = jnp.array([0.25, 0.75, 0.0])
    y = jnp.array([1.0, 0.0, 0.0])
    ce = float(cross_entropy_onehot(p, y))
    np.testing.assert_allclose(ce, -np.log(0.25), rtol=1e-5)
    tokens = jnp.array(0)
    ce_tok = float(cross_entropy(p, tokens))
    np.testing.assert_allclose(ce_tok, -np.log(0.25), rtol=1e-5)


def test_denoise_loss_mean_over_tokens():
    p = one_hot(jnp.array([[0, 1], [1, 1]]), vocab=3)
    # true tokens match first row, mismatch second-row pos 0
    tokens = jnp.array([[0, 1], [0, 1]])
    loss = float(denoise_loss(p, tokens))
    # three of four positions are exact (CE=0); one is p=0 for true class
    # clip makes -log(1e-12) on the mismatch
    assert loss > 0.0
