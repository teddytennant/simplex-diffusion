"""Toy training loop: CE decreases on synthetic sequences."""

import jax.numpy as jnp

from train import train


def test_train_ce_decreases():
    losses, _params = train(steps=80, seed=1)
    start, end = losses[0], losses[-1]
    assert end < start, f"CE did not drop: start={start:.4f} end={end:.4f}"
    assert jnp.isfinite(end)
