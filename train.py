#!/usr/bin/env python3
"""Toy SDM training on synthetic categorical sequences (CPU)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from sdm._host import preload

preload()

import jax
import jax.numpy as jnp

from sdm.loss import training_loss
from sdm.model import apply_denoiser, init_denoiser
from sdm.simplex import uniform_prior

VOCAB = 8
SEQ_LEN = 8
HIDDEN = 64
BATCH = 32
STEPS = 250
LR = 0.08
EPS = 1.0
T_MIN = 0.05


def toy_sequences():
    """A handful of repeating patterns over a small vocab."""
    patterns = jnp.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [7, 6, 5, 4, 3, 2, 1, 0],
            [0, 0, 1, 1, 2, 2, 3, 3],
            [1, 2, 1, 2, 1, 2, 1, 2],
            [3, 3, 3, 3, 5, 5, 5, 5],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [2, 3, 4, 2, 3, 4, 2, 3],
            [4, 4, 0, 0, 4, 4, 0, 0],
        ],
        dtype=jnp.int32,
    )
    return patterns


def sample_batch(key, data, batch=BATCH):
    idx = jax.random.randint(key, (batch,), 0, data.shape[0])
    return data[idx]


def sgd_update(params, grads, lr=LR):
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)


def train(steps: int = STEPS, seed: int = 0):
    data = toy_sequences()
    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)
    params = init_denoiser(k_init, vocab=VOCAB, seq_len=SEQ_LEN, hidden=HIDDEN)
    pi = uniform_prior(VOCAB)

    def loss_fn(p, rng, tokens, t):
        return training_loss(p, apply_denoiser, rng, tokens, t, vocab=VOCAB, pi=pi, eps=EPS)

    @jax.jit
    def step(params, key):
        k_data, k_t, k_fwd, k_next = jax.random.split(key, 4)
        tokens = sample_batch(k_data, data)
        t = jax.random.uniform(k_t, (tokens.shape[0],), minval=T_MIN, maxval=1.0)
        loss, grads = jax.value_and_grad(loss_fn)(params, k_fwd, tokens, t)
        params = sgd_update(params, grads)
        return params, k_next, loss

    losses = []
    for i in range(steps):
        params, key, loss = step(params, key)
        losses.append(float(loss))
        if i == 0 or i == steps - 1 or (i + 1) % 50 == 0:
            print(f"step {i + 1:4d}  ce {float(loss):.4f}")
    return losses, params


def main():
    losses, _params = train()
    start, end = losses[0], losses[-1]
    print(f"start CE {start:.4f}")
    print(f"end   CE {end:.4f}")
    if end >= start:
        raise SystemExit("expected training CE to decrease")


if __name__ == "__main__":
    main()
