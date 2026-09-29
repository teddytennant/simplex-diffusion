"""Simplex Diffusion Models (arXiv:2609.35553) JAX kernel."""

from sdm._host import preload as _preload

_preload()

from sdm.forward import forward_mean, sample_forward, sample_forward_interpolation
from sdm.loss import cross_entropy, cross_entropy_onehot, denoise_loss, training_loss
from sdm.model import apply_denoiser, init_denoiser
from sdm.reverse import reverse_mean, reverse_sample, reverse_step
from sdm.sample import sample
from sdm.schedules import alpha_linear, beta_params, concentration
from sdm.simplex import is_simplex, one_hot, project_simplex, uniform_prior

__all__ = [
    "alpha_linear",
    "apply_denoiser",
    "beta_params",
    "concentration",
    "cross_entropy",
    "cross_entropy_onehot",
    "denoise_loss",
    "forward_mean",
    "init_denoiser",
    "is_simplex",
    "one_hot",
    "project_simplex",
    "reverse_mean",
    "reverse_sample",
    "reverse_step",
    "sample",
    "sample_forward",
    "sample_forward_interpolation",
    "training_loss",
    "uniform_prior",
]
