JAX kernel of Simplex Diffusion Models (arXiv:2609.35553).

## Install

```
python3 -m venv .venv && .venv/bin/pip install -e ".[tests]"
```

## Run

```
.venv/bin/python -m pytest -q
.venv/bin/python train.py
```

## What does not match the paper

No OpenWebText, TinyGSM, GSM8K, adaptive grids, large embeddings, distillation to 8 steps, classifier-free guidance, or Marsaglia-Tsang Gamma sampling. Linear interpolation α_t = 1 − t and inverse-temperature concentration c_t = ε/(1 − α_t) only; tiny MLP denoiser on synthetic categorical sequences, not a language-model stack.
