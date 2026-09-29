"""Load host C++/zlib before test modules import jax."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from sdm._host import preload

preload()
