"""Separable time-induced neural networks, with SPINN and TINN baselines."""

import os

# float64 for LM; set before JAX is imported.
os.environ.setdefault("JAX_ENABLE_X64", "True")
