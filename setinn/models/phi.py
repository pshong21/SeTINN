"""The shared time code Phi(t) = (1 - alpha) t + alpha N(t), as in TINN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from jax.nn.initializers import glorot_uniform, zeros

from .precision import cast_floating, require_floating_dtype


def init_phi(seed, m, hidden=(16, 16)):
    """Weights, biases, output weights and gate for an ``m``-channel time code."""

    key = jax.random.PRNGKey(seed)
    dims = [1, *hidden]
    gu = glorot_uniform()
    Ws, bs = [], []
    for i in range(len(dims) - 1):
        key, k = jax.random.split(key)
        Ws.append(gu(k, (dims[i], dims[i + 1])))
        bs.append(zeros(k, (dims[i + 1],)))
    key, k = jax.random.split(key)
    Wout = gu(k, (dims[-1], m))
    alpha = jnp.ones((1, m))
    return {"Ws": Ws, "bs": bs, "Wout": Wout, "alpha": alpha}


def phi_apply(phi_params, t):
    """Evaluate Phi(t) for a scalar time, returning shape ``(m,)``."""

    t = jnp.reshape(t, (1, 1))
    h = t
    for W, b in zip(phi_params["Ws"], phi_params["bs"]):
        h = jnp.tanh(h @ W + b)
    n = h @ phi_params["Wout"]                    # (1, m)
    alpha = phi_params["alpha"]                   # (1, m)
    phi = (1.0 - alpha) * t + alpha * n           # t broadcasts from (1, 1)
    return phi[0]


@dataclass(frozen=True)
class PhiNetwork:
    """Width ``output_dim`` (m) and hidden widths of the time code."""

    output_dim: int = 8
    hidden_layers: Sequence[int] = (16, 16)

    def __post_init__(self) -> None:
        if self.output_dim < 1:
            raise ValueError("output_dim must be positive")
        if not self.hidden_layers or any(int(width) <= 0 for width in self.hidden_layers):
            raise ValueError("hidden_layers must contain positive widths")

    def init(self, seed: int, *, dtype: Any) -> Any:
        target_dtype = require_floating_dtype(dtype)
        params = init_phi(
            int(seed),
            int(self.output_dim),
            tuple(int(width) for width in self.hidden_layers),
        )
        return cast_floating(params, target_dtype)

    @staticmethod
    def apply(params: Any, time: Any) -> Any:
        return phi_apply(params, time)
