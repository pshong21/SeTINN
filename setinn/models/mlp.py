"""The one-dimensional MLP used for every axis body of SPINN and SeTINN."""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
from jax.nn.initializers import glorot_uniform, zeros


def init_axis_mlp(seed: int, out_dim: int, hidden: Sequence[int] = (10, 10)):
    """Initialize ``1 -> hidden... -> out_dim``: Glorot weights, zero biases."""

    if out_dim < 1:
        raise ValueError("out_dim must be positive")
    hidden = tuple(int(width) for width in hidden)
    if not hidden or any(width <= 0 for width in hidden):
        raise ValueError("hidden must contain positive widths")

    key = jax.random.PRNGKey(int(seed))
    dims = [1, *hidden]
    initializer = glorot_uniform()
    weights, biases = [], []
    for index in range(len(dims) - 1):
        key, subkey = jax.random.split(key)
        weights.append(initializer(subkey, (dims[index], dims[index + 1])))
        biases.append(zeros(subkey, (dims[index + 1],)))
    key, subkey = jax.random.split(key)
    weights.append(initializer(subkey, (dims[-1], int(out_dim))))
    biases.append(zeros(subkey, (int(out_dim),)))
    return {"Ws": weights, "bs": biases}


def axis_mlp_apply(params: Any, coordinate: Any):
    """Evaluate one axis body at a scalar coordinate."""

    hidden = jnp.reshape(coordinate, (1, 1))
    weights, biases = params["Ws"], params["bs"]
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        hidden = hidden @ weight + bias
        if index < len(weights) - 1:
            hidden = jnp.tanh(hidden)
    return hidden[0]
