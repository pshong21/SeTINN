"""Grid evaluation through the point function."""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np


def predict_grid_pointwise(
    model: Any,
    parameters: Any,
    axes: Sequence[Any],
    *,
    dtype: Any,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Evaluate the Cartesian grid spanned by ``axes`` in chunks of points."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    axes = tuple(np.asarray(axis) for axis in axes)
    if len(axes) != model.coordinate_dim or any(axis.ndim != 1 for axis in axes):
        raise ValueError(
            f"expected {model.coordinate_dim} one-dimensional grid axes"
        )
    mesh = np.meshgrid(*axes, indexing="ij")
    flat = [coordinate.reshape(-1) for coordinate in mesh]
    batches: list[np.ndarray] = []
    for start in range(0, flat[0].size, chunk_size):
        points = np.stack(
            [coordinate[start : start + chunk_size] for coordinate in flat],
            axis=1,
        )
        value = model.predict_points(parameters, jnp.asarray(points, dtype=dtype))
        value.block_until_ready()
        batches.append(np.asarray(jax.device_get(value), dtype=np.float64))
    return np.concatenate(batches, axis=0).reshape(
        *(axis.size for axis in axes), model.n_out
    )
