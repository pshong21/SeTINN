"""SPINN: one MLP per coordinate, combined by a rank-r product."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from jax.flatten_util import ravel_pytree

from .interface import ModelInterface, spatial_axis_names
from .mlp import axis_mlp_apply, init_axis_mlp
from .precision import jax, jnp, require_floating_dtype


@dataclass(frozen=True)
class SPINN:
    """``axis_layers`` describes one axis body, ``[1, hidden..., n_out]``."""

    spatial_dim: int
    n_out: int = 1
    rank: int = 10
    axis_layers: Sequence[int] = (1, 10, 10, 1)
    key_splits: int | None = None

    def __post_init__(self) -> None:
        dims = tuple(int(value) for value in self.axis_layers)
        if self.spatial_dim < 1 or self.n_out < 1 or self.rank < 1:
            raise ValueError("spatial_dim, n_out and rank must be positive")
        if len(dims) < 3 or dims[0] != 1 or dims[-1] not in {1, self.n_out}:
            raise ValueError("axis_layers must begin with one and end with n_out")
        if any(value <= 0 for value in dims):
            raise ValueError("all axis layer widths must be positive")
        if (self.key_splits or self.spatial_dim + 1) < self.spatial_dim + 1:
            raise ValueError("key_splits must cover every axis")

    def build(self, *, seed: int = 0, dtype: Any) -> ModelInterface:
        target_dtype = require_floating_dtype(dtype)
        n_axes = self.spatial_dim + 1
        axis_names = (*spatial_axis_names(self.spatial_dim), "t")
        output_width = self.n_out * self.rank
        hidden = tuple(int(v) for v in self.axis_layers[1:-1])
        keys = jax.random.split(jax.random.PRNGKey(seed), self.key_splits or n_axes)[:n_axes]

        params0: dict[str, Any] = {}
        for index in range(n_axes):
            axis_seed = int(jax.random.randint(keys[index], (), 0, 2**31 - 1))
            params0[axis_names[index]] = jax.tree_util.tree_map(
                lambda leaf: jnp.asarray(leaf, dtype=target_dtype),
                init_axis_mlp(axis_seed, output_width, hidden),
            )
        p0, unravel = ravel_pytree(params0)

        def features(params, index, values):
            name = axis_names[index]
            outputs = jax.vmap(
                lambda coordinate: axis_mlp_apply(params[name], coordinate)
            )(jnp.reshape(jnp.asarray(values), (-1,)))
            return outputs.reshape(-1, self.n_out, self.rank)

        def axis_features_fn(p, axis, coordinates, time=None):
            if time is not None:
                raise ValueError("SPINN factors time like any other axis")
            return features(unravel(p), axis, coordinates)

        def point_fn(p, coordinates):
            params = unravel(p)
            product = jnp.ones((self.n_out, self.rank), dtype=p.dtype)
            for index in range(n_axes):
                product = product * features(params, index, coordinates[index])[0]
            return jnp.sum(product, axis=-1)

        return ModelInterface(
            name="spinn",
            spatial_dim=self.spatial_dim,
            n_out=self.n_out,
            p0=p0,
            unravel=unravel,
            point_fn=point_fn,
            dtype=target_dtype,
            axis_features_fn=axis_features_fn,
            separable_rank=int(self.rank),
            separable_time_axis=True,
        )
