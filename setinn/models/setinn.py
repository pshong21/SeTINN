"""SeTINN: SPINN bodies whose first layer is induced by a shared time code."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from jax.flatten_util import ravel_pytree

from .interface import ModelInterface, spatial_axis_names
from .mlp import axis_mlp_apply, init_axis_mlp
from .phi import PhiNetwork
from .precision import jax, jnp, require_floating_dtype

INDUCED_PARAMETERS = ("Ws", "bs")


def _induced_layer_indices(induced_layers: str, n_layers: int) -> tuple[int, ...]:
    if induced_layers == "first":
        return (0,)
    if induced_layers == "all":
        return tuple(range(n_layers))
    raise ValueError("induced_layers must be 'first' or 'all'")


@dataclass(frozen=True)
class SeTINN:
    """``axis_layers`` describes one spatial body, ``[1, hidden..., n_out]``."""

    spatial_dim: int
    n_out: int = 1
    rank: int = 10
    axis_layers: Sequence[int] = (1, 10, 10, 1)
    induced_layers: str = "first"
    modulated_axes: Sequence[int] | None = None
    phi: PhiNetwork = PhiNetwork()
    wa_init_scale: float = 0.0
    key_splits: int | None = None

    def __post_init__(self) -> None:
        dims = tuple(int(value) for value in self.axis_layers)
        if self.spatial_dim < 1 or self.n_out < 1 or self.rank < 1:
            raise ValueError("spatial_dim, n_out and rank must be positive")
        if len(dims) < 3 or dims[0] != 1 or dims[-1] not in {1, self.n_out}:
            raise ValueError("axis_layers must begin with one and end with n_out")
        if any(value <= 0 for value in dims):
            raise ValueError("all axis layer widths must be positive")
        _induced_layer_indices(self.induced_layers, len(dims) - 1)
        axes = self._axes()
        if not axes:
            raise ValueError("at least one spatial axis must be modulated")
        if len(set(axes)) != len(axes) or any(
            axis < 0 or axis >= self.spatial_dim for axis in axes
        ):
            raise ValueError("modulated_axes must be unique valid spatial indices")
        if not math.isfinite(self.wa_init_scale) or self.wa_init_scale < 0.0:
            raise ValueError("wa_init_scale must be finite and non-negative")
        if (self.key_splits or self.spatial_dim) < self.spatial_dim:
            raise ValueError("key_splits must cover every spatial axis")

    def _axes(self) -> tuple[int, ...]:
        source = (
            range(self.spatial_dim)
            if self.modulated_axes is None
            else self.modulated_axes
        )
        return tuple(int(axis) for axis in source)

    def build(self, *, seed: int = 0, dtype: Any) -> ModelInterface:
        target_dtype = require_floating_dtype(dtype)
        output_width = self.n_out * self.rank
        hidden = tuple(int(v) for v in self.axis_layers[1:-1])
        induced = _induced_layer_indices(self.induced_layers, len(hidden) + 1)
        modulated_axes = self._axes()
        axis_names = spatial_axis_names(self.spatial_dim)
        axis_keys = jax.random.split(
            jax.random.PRNGKey(seed), self.key_splits or self.spatial_dim
        )[: self.spatial_dim]

        spaces: dict[str, Any] = {}
        shapes: dict[str, dict[str, dict[str, tuple[int, ...]]]] = {}
        lifts: dict[str, dict[str, dict[str, Any]]] = {}
        phi_params = self.phi.init(seed + 100, dtype=target_dtype)
        n_lifts = len(modulated_axes) * len(induced) * len(INDUCED_PARAMETERS)
        lift_keys = iter(
            jax.random.split(jax.random.PRNGKey(seed + 200), max(1, n_lifts))
        )

        for axis in range(self.spatial_dim):
            axis_name = axis_names[axis]
            axis_seed = int(jax.random.randint(axis_keys[axis], (), 0, 2**31 - 1))
            params = jax.tree_util.tree_map(
                lambda leaf: jnp.asarray(leaf, dtype=target_dtype),
                init_axis_mlp(axis_seed, output_width, hidden),
            )
            spatial = {key: list(value) for key, value in params.items()}
            if axis in modulated_axes:
                lifts[axis_name] = {}
                shapes[axis_name] = {}
                for layer in induced:
                    layer_name = str(layer)
                    lifts[axis_name][layer_name] = {}
                    shapes[axis_name][layer_name] = {}
                    for key in INDUCED_PARAMETERS:
                        target = spatial[key][layer]
                        shapes[axis_name][layer_name][key] = tuple(target.shape)
                        wa_shape = (int(target.size), self.phi.output_dim)
                        lift_key = next(lift_keys)
                        wa = jnp.zeros(wa_shape, dtype=target.dtype)
                        if self.wa_init_scale > 0.0:
                            wa = self.wa_init_scale * jax.random.normal(
                                lift_key, wa_shape, dtype=target.dtype
                            )
                        lifts[axis_name][layer_name][key] = {
                            "Wa": wa,
                            "b": target.reshape(-1),
                        }
                        # The static copy is replaced at call time by the lift.
                        spatial[key][layer] = None
            spaces[axis_name] = spatial

        # ravel_pytree cannot carry None leaves.
        def strip(tree):
            return {
                key: [leaf for leaf in value if leaf is not None]
                for key, value in tree.items()
            }

        positions = {
            axis_name: {
                key: [index for index, leaf in enumerate(value) if leaf is None]
                for key, value in spatial.items()
            }
            for axis_name, spatial in spaces.items()
        }
        params0 = {
            "space": {name: strip(tree) for name, tree in spaces.items()},
            "phi": phi_params,
            "lift": lifts,
        }
        p0, unravel = ravel_pytree(params0)

        def assemble(params, axis_name, phi_value):
            """One axis's layer lists with the induced layers filled in."""

            stripped = params["space"][axis_name]
            rebuilt = {key: list(value) for key, value in stripped.items()}
            missing = positions[axis_name]
            for key, indices in missing.items():
                for layer_index in indices:
                    layer_name = str(layer_index)
                    lift = params["lift"][axis_name][layer_name][key]
                    value = (lift["Wa"] @ phi_value + lift["b"]).reshape(
                        shapes[axis_name][layer_name][key]
                    )
                    rebuilt[key].insert(layer_index, value)
            return rebuilt

        def axis_features(params, axis, coordinates, phi_value):
            axis_name = axis_names[axis]
            if axis_name in params["lift"]:
                spatial = assemble(params, axis_name, phi_value)
            else:
                spatial = params["space"][axis_name]
            outputs = jax.vmap(
                lambda coordinate: axis_mlp_apply(spatial, coordinate)
            )(jnp.reshape(jnp.asarray(coordinates), (-1,)))
            return outputs.reshape(-1, self.n_out, self.rank)

        def axis_features_fn(p, axis, coordinates, time=None):
            if time is None:
                raise ValueError("SeTINN axis factors depend on time; pass it")
            params = unravel(p)
            phi_value = self.phi.apply(params["phi"], time)
            return axis_features(params, axis, coordinates, phi_value)

        def point_fn(p, coordinates):
            params = unravel(p)
            phi_value = self.phi.apply(params["phi"], coordinates[-1])
            product = jnp.ones((self.n_out, self.rank), dtype=p.dtype)
            for axis in range(self.spatial_dim):
                product = product * axis_features(
                    params, axis, coordinates[axis], phi_value
                )[0]
            return jnp.sum(product, axis=-1)

        return ModelInterface(
            name=f"setinn_{self.induced_layers}",
            spatial_dim=self.spatial_dim,
            n_out=self.n_out,
            p0=p0,
            unravel=unravel,
            point_fn=point_fn,
            dtype=target_dtype,
            axis_features_fn=axis_features_fn,
            separable_rank=int(self.rank),
            separable_time_axis=False,
        )
