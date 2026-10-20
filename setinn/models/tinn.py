"""TINN: a spatial MLP whose layers are induced by a time code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from flax import linen as nn
from jax.flatten_util import ravel_pytree

from .interface import ModelInterface
from .precision import cast_floating, jax, jnp, require_floating_dtype


class TINNNetwork(nn.Module):
    """TINN with a ``d``-coordinate spatial backbone and ``n_out`` outputs."""

    temporal_layers: Sequence[int] = (1, 10, 10, 5)
    spatial_layers: Sequence[int] = (1, 20, 20, 1)

    @nn.compact
    def __call__(self, xt):
        d = int(self.spatial_layers[0])
        n_out = int(self.spatial_layers[-1])
        n_hidden = len(self.spatial_layers) - 2

        xt = jnp.asarray(xt)
        scalar = xt.ndim == 1
        if scalar:
            xt = xt.reshape(1, d + 1)
        x = xt[:, :d]
        t = xt[:, d : d + 1]

        temporal = t
        for index in range(len(self.temporal_layers) - 2):
            in_dim = int(self.temporal_layers[index])
            out_dim = int(self.temporal_layers[index + 1])
            weight = self.param(
                f"tW{index}", nn.initializers.xavier_uniform(), (in_dim, out_dim)
            )
            bias = self.param(f"tb{index}", nn.initializers.zeros, (out_dim,))
            temporal = jnp.tanh(temporal @ weight + bias)
        weight_out = self.param(
            "tWout",
            nn.initializers.xavier_uniform(),
            (int(self.temporal_layers[-2]), int(self.temporal_layers[-1])),
        )
        temporal = temporal @ weight_out
        alpha = self.param(
            "t_alpha", nn.initializers.ones, (1, int(self.temporal_layers[-1]))
        )
        temporal = (1.0 - alpha) * t + alpha * temporal

        z = x
        for index in range(n_hidden):
            in_dim = int(self.spatial_layers[index])
            out_dim = int(self.spatial_layers[index + 1])
            modulated_weight = self.param(
                f"aW{index}", nn.initializers.zeros, (in_dim, out_dim)
            )
            modulated_bias = self.param(
                f"ab{index}", nn.initializers.xavier_uniform(), (1, out_dim)
            )
            base_weight = self.param(
                f"bW{index}", nn.initializers.xavier_uniform(), (in_dim, out_dim)
            )
            base_bias = self.param(f"bb{index}", nn.initializers.zeros, (out_dim,))
            weight_code = temporal[..., 2 * index : 2 * index + 1]
            bias_code = temporal[..., 2 * index + 1 : 2 * index + 2]
            z = jnp.tanh(
                (z @ modulated_weight) * weight_code
                + z @ base_weight
                + modulated_bias * bias_code
                + base_bias
            )

        in_dim = int(self.spatial_layers[-2])
        out_dim = int(self.spatial_layers[-1])
        modulated_out = self.param(
            "aWout", nn.initializers.zeros, (in_dim, out_dim)
        )
        base_out = self.param(
            "bWout", nn.initializers.xavier_uniform(), (in_dim, out_dim)
        )
        output_code = temporal[..., -1:]
        prediction = (z @ modulated_out) * output_code + z @ base_out
        if n_out == 1:
            prediction = jnp.squeeze(prediction, axis=-1)
        return prediction[0] if scalar else prediction


@dataclass(frozen=True)
class TINN:
    """``spatial_layers`` is ``[d, hidden..., n_out]``."""

    temporal_layers: Sequence[int] = (1, 10, 10, 5)
    spatial_layers: Sequence[int] = (1, 20, 20, 1)

    def build(self, *, seed: int = 0, dtype: Any) -> ModelInterface:
        target_dtype = require_floating_dtype(dtype)
        temporal_layers = tuple(int(value) for value in self.temporal_layers)
        spatial_layers = tuple(int(value) for value in self.spatial_layers)
        if len(temporal_layers) < 3 or len(spatial_layers) < 3:
            raise ValueError(
                "temporal and spatial networks each require input, hidden and "
                "output layers"
            )
        if temporal_layers[0] != 1:
            raise ValueError("the temporal network consumes the scalar time input")
        if any(value <= 0 for value in (*temporal_layers, *spatial_layers)):
            raise ValueError("all layer widths must be positive")
        d = spatial_layers[0]
        n_out = spatial_layers[-1]
        required = 2 * (len(spatial_layers) - 2) + 1
        if temporal_layers[-1] != required:
            raise ValueError(
                f"temporal_layers[-1] must equal 2H + 1 = {required} for "
                f"H = {len(spatial_layers) - 2} hidden spatial layers, "
                f"got {temporal_layers[-1]}"
            )

        network = TINNNetwork(
            temporal_layers=temporal_layers, spatial_layers=spatial_layers
        )
        params0 = network.init(jax.random.PRNGKey(seed), jnp.ones((1, d + 1)))["params"]
        params0 = cast_floating(params0, target_dtype)
        p0, unravel = ravel_pytree(params0)

        def point_fn(p, coordinates):
            value = network.apply({"params": unravel(p)}, coordinates)
            return jnp.asarray(value).reshape(n_out)

        return ModelInterface(
            name="tinn",
            spatial_dim=d,
            n_out=n_out,
            p0=p0,
            unravel=unravel,
            point_fn=point_fn,
            dtype=target_dtype,
        )
