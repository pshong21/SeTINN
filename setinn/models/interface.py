"""The point-model interface shared by TINN, SPINN and SeTINN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .precision import jax, jnp, require_floating_dtype


def spatial_axis_names(spatial_dim: int) -> tuple[str, ...]:
    """``x, y, z`` for up to three dimensions, ``x_0, x_1, ...`` beyond."""

    if spatial_dim <= 3:
        return ("x", "y", "z")[:spatial_dim]
    return tuple(f"x_{index}" for index in range(spatial_dim))


@dataclass(frozen=True)
class ModelInterface:
    """A model as a flat parameter vector and a point function."""

    name: str
    spatial_dim: int
    n_out: int
    p0: Any
    unravel: Callable[[Any], Any]
    point_fn: Callable[[Any, Any], Any]
    dtype: Any
    axis_features_fn: Callable[[Any, int, Any, Any], Any] | None = None
    separable_rank: int | None = None
    separable_time_axis: bool = False

    def __post_init__(self) -> None:
        if self.spatial_dim < 1:
            raise ValueError("spatial_dim must be at least one")
        if self.n_out < 1:
            raise ValueError("n_out must be at least one")
        if self.axis_features_fn is not None and self.separable_rank is None:
            raise ValueError("axis_features_fn requires separable_rank")
        if self.separable_rank is not None and self.separable_rank < 1:
            raise ValueError("separable_rank must be positive")
        dtype = require_floating_dtype(self.dtype)
        object.__setattr__(self, "dtype", dtype)
        flat = jnp.asarray(self.p0)
        if flat.ndim != 1:
            raise ValueError(f"p0 must be flat, got shape {flat.shape}")
        if flat.dtype != dtype:
            raise ValueError(f"p0 dtype {flat.dtype} does not match model dtype {dtype}")

    @property
    def coordinate_dim(self) -> int:
        return self.spatial_dim + 1

    @property
    def n_params(self) -> int:
        return int(self.p0.shape[0])

    @property
    def supports_separable_residual(self) -> bool:
        return self.axis_features_fn is not None

    def predict_point(self, p: Any, coordinates: Any) -> Any:
        coordinates = jnp.asarray(coordinates, dtype=self.dtype)
        if coordinates.ndim != 1 or coordinates.shape[0] != self.coordinate_dim:
            raise ValueError(
                f"coordinates must have shape ({self.coordinate_dim},), "
                f"got {coordinates.shape}"
            )
        return jnp.asarray(
            self.point_fn(p, coordinates), dtype=self.dtype
        ).reshape(self.n_out)

    def predict_points(self, p: Any, coordinates: Any) -> Any:
        coordinates = jnp.asarray(coordinates, dtype=self.dtype)
        if coordinates.ndim != 2 or coordinates.shape[1] != self.coordinate_dim:
            raise ValueError(
                f"coordinates must have shape (N, {self.coordinate_dim}), "
                f"got {coordinates.shape}"
            )
        return jax.vmap(lambda point: self.predict_point(p, point))(coordinates)

    def field(self, index: int = 0) -> Callable[..., Any]:
        """One output as a scalar function ``(p, x_1, ..., x_d, t)``, for autodiff."""

        if not 0 <= index < self.n_out:
            raise ValueError(f"field index must be in [0, {self.n_out}), got {index}")

        def scalar(p: Any, *coordinates: Any) -> Any:
            if len(coordinates) != self.coordinate_dim:
                raise ValueError(
                    f"field expects {self.coordinate_dim} scalar coordinates, "
                    f"got {len(coordinates)}"
                )
            value = self.point_fn(
                p, jnp.asarray(jnp.stack(coordinates), dtype=self.dtype)
            )
            return jnp.asarray(value, dtype=self.dtype).reshape(self.n_out)[index]

        return scalar

    def fields(self) -> Callable[..., Any]:
        """All outputs as a vector function ``(p, x_1, ..., x_d, t)``."""

        def vector(p: Any, *coordinates: Any) -> Any:
            if len(coordinates) != self.coordinate_dim:
                raise ValueError(
                    f"fields expects {self.coordinate_dim} scalar coordinates, "
                    f"got {len(coordinates)}"
                )
            value = self.point_fn(
                p, jnp.asarray(jnp.stack(coordinates), dtype=self.dtype)
            )
            return jnp.asarray(value, dtype=self.dtype).reshape(self.n_out)

        return vector
