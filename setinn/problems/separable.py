"""Tensor-grid residuals for separable models (SPINN and SeTINN)."""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp


def _scalar_factor(model, p, axis: int, time):
    """A single-coordinate view of one axis factor, shape ``(n_out, rank)``."""

    def factor(coordinate):
        values = model.axis_features_fn(
            p, axis, jnp.reshape(coordinate, (1,)), time
        )
        return jnp.reshape(values, (model.n_out, int(model.separable_rank)))

    return factor


def axis_derivatives(model, p, axis: int, coordinates, time=None, order: int = 2):
    """Return ``[f, f', ...]`` up to ``order`` for one axis."""

    factor = _scalar_factor(model, p, axis, time)
    functions = [factor]
    for _ in range(order):
        functions.append(jax.jacfwd(functions[-1]))
    coordinates = jnp.reshape(coordinates, (-1,))
    return [jax.vmap(function)(coordinates) for function in functions]


def time_derivatives(model, p, axis: int, coordinates, time, order: int = 2):
    """Return ``[f, df/dt, ...]`` for one spatial axis of a SeTINN model."""

    if model.separable_time_axis:
        raise ValueError("time_derivatives is only defined when time is not factored")

    def batch(t):
        return model.axis_features_fn(p, axis, coordinates, t)

    functions = [batch]
    for _ in range(order):
        functions.append(jax.jacfwd(functions[-1]))
    return [function(time) for function in functions]


def mixed_derivatives(
    model, p, axis: int, coordinates, time, space_order: int = 1,
    time_order: int = 1,
):
    """``d^m/dt^m ( d^n/dx^n f )`` for one spatial axis of a SeTINN model."""

    if model.separable_time_axis:
        raise ValueError(
            "mixed_derivatives is only defined when time is not factored; when "
            "time is a separable axis its derivative is just another axis derivative"
        )
    coordinates = jnp.reshape(coordinates, (-1,))

    def spatial(t):
        return axis_derivatives(
            model, p, axis, coordinates, time=t, order=space_order
        )[space_order]

    function = spatial
    for _ in range(time_order):
        function = jax.jacfwd(function)
    return function(time)


def _broadcast(values, axis: int, n_axes: int, field: int):
    """Place ``(n, n_out, rank)`` factors on their grid axis for broadcasting."""

    selected = values[:, field, :]
    shape = [1] * n_axes + [selected.shape[-1]]
    shape[axis] = selected.shape[0]
    return jnp.reshape(selected, shape)


def combine(factors_per_axis: Sequence[Any], *, field: int = 0):
    """Contract one choice of factor per axis into a grid, summing over rank."""

    n_axes = len(factors_per_axis)
    product = None
    for axis, values in enumerate(factors_per_axis):
        placed = _broadcast(values, axis, n_axes, field)
        product = placed if product is None else product * placed
    assert product is not None
    return jnp.sum(product, axis=-1)


def separable_grid_terms(
    model,
    p,
    axes: Sequence[Any],
    *,
    field: int = 0,
    orders: Sequence[int] | None = None,
):
    """Value and per-axis second derivatives of a complete factorization."""

    if not model.separable_time_axis:
        raise ValueError(
            "separable_grid_terms requires time as a separable axis; "
            "SeTINN factors must be combined per time slice"
        )
    wanted = list(orders) if orders is not None else list(range(len(axes)))
    derivatives = [
        axis_derivatives(model, p, axis, coordinates, order=2)
        for axis, coordinates in enumerate(axes)
    ]
    base = [entry[0] for entry in derivatives]
    value = combine(base, field=field)
    second = {}
    for axis in wanted:
        factors = list(base)
        factors[axis] = derivatives[axis][2]
        second[axis] = combine(factors, field=field)
    return value, second


def tensor_grid_axes(points: dict, names: Sequence[str]) -> tuple[Any, ...] | None:
    """Recover the 1-D axes of a flattened Cartesian grid, or None."""

    columns = [jnp.reshape(jnp.asarray(points[name]), (-1,)) for name in names]
    total = columns[0].shape[0]
    if any(column.shape[0] != total for column in columns):
        return None
    axes = []
    stride = total
    for column in columns:
        # C-order meshgrid.
        unique = jnp.unique(column)
        count = int(unique.shape[0])
        if count < 1 or total % count:
            return None
        stride //= count
        axes.append(unique)
    if stride != 1:
        return None
    # The points must be exactly this meshgrid.
    mesh = jnp.meshgrid(*axes, indexing="ij")
    if not all(
        bool(jnp.array_equal(column, jnp.reshape(coordinate, (-1,))))
        for column, coordinate in zip(columns, mesh)
    ):
        return None
    return tuple(axes)
