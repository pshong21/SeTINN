"""(3+1)-D incompressible Navier--Stokes in vorticity form (``ns3d``)."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from ..models.grid import predict_grid_pointwise
from .interface import EvaluationResult, ExperimentSetup, ProblemSpec
from .normal_equations import (
    accumulate_loss,
    accumulate_normal_equations,
    chunk_slices,
)
from .separable import (
    axis_derivatives,
    combine,
    mixed_derivatives,
    tensor_grid_axes,
)


# --- Physics ---------------------------------------------------------------


FORMULA_ID = "ns3d_taylor_green_vorticity_v1"
FORMULA_TEXT = (
    "omega_t+(u.grad)omega=(omega.grad)u+nu*Lap(omega)+f, div u=0 on "
    "[0,2pi]^3 x [0,t_max]; u=(2cos2x sin2y sin z, -sin2x cos2y sin z, "
    "-2 sin2x sin2y cos z) e^{-9 nu t}; omega=curl u; "
    "f=(-6 sin4y sin2z, -6 sin4x sin2z, 6 sin4x sin4y) e^{-18 nu t}"
)

TWO_PI = 2.0 * math.pi
SIDES = ("x0", "x1", "y0", "y1", "z0", "z1")
BLOCKS = (
    "mom_x", "mom_y", "mom_z", "cont",
    "ic_u", "ic_w",
    *(f"bc_{side}" for side in SIDES),
)


def exact_velocity(x, y, z, t, nu):
    decay = jnp.exp(-9.0 * nu * t)
    return jnp.array([
        2.0 * decay * jnp.cos(2 * x) * jnp.sin(2 * y) * jnp.sin(z),
        -1.0 * decay * jnp.sin(2 * x) * jnp.cos(2 * y) * jnp.sin(z),
        -2.0 * decay * jnp.sin(2 * x) * jnp.sin(2 * y) * jnp.cos(z),
    ])


def exact_vorticity(x, y, z, t, nu):
    decay = jnp.exp(-9.0 * nu * t)
    return jnp.array([
        -3.0 * decay * jnp.sin(2 * x) * jnp.cos(2 * y) * jnp.cos(z),
        6.0 * decay * jnp.cos(2 * x) * jnp.sin(2 * y) * jnp.cos(z),
        -6.0 * decay * jnp.cos(2 * x) * jnp.cos(2 * y) * jnp.sin(z),
    ])


def forcing(x, y, z, t, nu):
    """The vector source term, as a stack along a leading axis."""

    decay = jnp.exp(-18.0 * nu * t)
    components = [
        -6.0 * decay * jnp.sin(4 * y) * jnp.sin(2 * z),
        -6.0 * decay * jnp.sin(4 * x) * jnp.sin(2 * z),
        6.0 * decay * jnp.sin(4 * x) * jnp.sin(4 * y),
    ]
    shape = jnp.broadcast_shapes(*(term.shape for term in components))
    return jnp.stack([jnp.broadcast_to(term, shape) for term in components])


def exact_velocity_numpy(x, y, z, t, nu):
    decay = np.exp(-9.0 * nu * t)
    return np.stack([
        2.0 * decay * np.cos(2 * x) * np.sin(2 * y) * np.sin(z),
        -1.0 * decay * np.sin(2 * x) * np.cos(2 * y) * np.sin(z),
        -2.0 * decay * np.sin(2 * x) * np.sin(2 * y) * np.cos(z),
    ], axis=-1)


def exact_vorticity_numpy(x, y, z, t, nu):
    decay = np.exp(-9.0 * nu * t)
    return np.stack([
        -3.0 * decay * np.sin(2 * x) * np.cos(2 * y) * np.cos(z),
        6.0 * decay * np.cos(2 * x) * np.sin(2 * y) * np.cos(z),
        -6.0 * decay * np.cos(2 * x) * np.cos(2 * y) * np.sin(z),
    ], axis=-1)


def formula_sha256(formula_id: str = FORMULA_ID) -> str:
    if formula_id != FORMULA_ID:
        raise ValueError(f"unknown NS3D formula_id: {formula_id!r}")
    return hashlib.sha256(FORMULA_TEXT.encode("utf-8")).hexdigest()


# --- Sampling --------------------------------------------------------------


CARTESIAN_TENSOR_GRID = "cartesian_tensor_grid"


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def sample_collocation(sampling: Mapping[str, Any], *, t_max: float):
    """One float64 point set, shared byte for byte by every method of a seed."""

    n_pde = _positive_integer(sampling["n_pde"], "n_pde")
    n_ic = _positive_integer(sampling["n_ic"], "n_ic")
    n_bc_each = _positive_integer(sampling["n_bc_each"], "n_bc_each")
    strategy = str(sampling.get("sampling_strategy"))

    if strategy == CARTESIAN_TENSOR_GRID:
        n = _positive_integer(sampling["grid_n"], "grid_n")
        nt = _positive_integer(sampling["grid_nt"], "grid_nt")
        if n_pde != n**3 * nt:
            raise ValueError("n_pde must equal grid_n^3 * grid_nt")
        if n_ic != n**3:
            raise ValueError("n_ic must equal grid_n^3")
        if n_bc_each != n**2 * nt:
            raise ValueError("n_bc_each must equal grid_n^2 * grid_nt")
        space = np.linspace(0.0, TWO_PI, n, dtype=np.float64)
        time = np.linspace(0.0, t_max, nt, dtype=np.float64)
        px, py, pz, pt = np.meshgrid(space, space, space, time, indexing="ij")
        ix, iy, iz = np.meshgrid(space, space, space, indexing="ij")
        fa, fb, ft = np.meshgrid(space, space, time, indexing="ij")
        points = {
            "pde_x": px.ravel(order="C"), "pde_y": py.ravel(order="C"),
            "pde_z": pz.ravel(order="C"), "pde_t": pt.ravel(order="C"),
            "ic_x": ix.ravel(order="C"), "ic_y": iy.ravel(order="C"),
            "ic_z": iz.ravel(order="C"),
        }
        for side in SIDES:
            points[f"bc_{side}_a"] = fa.ravel(order="C")
            points[f"bc_{side}_b"] = fb.ravel(order="C")
            points[f"bc_{side}_t"] = ft.ravel(order="C")
        return points

    raise ValueError(f"unknown sampling_strategy: {strategy!r}")


# --- Residual system -------------------------------------------------------


def _scale_map(scales: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    for name in BLOCKS:
        if name not in scales:
            raise ValueError(f"residual_scales is missing {name!r}")
        value = float(scales[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid residual scale {name}={value}")
        result[name] = value
    if set(scales) != set(BLOCKS):
        raise ValueError(f"residual_scales must contain exactly {sorted(BLOCKS)}")
    return result


FACE_PLACEMENT = {
    "x0": lambda a, b, lo, hi: (lo, a, b),
    "x1": lambda a, b, lo, hi: (hi, a, b),
    "y0": lambda a, b, lo, hi: (a, lo, b),
    "y1": lambda a, b, lo, hi: (a, hi, b),
    "z0": lambda a, b, lo, hi: (a, b, lo),
    "z1": lambda a, b, lo, hi: (a, b, hi),
}


def make_system(
    uvw_flat,
    points: Mapping[str, Any],
    *,
    nu: float,
    residual_scales: Mapping[str, Any],
    dtype: Any,
    jacobian_pde_chunk_size: int,
    formula_id: str = FORMULA_ID,
    model: Any = None,
    separable: bool = True,
):
    """The residual system of one model; see ``ProblemSpec.make_system``."""

    if formula_id != FORMULA_ID:
        raise ValueError(f"unknown NS3D formula_id: {formula_id!r}")
    nu = float(nu)
    if not math.isfinite(nu) or nu <= 0:
        raise ValueError("nu must be finite and positive")
    scale = _scale_map(residual_scales)
    coll = {name: jnp.asarray(value, dtype=dtype) for name, value in points.items()}
    sizes = {
        "mom_x": int(coll["pde_x"].size), "mom_y": int(coll["pde_x"].size),
        "mom_z": int(coll["pde_x"].size), "cont": int(coll["pde_x"].size),
        "ic_u": 3 * int(coll["ic_x"].size), "ic_w": 3 * int(coll["ic_x"].size),
        **{f"bc_{s}": 3 * int(coll[f"bc_{s}_a"].size) for s in SIDES},
    }
    chunk = min(
        _positive_integer(jacobian_pde_chunk_size, "jacobian_pde_chunk_size"),
        sizes["mom_x"],
    )
    zero = jnp.asarray(0.0, dtype=dtype)
    low, high = jnp.asarray(0.0, dtype=dtype), jnp.asarray(TWO_PI, dtype=dtype)

    def velocity_at(p, coordinate, t):
        """Velocity as a function of a coordinate vector."""

        return uvw_flat(p, coordinate[0], coordinate[1], coordinate[2], t)

    def curl_of(matrix):
        """The curl combination of ``M[i][a] = d_a f^i``."""

        return jnp.array([
            matrix[2][1] - matrix[1][2],
            matrix[0][2] - matrix[2][0],
            matrix[1][0] - matrix[0][1],
        ])

    def spatial_jacobian(p, coordinate, t):
        """``J[i][a] = d u^i / d x_a``."""

        return jax.jacfwd(velocity_at, argnums=1)(p, coordinate, t)

    def laplacian_velocity(p, coordinate, t):
        """``Laplacian u^i``, the trace of the spatial Hessian."""

        hessian = jax.jacfwd(spatial_jacobian, argnums=1)(p, coordinate, t)
        return jnp.trace(hessian, axis1=1, axis2=2)

    def pde_point(p, x, y, z, t):
        """The three momentum residuals and the continuity residual."""

        coordinate = jnp.stack([x, y, z])
        u = velocity_at(p, coordinate, t)
        jacobian = spatial_jacobian(p, coordinate, t)                 # (3, 3)
        hessian = jax.jacfwd(spatial_jacobian, argnums=1)(p, coordinate, t)
        grad_laplacian = jax.jacfwd(laplacian_velocity, argnums=1)(p, coordinate, t)
        jacobian_t = jax.jacfwd(spatial_jacobian, argnums=2)(p, coordinate, t)

        w = curl_of(jacobian)
        # d omega^i / d x_a: the same contraction on each slice of the Hessian.
        grad_w = jnp.stack(
            [curl_of(hessian[:, :, a]) for a in range(3)], axis=1
        )
        w_t = curl_of(jacobian_t)
        laplacian_w = curl_of(grad_laplacian)

        advection = grad_w @ u          # (u . grad) omega
        stretching = jacobian @ w       # (omega . grad) u
        momentum = (
            w_t + advection - stretching - nu * laplacian_w
            - forcing(x, y, z, t, nu)
        )
        continuity = jnp.trace(jacobian)
        return jnp.concatenate([momentum, jnp.array([continuity])])

    def ic_u_point(p, x, y, z):
        return uvw_flat(p, x, y, z, zero) - exact_velocity(x, y, z, 0.0, nu)

    def vorticity_at(p, x, y, z, t):
        return curl_of(spatial_jacobian(p, jnp.stack([x, y, z]), t))

    def ic_w_point(p, x, y, z):
        return vorticity_at(p, x, y, z, zero) - exact_vorticity(x, y, z, 0.0, nu)

    def bc_point(p, side, a, b, t):
        x, y, z = FACE_PLACEMENT[side](a, b, low, high)
        return vorticity_at(p, x, y, z, t) - exact_vorticity(x, y, z, t, nu)

    # --- Separable interior, complete factorization only -------------------

    grid_axes = None
    if (
        separable
        and model is not None
        and getattr(model, "supports_separable_residual", False)
    ):
        grid_axes = tensor_grid_axes(points, ("pde_x", "pde_y", "pde_z", "pde_t"))

    def _separable_interior(p):
        """Momentum and continuity on the grid, from per-axis factors."""

        derivatives = [
            axis_derivatives(model, p, index, coordinates, order=3)
            for index, coordinates in enumerate(grid_axes)
        ]

        def term(orders, field):
            return combine(
                [derivatives[a][orders[a]] for a in range(4)], field=field
            )

        def velocity_field(field):
            return term((0, 0, 0, 0), field)

        def d(field, axis, order=1):
            orders = [0, 0, 0, 0]
            orders[axis] = order
            return term(tuple(orders), field)

        def d2(field, axis_a, axis_b):
            orders = [0, 0, 0, 0]
            orders[axis_a] += 1
            orders[axis_b] += 1
            return term(tuple(orders), field)

        def d3(field, axis_a, axis_b):
            """Two derivatives along axis_a, one along axis_b."""

            orders = [0, 0, 0, 0]
            orders[axis_a] += 2
            orders[axis_b] += 1
            return term(tuple(orders), field)

        u = [velocity_field(i) for i in range(3)]
        grad_u = [[d(i, a) for a in range(3)] for i in range(3)]
        w = [
            grad_u[2][1] - grad_u[1][2],
            grad_u[0][2] - grad_u[2][0],
            grad_u[1][0] - grad_u[0][1],
        ]
        # d omega^i / d x_a, through the velocity.
        pairs = ((2, 1, 1, 2), (0, 2, 2, 0), (1, 0, 0, 1))
        grad_w = [
            [d2(fa, a, ax) - d2(fb, a, bx) for a in range(3)]
            for fa, ax, fb, bx in pairs
        ]
        w_t = [d2(fa, 3, ax) - d2(fb, 3, bx) for fa, ax, fb, bx in pairs]
        laplacian = [
            sum(d3(fa, a, ax) - d3(fb, a, bx) for a in range(3))
            for fa, ax, fb, bx in pairs
        ]
        momentum = []
        for i in range(3):
            advection = sum(grad_w[i][a] * u[a] for a in range(3))
            stretching = sum(grad_u[i][a] * w[a] for a in range(3))
            momentum.append(w_t[i] + advection - stretching - nu * laplacian[i])
        x, y, z, t = (jnp.reshape(axis, (-1,)) for axis in grid_axes)
        source = forcing(
            x[:, None, None, None], y[None, :, None, None],
            z[None, None, :, None], t[None, None, None, :], nu,
        )
        momentum = [momentum[i] - source[i] for i in range(3)]
        continuity = grad_u[0][0] + grad_u[1][1] + grad_u[2][2]
        return jnp.stack(
            [jnp.reshape(term, (-1,))
             for term in (*momentum, continuity)],
            axis=1,
        )

    def _modulated_interior(p):
        """The same residual when time reaches the factors through Phi."""

        x_axis, y_axis, z_axis, t_axis = grid_axes
        spatial_axes = (x_axis, y_axis, z_axis)

        def at_time(t):
            factors = []
            for axis, coordinates in enumerate(spatial_axes):
                # Spatial orders up to three, time orders up to one.
                spatial = axis_derivatives(
                    model, p, axis, coordinates, time=t, order=3
                )
                timed = [
                    mixed_derivatives(
                        model, p, axis, coordinates, t,
                        space_order=order, time_order=1,
                    )
                    for order in range(2)
                ]
                factors.append([
                    [spatial[n]] + ([timed[n]] if n < len(timed) else [])
                    for n in range(4)
                ])

            def product(orders, times=(0, 0, 0), field=0):
                return combine(
                    [factors[a][orders[a]][times[a]] for a in range(3)],
                    field=field,
                )

            def d_time(orders, field):
                # Product rule over three factors.
                total = None
                for axis in range(3):
                    times = [0, 0, 0]
                    times[axis] = 1
                    term = product(orders, tuple(times), field)
                    total = term if total is None else total + term
                return total

            def first(field, axis):
                orders = [0, 0, 0]
                orders[axis] = 1
                return product(tuple(orders), field=field)

            def second(field, axis_a, axis_b):
                orders = [0, 0, 0]
                orders[axis_a] += 1
                orders[axis_b] += 1
                return product(tuple(orders), field=field)

            def third(field, axis_a, axis_b):
                orders = [0, 0, 0]
                orders[axis_a] += 2
                orders[axis_b] += 1
                return product(tuple(orders), field=field)

            def first_dt(field, axis):
                orders = [0, 0, 0]
                orders[axis] = 1
                return d_time(tuple(orders), field)

            u = [product((0, 0, 0), field=i) for i in range(3)]
            grad_u = [[first(i, a) for a in range(3)] for i in range(3)]
            w = [
                grad_u[2][1] - grad_u[1][2],
                grad_u[0][2] - grad_u[2][0],
                grad_u[1][0] - grad_u[0][1],
            ]
            pairs = ((2, 1, 1, 2), (0, 2, 2, 0), (1, 0, 0, 1))
            grad_w = [
                [second(fa, a, ax) - second(fb, a, bx) for a in range(3)]
                for fa, ax, fb, bx in pairs
            ]
            w_t = [first_dt(fa, ax) - first_dt(fb, bx) for fa, ax, fb, bx in pairs]
            laplacian = [
                sum(third(fa, a, ax) - third(fb, a, bx) for a in range(3))
                for fa, ax, fb, bx in pairs
            ]
            momentum = []
            for i in range(3):
                advection = sum(grad_w[i][a] * u[a] for a in range(3))
                stretching = sum(grad_u[i][a] * w[a] for a in range(3))
                momentum.append(w_t[i] + advection - stretching - nu * laplacian[i])
            continuity = grad_u[0][0] + grad_u[1][1] + grad_u[2][2]
            return (*momentum, continuity)

        stacked = jax.vmap(at_time)(jnp.reshape(t_axis, (-1,)))
        # vmap puts time first; the grid convention is (x, y, z, t).
        terms = [jnp.moveaxis(term, 0, -1) for term in stacked]
        x, y, z, t = (jnp.reshape(axis, (-1,)) for axis in grid_axes)
        source = forcing(
            x[:, None, None, None], y[None, :, None, None],
            z[None, None, :, None], t[None, None, None, :], nu,
        )
        terms = [terms[i] - source[i] for i in range(3)] + [terms[3]]
        return jnp.stack([jnp.reshape(term, (-1,)) for term in terms], axis=1)

    separable_interior_fn = None
    if grid_axes is not None:
        _interior = (
            _separable_interior if model.separable_time_axis else _modulated_interior
        )
        separable_interior_fn = jax.jit(_interior)

    def interior_values(p):
        """Raw (n_pde, 4) interior residuals: three momentum plus continuity."""

        if separable_interior_fn is not None:
            return separable_interior_fn(p)
        return jax.vmap(lambda x, y, z, t: pde_point(p, x, y, z, t))(
            coll["pde_x"], coll["pde_y"], coll["pde_z"], coll["pde_t"]
        )

    def bc_block(p, side):
        return jax.vmap(lambda a, b, t: bc_point(p, side, a, b, t))(
            coll[f"bc_{side}_a"], coll[f"bc_{side}_b"], coll[f"bc_{side}_t"]
        )

    def raw_blocks(p):
        interior = interior_values(p)
        blocks = {
            "mom_x": interior[:, 0], "mom_y": interior[:, 1],
            "mom_z": interior[:, 2], "cont": interior[:, 3],
            "ic_u": jax.vmap(lambda x, y, z: ic_u_point(p, x, y, z))(
                coll["ic_x"], coll["ic_y"], coll["ic_z"]
            ).reshape(-1),
            "ic_w": jax.vmap(lambda x, y, z: ic_w_point(p, x, y, z))(
                coll["ic_x"], coll["ic_y"], coll["ic_z"]
            ).reshape(-1),
        }
        for side in SIDES:
            blocks[f"bc_{side}"] = bc_block(p, side).reshape(-1)
        return blocks

    grad_pde = jax.jacrev(pde_point)
    grad_ic_u = jax.jacrev(ic_u_point)
    grad_ic_w = jax.jacrev(ic_w_point)
    grad_bc = jax.jacrev(bc_point)

    @jax.jit
    def remaining_jacobians(p):
        pieces = []
        for name, fn in (("ic_u", grad_ic_u), ("ic_w", grad_ic_w)):
            block = jax.vmap(fn, in_axes=(None, 0, 0, 0))(
                p, coll["ic_x"], coll["ic_y"], coll["ic_z"]
            )
            pieces.append(
                block.reshape(-1, p.shape[0])
                * (scale[name] / jnp.sqrt(sizes[name]))
            )
        for side in SIDES:
            block = jax.vmap(
                lambda a, b, t, s=side: grad_bc(p, s, a, b, t)
            )(coll[f"bc_{side}_a"], coll[f"bc_{side}_b"], coll[f"bc_{side}_t"])
            pieces.append(
                block.reshape(-1, p.shape[0])
                * (scale[f"bc_{side}"] / jnp.sqrt(sizes[f"bc_{side}"]))
            )
        return tuple(pieces)

    @jax.jit
    def remaining_residuals(p):
        raw = raw_blocks(p)
        return tuple(
            raw[name] * (scale[name] / jnp.sqrt(sizes[name])) for name in BLOCKS[4:]
        )

    pde_weight = jnp.asarray(
        [scale[name] / math.sqrt(sizes[name])
         for name in ("mom_x", "mom_y", "mom_z", "cont")],
        dtype=dtype,
    )

    @jax.jit
    def pde_chunk_system(p, xs, ys, zs, ts):
        values = jax.vmap(lambda x, y, z, t: pde_point(p, x, y, z, t))(xs, ys, zs, ts)
        gradients = jax.vmap(grad_pde, in_axes=(None, 0, 0, 0, 0))(p, xs, ys, zs, ts)
        residual = (values * pde_weight[None, :]).reshape(-1)
        jacobian = (gradients * pde_weight[None, :, None]).reshape(-1, p.shape[0])
        return jacobian, residual

    @jax.jit
    def pde_residual_chunk(p, xs, ys, zs, ts):
        values = jax.vmap(lambda x, y, z, t: pde_point(p, x, y, z, t))(xs, ys, zs, ts)
        return (values * pde_weight[None, :]).reshape(-1)

    def _interior_slices():
        return chunk_slices(sizes["mom_x"], chunk)

    def _interior_arguments(bounds):
        return (coll["pde_x"][bounds], coll["pde_y"][bounds],
                coll["pde_z"][bounds], coll["pde_t"][bounds])

    def normal_equations_fn(p):
        """Return ``(J^T J, J^T r, loss)`` without materialising J or r."""

        def blocks():
            for bounds in _interior_slices():
                yield pde_chunk_system(p, *_interior_arguments(bounds))
            for jacobian, residual in zip(
                remaining_jacobians(p), remaining_residuals(p)
            ):
                yield jacobian, residual

        return accumulate_normal_equations(p, blocks())

    def loss_fn(p):
        def residuals():
            if separable_interior_fn is not None:
                yield jnp.reshape(
                    interior_values(p) * pde_weight[None, :], (-1,)
                )
            else:
                for bounds in _interior_slices():
                    yield pde_residual_chunk(p, *_interior_arguments(bounds))
            for residual in remaining_residuals(p):
                yield residual

        return accumulate_loss(p, residuals())

    def block_mse_fn(p):
        raw = raw_blocks(p)
        return {name: jnp.mean(raw[name] ** 2) for name in BLOCKS}

    return {
        "normal_equations_fn": normal_equations_fn,
        "loss_fn": loss_fn,
        "block_mse_fn": block_mse_fn,
        "residual_order": BLOCKS,
        "residual_scales": scale,
    }


# --- Experiment defaults ----------------------------------------------------


def _tensor_grid(n: int, nt: int) -> dict[str, Any]:
    """An ``n^3`` spatial grid times ``nt`` time levels."""

    return {
        "sampling_strategy": CARTESIAN_TENSOR_GRID,
        "grid_n": n,
        "grid_nt": nt,
        "n_pde": n**3 * nt,
        "n_ic": n**3,
        "n_bc_each": n**2 * nt,
    }


SETUP = ExperimentSetup(
    problem={"formula_id": FORMULA_ID, "nu": 0.05, "t_max": 5.0},
    sampling={
        "sample_seed_offset": 17000,
        "jacobian_pde_chunk_size": 512,
    },
    profiles={
        f"tensor_{n}cubed_x_{n}": _tensor_grid(n, n)
        for n in (16, 25)
    },
    evaluation={"nx": 24, "nt": 24},
    model={"layer_dims": [1, 10, 10, 3], "r": 5, "phi_m": 5, "phi_hidden": [10, 10]},
    tinn={"temporal_layers": [1, 10, 10, 5], "spatial_layers": [3, 20, 20, 3]},
)


# --- Specification ----------------------------------------------------------


class NavierStokes3DProblem(ProblemSpec):
    key = "ns3d"
    spatial_dim = 3
    field_names = ("u", "v", "w")
    residual_names = BLOCKS
    architecture_suffix = "ns3d"
    method_id_suffix = "_ns3d"
    spinn_key_splits = 4
    setinn_key_splits = 4
    setup = SETUP

    def validate(self, problem, sampling, evaluation, residual_scales) -> None:
        if problem.get("formula_id") != FORMULA_ID:
            raise ValueError(f"ns3d requires formula_id={FORMULA_ID!r}")
        for name in ("nu", "t_max"):
            value = float(problem.get(name, float("nan")))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"problem.{name} must be finite and positive")
        for name in ("n_pde", "n_ic", "n_bc_each"):
            value = sampling.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"sampling.{name} must be a positive integer")
        strategy = str(sampling.get("sampling_strategy"))
        if strategy == CARTESIAN_TENSOR_GRID:
            n = _positive_integer(sampling.get("grid_n"), "grid_n")
            nt = _positive_integer(sampling.get("grid_nt"), "grid_nt")
            if sampling["n_pde"] != n**3 * nt:
                raise ValueError("sampling.n_pde must equal grid_n^3 * grid_nt")
            if sampling["n_ic"] != n**3:
                raise ValueError("sampling.n_ic must equal grid_n^3")
            if sampling["n_bc_each"] != n**2 * nt:
                raise ValueError("sampling.n_bc_each must equal grid_n^2 * grid_nt")
        else:
            raise ValueError(f"unsupported sampling_strategy {strategy!r}")
        self.validate_sampling_controls(sampling)
        self.validate_residual_scales(residual_scales)
        for name in ("nx", "nt"):
            value = evaluation.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"evaluation.{name} must be an integer >= 2")

    def sample_collocation(self, sampling, problem):
        return sample_collocation(sampling, t_max=float(problem["t_max"]))

    def formula_sha256(self, problem) -> str:
        return formula_sha256(str(problem["formula_id"]))

    def make_system(self, model, points, problem, sampling, residual_scales, *, dtype):
        self.validate_model(model)
        self.validate_dtype(model, dtype)
        return make_system(
            model.fields(),
            points,
            model=model,
            nu=float(problem["nu"]),
            residual_scales=residual_scales,
            dtype=dtype,
            jacobian_pde_chunk_size=int(sampling["jacobian_pde_chunk_size"]),
            formula_id=str(problem["formula_id"]),
        )

    def evaluate(self, model, parameters, problem, evaluation, *, dtype, chunk_size=8192):
        """Relative L2 of the vorticity, as both source papers report it."""

        self.validate_model(model)
        self.validate_dtype(model, dtype)
        nu = float(problem["nu"])
        nx, nt = int(evaluation["nx"]), int(evaluation["nt"])
        space = np.linspace(0.0, TWO_PI, nx, dtype=np.float64)
        axes = (space, space, space,
                np.linspace(0.0, float(problem["t_max"]), nt, dtype=np.float64))

        velocity = predict_grid_pointwise(
            model, parameters, axes, dtype=dtype, chunk_size=chunk_size
        )
        mesh = np.meshgrid(*axes, indexing="ij")
        exact_u = exact_velocity_numpy(*mesh, nu)
        exact_w = exact_vorticity_numpy(*mesh, nu)

        # Vorticity by automatic differentiation, as in the residual.
        uvw_flat = model.fields()

        def vorticity_point(p, x, y, z, t):
            jacobian = jax.jacfwd(
                lambda c: uvw_flat(p, c[0], c[1], c[2], t)
            )(jnp.stack([x, y, z]))
            return jnp.array([
                jacobian[2][1] - jacobian[1][2],
                jacobian[0][2] - jacobian[2][0],
                jacobian[1][0] - jacobian[0][1],
            ])

        batched = jax.jit(jax.vmap(vorticity_point, in_axes=(None, 0, 0, 0, 0)))
        flat = [jnp.asarray(component.reshape(-1), dtype=dtype) for component in mesh]
        pieces = []
        for start in range(0, flat[0].size, chunk_size):
            bounds = slice(start, start + chunk_size)
            piece = batched(parameters, *(component[bounds] for component in flat))
            pieces.append(np.asarray(jax.device_get(piece)))
        predicted_w = np.concatenate(pieces, axis=0).reshape(exact_w.shape)

        def relative(produced, reference):
            denominator = float(np.linalg.norm(reference))
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise ValueError("reference norm must be finite and positive")
            return float(np.linalg.norm(produced - reference) / denominator)

        metrics = {
            "relative_l2": relative(predicted_w, exact_w),
            "relative_l2_velocity": relative(velocity, exact_u),
            "rmse": float(np.sqrt(np.mean((predicted_w - exact_w) ** 2))),
            "max_abs_error": float(np.max(np.abs(predicted_w - exact_w))),
            "evaluation_points": int(exact_w.size),
        }
        arrays = {
            "x": axes[0], "y": axes[1], "z": axes[2], "t": axes[3],
            "prediction": predicted_w, "exact": exact_w,
            "prediction_velocity": velocity, "exact_velocity": exact_u,
        }
        return EvaluationResult(metrics, arrays)


NAVIER_STOKES_3D = NavierStokes3DProblem()

