"""(3+1)-D Klein--Gordon equation (``kg3d``)."""

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
    separable_grid_terms,
    tensor_grid_axes,
    time_derivatives,
)


# --- Physics ---------------------------------------------------------------


FORMULA_ID = "kg3d_manufactured_tinn_b8_v1"
FORMULA_TEXT = (
    "u=(x+y+z)cos(2t)+x*y*z*sin(2t) on [-1,1]^3 x [0,t_max]; "
    "f=u_tt-(u_xx+u_yy+u_zz)+u^2; the solution is affine in each spatial "
    "coordinate so the Laplacian vanishes and f=-4u+u^2"
)

SIDES = ("x0", "x1", "y0", "y1", "z0", "z1")
BLOCKS = ("pde", "ic_u", "ic_ut", *(f"bc_{side}" for side in SIDES))


def exact_u(x, y, z, t):
    return (x + y + z) * jnp.cos(2.0 * t) + x * y * z * jnp.sin(2.0 * t)


def exact_ut(x, y, z, t):
    return -2.0 * (x + y + z) * jnp.sin(2.0 * t) + 2.0 * x * y * z * jnp.cos(2.0 * t)


def forcing(x, y, z, t):
    """``u_tt - Laplacian(u) + u^2``, which reduces to ``-4u + u^2``."""

    u = exact_u(x, y, z, t)
    return -4.0 * u + u**2


def exact_u_numpy(x, y, z, t):
    return (x + y + z) * np.cos(2.0 * t) + x * y * z * np.sin(2.0 * t)


def formula_sha256(formula_id: str = FORMULA_ID) -> str:
    if formula_id != FORMULA_ID:
        raise ValueError(f"unknown KG3D formula_id: {formula_id!r}")
    return hashlib.sha256(FORMULA_TEXT.encode("utf-8")).hexdigest()


# --- Sampling --------------------------------------------------------------


CARTESIAN_TENSOR_GRID = "cartesian_tensor_grid"


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def sample_collocation(sampling: Mapping[str, Any], *, t_max: float) -> dict[str, np.ndarray]:
    """One float64 point set, shared byte for byte by every method of a seed."""

    n_pde = _positive_integer(sampling["n_pde"], "n_pde")
    n_ic_u = _positive_integer(sampling["n_ic_u"], "n_ic_u")
    n_ic_ut = _positive_integer(sampling["n_ic_ut"], "n_ic_ut")
    n_bc_each = _positive_integer(sampling["n_bc_each"], "n_bc_each")
    strategy = str(sampling.get("sampling_strategy"))

    if strategy == CARTESIAN_TENSOR_GRID:
        n = _positive_integer(sampling["grid_n"], "grid_n")
        nt = _positive_integer(sampling["grid_nt"], "grid_nt")
        if n_pde != n**3 * nt:
            raise ValueError("n_pde must equal grid_n^3 * grid_nt")
        if n_ic_u != n**3 or n_ic_ut != n**3:
            raise ValueError("initial counts must equal grid_n^3")
        if n_bc_each != n**2 * nt:
            raise ValueError("n_bc_each must equal grid_n^2 * grid_nt")
        space = np.linspace(-1.0, 1.0, n, dtype=np.float64)
        time = np.linspace(0.0, t_max, nt, dtype=np.float64)
        pde_x, pde_y, pde_z, pde_t = np.meshgrid(
            space, space, space, time, indexing="ij"
        )
        ic_x, ic_y, ic_z = np.meshgrid(space, space, space, indexing="ij")
        face_a, face_b, face_t = np.meshgrid(space, space, time, indexing="ij")
        points = {
            "pde_x": pde_x.ravel(order="C"),
            "pde_y": pde_y.ravel(order="C"),
            "pde_z": pde_z.ravel(order="C"),
            "pde_t": pde_t.ravel(order="C"),
            "ic_u_x": ic_x.ravel(order="C"),
            "ic_u_y": ic_y.ravel(order="C"),
            "ic_u_z": ic_z.ravel(order="C"),
            "ic_ut_x": ic_x.ravel(order="C"),
            "ic_ut_y": ic_y.ravel(order="C"),
            "ic_ut_z": ic_z.ravel(order="C"),
        }
        for side in SIDES:
            points[f"bc_{side}_a"] = face_a.ravel(order="C")
            points[f"bc_{side}_b"] = face_b.ravel(order="C")
            points[f"bc_{side}_t"] = face_t.ravel(order="C")
        return points

    raise ValueError(f"unknown sampling_strategy: {strategy!r}")


# --- Residual system -------------------------------------------------------


def _scale_map(scales: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
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


# Each face fixes one coordinate and leaves the other two free.
FACE_PLACEMENT = {
    "x0": lambda a, b, lo, hi: (lo, a, b),
    "x1": lambda a, b, lo, hi: (hi, a, b),
    "y0": lambda a, b, lo, hi: (a, lo, b),
    "y1": lambda a, b, lo, hi: (a, hi, b),
    "z0": lambda a, b, lo, hi: (a, b, lo),
    "z1": lambda a, b, lo, hi: (a, b, hi),
}


def make_system(
    u_flat,
    points: Mapping[str, Any],
    *,
    residual_scales: Mapping[str, Any],
    dtype: Any,
    jacobian_pde_chunk_size: int,
    formula_id: str = FORMULA_ID,
    model: Any = None,
    separable: bool = True,
):
    """The residual system of one model; see ``ProblemSpec.make_system``."""

    if formula_id != FORMULA_ID:
        raise ValueError(f"unknown KG3D formula_id: {formula_id!r}")
    scale = _scale_map(residual_scales)
    coll = {name: jnp.asarray(value, dtype=dtype) for name, value in points.items()}
    sizes = {
        "pde": int(coll["pde_x"].size),
        "ic_u": int(coll["ic_u_x"].size),
        "ic_ut": int(coll["ic_ut_x"].size),
        **{f"bc_{side}": int(coll[f"bc_{side}_a"].size) for side in SIDES},
    }
    chunk = min(
        _positive_integer(jacobian_pde_chunk_size, "jacobian_pde_chunk_size"),
        sizes["pde"],
    )
    zero = jnp.asarray(0.0, dtype=dtype)
    low = jnp.asarray(-1.0, dtype=dtype)
    high = jnp.asarray(1.0, dtype=dtype)

    def second(argnum):
        def derivative(p, x, y, z, t):
            return jax.grad(jax.grad(u_flat, argnums=argnum), argnums=argnum)(
                p, x, y, z, t
            )

        return derivative

    u_xx, u_yy, u_zz, u_tt = (second(index) for index in (1, 2, 3, 4))

    def pde_point(p, x, y, z, t):
        u = u_flat(p, x, y, z, t)
        laplacian = (
            u_xx(p, x, y, z, t) + u_yy(p, x, y, z, t) + u_zz(p, x, y, z, t)
        )
        return u_tt(p, x, y, z, t) - laplacian + u**2 - forcing(x, y, z, t)

    def ic_u_point(p, x, y, z):
        return u_flat(p, x, y, z, zero) - exact_u(x, y, z, 0.0)

    def ic_ut_point(p, x, y, z):
        value = jax.grad(u_flat, argnums=4)(p, x, y, z, zero)
        return value - exact_ut(x, y, z, 0.0)

    def bc_point(p, side, a, b, t):
        x, y, z = FACE_PLACEMENT[side](a, b, low, high)
        return u_flat(p, x, y, z, t) - exact_u(x, y, z, t)

    grad_pde = jax.grad(pde_point)
    grad_ic_u = jax.grad(ic_u_point)
    grad_ic_ut = jax.grad(ic_ut_point)
    grad_bc = {side: jax.grad(bc_point, argnums=0) for side in SIDES}

    def bc_block(p, side):
        return jax.vmap(lambda a, b, t: bc_point(p, side, a, b, t))(
            coll[f"bc_{side}_a"], coll[f"bc_{side}_b"], coll[f"bc_{side}_t"]
        )

    # --- Separable interior ------------------------------------------------

    grid_axes = None
    if (
        separable
        and model is not None
        and getattr(model, "supports_separable_residual", False)
    ):
        grid_axes = tensor_grid_axes(points, ("pde_x", "pde_y", "pde_z", "pde_t"))

    def _forcing_grid(axes):
        x, y, z, t = (jnp.reshape(axis, (-1,)) for axis in axes)
        return forcing(
            x[:, None, None, None],
            y[None, :, None, None],
            z[None, None, :, None],
            t[None, None, None, :],
        )

    def _complete_interior(p, axes):
        value, second_terms = separable_grid_terms(model, p, axes, field=0)
        laplacian = second_terms[0] + second_terms[1] + second_terms[2]
        return second_terms[3] - laplacian + value**2 - _forcing_grid(axes)

    def _modulated_interior(p, axes):
        """Interior residual when time acts through the modulation network."""

        x_axis, y_axis, z_axis, t_axis = axes
        spatial = (x_axis, y_axis, z_axis)

        def at_time(t):
            derivatives = [
                axis_derivatives(model, p, index, coordinates, time=t, order=2)
                for index, coordinates in enumerate(spatial)
            ]
            temporal = [
                time_derivatives(model, p, index, coordinates, t, order=2)
                for index, coordinates in enumerate(spatial)
            ]
            base = [entry[0] for entry in derivatives]
            value = combine(base, field=0)
            laplacian = None
            for axis in range(3):
                factors = list(base)
                factors[axis] = derivatives[axis][2]
                term = combine(factors, field=0)
                laplacian = term if laplacian is None else laplacian + term
            # Second time derivative: three "pure" terms and three cross terms.
            utt = None
            for axis in range(3):
                orders = [0, 0, 0]
                orders[axis] = 2
                term = combine(
                    [temporal[a][orders[a]] for a in range(3)], field=0
                )
                utt = term if utt is None else utt + term
            for first in range(3):
                for other in range(first + 1, 3):
                    orders = [0, 0, 0]
                    orders[first] = 1
                    orders[other] = 1
                    term = combine(
                        [temporal[a][orders[a]] for a in range(3)], field=0
                    )
                    utt = utt + 2.0 * term
            return value, laplacian, utt

        value, laplacian, utt = jax.vmap(at_time)(jnp.reshape(t_axis, (-1,)))
        value, laplacian, utt = (
            jnp.moveaxis(term, 0, -1) for term in (value, laplacian, utt)
        )
        return utt - laplacian + value**2 - _forcing_grid(axes)

    separable_interior_fn = None
    if grid_axes is not None:
        _interior = (
            _complete_interior if model.separable_time_axis else _modulated_interior
        )

        @jax.jit
        def separable_interior_fn(p):  # noqa: F811
            return jnp.reshape(_interior(p, grid_axes), (-1,))

    def interior_raw(p):
        if separable_interior_fn is not None:
            return separable_interior_fn(p)
        return jax.vmap(lambda x, y, z, t: pde_point(p, x, y, z, t))(
            coll["pde_x"], coll["pde_y"], coll["pde_z"], coll["pde_t"]
        )

    def raw_blocks(p):
        blocks = {
            "pde": interior_raw(p),
            "ic_u": jax.vmap(lambda x, y, z: ic_u_point(p, x, y, z))(
                coll["ic_u_x"], coll["ic_u_y"], coll["ic_u_z"]
            ),
            "ic_ut": jax.vmap(lambda x, y, z: ic_ut_point(p, x, y, z))(
                coll["ic_ut_x"], coll["ic_ut_y"], coll["ic_ut_z"]
            ),
        }
        for side in SIDES:
            blocks[f"bc_{side}"] = bc_block(p, side)
        return blocks

    @jax.jit
    def remaining_jacobians(p):
        pieces = [
            jax.vmap(grad_ic_u, in_axes=(None, 0, 0, 0))(
                p, coll["ic_u_x"], coll["ic_u_y"], coll["ic_u_z"]
            ) * (scale["ic_u"] / jnp.sqrt(sizes["ic_u"])),
            jax.vmap(grad_ic_ut, in_axes=(None, 0, 0, 0))(
                p, coll["ic_ut_x"], coll["ic_ut_y"], coll["ic_ut_z"]
            ) * (scale["ic_ut"] / jnp.sqrt(sizes["ic_ut"])),
        ]
        for side in SIDES:
            block = jax.vmap(
                lambda a, b, t, s=side: grad_bc[s](p, s, a, b, t)
            )(coll[f"bc_{side}_a"], coll[f"bc_{side}_b"], coll[f"bc_{side}_t"])
            pieces.append(
                block * (scale[f"bc_{side}"] / jnp.sqrt(sizes[f"bc_{side}"]))
            )
        return tuple(pieces)

    @jax.jit
    def remaining_residuals(p):
        raw = raw_blocks(p)
        return tuple(
            raw[name] * (scale[name] / jnp.sqrt(sizes[name])) for name in BLOCKS[1:]
        )

    @jax.jit
    def pde_chunk_system(p, xs, ys, zs, ts):
        weight = scale["pde"] / jnp.sqrt(sizes["pde"])
        values = jax.vmap(lambda x, y, z, t: pde_point(p, x, y, z, t))(xs, ys, zs, ts)
        gradients = jax.vmap(grad_pde, in_axes=(None, 0, 0, 0, 0))(p, xs, ys, zs, ts)
        return gradients * weight, values * weight

    def _interior_slices():
        return chunk_slices(sizes["pde"], chunk)

    def _interior_arguments(bounds):
        return (
            coll["pde_x"][bounds], coll["pde_y"][bounds],
            coll["pde_z"][bounds], coll["pde_t"][bounds],
        )

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

    @jax.jit
    def pde_residual_chunk(p, xs, ys, zs, ts):
        weight = scale["pde"] / jnp.sqrt(sizes["pde"])
        values = jax.vmap(lambda x, y, z, t: pde_point(p, x, y, z, t))(xs, ys, zs, ts)
        return values * weight

    def loss_fn(p):
        """Return the scalar weighted loss without materialising r."""

        def residuals():
            if separable_interior_fn is not None:
                yield interior_raw(p) * (scale["pde"] / jnp.sqrt(sizes["pde"]))
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
        "n_ic_u": n**3,
        "n_ic_ut": n**3,
        "n_bc_each": n**2 * nt,
    }


SETUP = ExperimentSetup(
    problem={"formula_id": FORMULA_ID, "t_max": 10.0},
    sampling={
        "sample_seed_offset": 11000,
        "jacobian_pde_chunk_size": 2048,
    },
    profiles={
        f"tensor_{n}cubed_x_{n}": _tensor_grid(n, n)
        for n in (16, 25)
    },
    evaluation={"nx": 24, "nt": 24},
    model={"layer_dims": [1, 10, 10, 1], "r": 5, "phi_m": 5, "phi_hidden": [10, 10]},
    tinn={"temporal_layers": [1, 10, 10, 5], "spatial_layers": [3, 20, 20, 1]},
)


# --- Specification ----------------------------------------------------------


class KleinGordon3DProblem(ProblemSpec):
    key = "kg3d"
    spatial_dim = 3
    field_names = ("u",)
    residual_names = BLOCKS
    architecture_suffix = "3d"
    method_id_suffix = "_3d"
    spinn_key_splits = 4
    setinn_key_splits = 4
    setup = SETUP

    def validate(self, problem, sampling, evaluation, residual_scales) -> None:
        if problem.get("formula_id") != FORMULA_ID:
            raise ValueError(f"kg3d requires formula_id={FORMULA_ID!r}")
        t_max = float(problem.get("t_max", float("nan")))
        if not np.isfinite(t_max) or t_max <= 0.0:
            raise ValueError("problem.t_max must be finite and positive")
        for name in ("n_pde", "n_ic_u", "n_ic_ut", "n_bc_each"):
            value = sampling.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"sampling.{name} must be a positive integer")
        strategy = str(sampling.get("sampling_strategy"))
        if strategy == CARTESIAN_TENSOR_GRID:
            n = _positive_integer(sampling.get("grid_n"), "grid_n")
            nt = _positive_integer(sampling.get("grid_nt"), "grid_nt")
            if sampling["n_pde"] != n**3 * nt:
                raise ValueError("sampling.n_pde must equal grid_n^3 * grid_nt")
            if sampling["n_ic_u"] != n**3 or sampling["n_ic_ut"] != n**3:
                raise ValueError("initial counts must equal grid_n^3")
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
            model.field(0),
            points,
            model=model,
            residual_scales=residual_scales,
            dtype=dtype,
            jacobian_pde_chunk_size=int(sampling["jacobian_pde_chunk_size"]),
            formula_id=str(problem["formula_id"]),
        )

    def evaluate(self, model, parameters, problem, evaluation, *, dtype, chunk_size=8192):
        self.validate_model(model)
        self.validate_dtype(model, dtype)
        nx = int(evaluation["nx"])
        nt = int(evaluation["nt"])
        space = np.linspace(-1.0, 1.0, nx, dtype=np.float64)
        axes = (
            space, space, space,
            np.linspace(0.0, float(problem["t_max"]), nt, dtype=np.float64),
        )
        prediction = predict_grid_pointwise(
            model, parameters, axes, dtype=dtype, chunk_size=chunk_size
        )[..., 0]
        mesh = np.meshgrid(*axes, indexing="ij")
        exact = exact_u_numpy(*mesh)
        error = prediction - exact
        denominator = float(np.linalg.norm(exact))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("KG3D reference norm must be finite and positive")
        metrics = {
            "relative_l2": float(np.linalg.norm(error) / denominator),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "max_abs_error": float(np.max(np.abs(error))),
            "evaluation_points": int(exact.size),
        }
        arrays = {
            "x": axes[0], "y": axes[1], "z": axes[2], "t": axes[3],
            "prediction": prediction, "exact": exact,
        }
        return EvaluationResult(metrics, arrays)


KLEIN_GORDON_3D = KleinGordon3DProblem()

