"""(2+1)-D Klein--Gordon equation (``kg2d``)."""

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


FORMULA_ID = "kg2d_axis_anisotropic_manufactured_v2"
FORMULA_TEXT = (
    "u=tanh((2+2.5t)(x-0.5))*sin(pi*y); "
    "f=u_tt-kappa_x*u_xx-kappa_y*u_yy+u^2"
)

K0 = 2.0
K1 = 2.5


def exact_u(x, y, t):
    z = (K0 + K1 * t) * (x - 0.5)
    return jnp.tanh(z) * jnp.sin(jnp.pi * y)


def exact_ut(x, y, t):
    z = (K0 + K1 * t) * (x - 0.5)
    sech2 = 1.0 - jnp.tanh(z) ** 2
    return K1 * (x - 0.5) * sech2 * jnp.sin(jnp.pi * y)


def exact_derivatives(x, y, t):
    distance = x - 0.5
    k = K0 + K1 * t
    z = k * distance
    tanh_z = jnp.tanh(z)
    sech2 = 1.0 - tanh_z**2
    sin_y = jnp.sin(jnp.pi * y)
    u = tanh_z * sin_y
    utt = -2.0 * tanh_z * sech2 * (K1 * distance) ** 2 * sin_y
    uxx = -2.0 * tanh_z * sech2 * k**2 * sin_y
    uyy = -(jnp.pi**2) * u
    return u, utt, uxx, uyy


def forcing(x, y, t, kappa_x: float, kappa_y: float):
    u, utt, uxx, uyy = exact_derivatives(x, y, t)
    return utt - kappa_x * uxx - kappa_y * uyy + u**2


def exact_u_numpy(x, y, t):
    return np.tanh((K0 + K1 * t) * (x - 0.5)) * np.sin(np.pi * y)


def formula_sha256(formula_id: str = FORMULA_ID) -> str:
    if formula_id != FORMULA_ID:
        raise ValueError(f"unknown KG2D formula_id: {formula_id!r}")
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
        nx = _positive_integer(sampling["grid_nx"], "grid_nx")
        ny = _positive_integer(sampling["grid_ny"], "grid_ny")
        nt = _positive_integer(sampling["grid_nt"], "grid_nt")
        if n_pde != nx * ny * nt:
            raise ValueError("n_pde must equal grid_nx * grid_ny * grid_nt")
        if n_ic_u != nx * ny or n_ic_ut != nx * ny:
            raise ValueError("n_ic_u and n_ic_ut must equal grid_nx * grid_ny")
        if n_bc_each != nx * nt or nx != ny:
            raise ValueError(
                "square-grid KG requires n_bc_each=grid_nx*grid_nt and grid_nx=grid_ny"
            )
        x = np.linspace(0.0, 1.0, nx, dtype=np.float64)
        y = np.linspace(0.0, 1.0, ny, dtype=np.float64)
        t = np.linspace(0.0, t_max, nt, dtype=np.float64)
        pde_x, pde_y, pde_t = np.meshgrid(x, y, t, indexing="ij")
        ic_x, ic_y = np.meshgrid(x, y, indexing="ij")
        y_face, y_time = np.meshgrid(y, t, indexing="ij")
        x_face, x_time = np.meshgrid(x, t, indexing="ij")
        return {
            "pde_x": pde_x.ravel(order="C"),
            "pde_y": pde_y.ravel(order="C"),
            "pde_t": pde_t.ravel(order="C"),
            "ic_u_x": ic_x.ravel(order="C"),
            "ic_u_y": ic_y.ravel(order="C"),
            "ic_ut_x": ic_x.ravel(order="C"),
            "ic_ut_y": ic_y.ravel(order="C"),
            "bc_x0_s": y_face.ravel(order="C"),
            "bc_x0_t": y_time.ravel(order="C"),
            "bc_x1_s": y_face.ravel(order="C"),
            "bc_x1_t": y_time.ravel(order="C"),
            "bc_y0_s": x_face.ravel(order="C"),
            "bc_y0_t": x_time.ravel(order="C"),
            "bc_y1_s": x_face.ravel(order="C"),
            "bc_y1_t": x_time.ravel(order="C"),
        }
    raise ValueError(f"unknown sampling_strategy: {strategy!r}")


# --- Residual system -------------------------------------------------------


def _scale_map(scales: Mapping[str, Any]) -> dict[str, float]:
    names = ("pde", "ic_u", "ic_ut", "bc_x0", "bc_x1", "bc_y0", "bc_y1")
    result: dict[str, float] = {}
    for name in names:
        if name not in scales:
            raise ValueError(f"residual_scales is missing {name!r}")
        value = float(scales[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid residual scale {name}={value}")
        result[name] = value
    if set(scales) != set(names):
        raise ValueError(f"residual_scales must contain exactly {names}")
    return result


def make_system(
    u_flat,
    points: Mapping[str, Any],
    *,
    kappa_x: float,
    kappa_y: float,
    residual_scales: Mapping[str, Any],
    dtype: Any,
    jacobian_pde_chunk_size: int,
    formula_id: str = FORMULA_ID,
    model: Any = None,
    separable: bool = True,
):
    """The residual system of one model; see ``ProblemSpec.make_system``."""

    if formula_id != FORMULA_ID:
        raise ValueError(f"unknown KG2D formula_id: {formula_id!r}")
    formula_exact_u = exact_u
    formula_exact_ut = exact_ut
    formula_forcing = forcing
    scale = _scale_map(residual_scales)
    coll = {name: jnp.asarray(value, dtype=dtype) for name, value in points.items()}
    order = ("pde", "ic_u", "ic_ut", "bc_x0", "bc_x1", "bc_y0", "bc_y1")
    sizes = {
        "pde": int(coll["pde_x"].size),
        "ic_u": int(coll["ic_u_x"].size),
        "ic_ut": int(coll["ic_ut_x"].size),
        "bc_x0": int(coll["bc_x0_s"].size),
        "bc_x1": int(coll["bc_x1_s"].size),
        "bc_y0": int(coll["bc_y0_s"].size),
        "bc_y1": int(coll["bc_y1_s"].size),
    }
    chunk = min(_positive_integer(jacobian_pde_chunk_size, "jacobian_pde_chunk_size"), sizes["pde"])

    def ux(p, x, y, t):
        return jax.grad(u_flat, argnums=1)(p, x, y, t)

    def uy(p, x, y, t):
        return jax.grad(u_flat, argnums=2)(p, x, y, t)

    def pde_point(p, x, y, t):
        u = u_flat(p, x, y, t)
        utt = jax.grad(jax.grad(u_flat, argnums=3), argnums=3)(p, x, y, t)
        uxx = jax.grad(ux, argnums=1)(p, x, y, t)
        uyy = jax.grad(uy, argnums=2)(p, x, y, t)
        return utt - kappa_x * uxx - kappa_y * uyy + u**2 - formula_forcing(
            x, y, t, kappa_x, kappa_y
        )

    def ic_u_point(p, x, y):
        return u_flat(p, x, y, jnp.asarray(0.0, dtype=dtype)) - formula_exact_u(x, y, 0.0)

    def ic_ut_point(p, x, y):
        value = jax.grad(u_flat, argnums=3)(p, x, y, jnp.asarray(0.0, dtype=dtype))
        return value - formula_exact_ut(x, y, 0.0)

    def bc_x_point(p, side, y, t):
        return u_flat(p, side, y, t) - formula_exact_u(side, y, t)

    def bc_y_point(p, side, x, t):
        return u_flat(p, x, side, t) - formula_exact_u(x, side, t)

    grad_pde = jax.grad(pde_point)
    grad_ic_u = jax.grad(ic_u_point)
    grad_ic_ut = jax.grad(ic_ut_point)
    grad_bc_x = jax.grad(bc_x_point)
    grad_bc_y = jax.grad(bc_y_point)

    @jax.jit
    def remaining_jacobians(p):
        result = {
            "ic_u": jax.vmap(grad_ic_u, in_axes=(None, 0, 0))(
                p, coll["ic_u_x"], coll["ic_u_y"]
            ) * (scale["ic_u"] / jnp.sqrt(sizes["ic_u"])),
            "ic_ut": jax.vmap(grad_ic_ut, in_axes=(None, 0, 0))(
                p, coll["ic_ut_x"], coll["ic_ut_y"]
            ) * (scale["ic_ut"] / jnp.sqrt(sizes["ic_ut"])),
            "bc_x0": jax.vmap(grad_bc_x, in_axes=(None, None, 0, 0))(
                p, 0.0, coll["bc_x0_s"], coll["bc_x0_t"]
            ) * (scale["bc_x0"] / jnp.sqrt(sizes["bc_x0"])),
            "bc_x1": jax.vmap(grad_bc_x, in_axes=(None, None, 0, 0))(
                p, 1.0, coll["bc_x1_s"], coll["bc_x1_t"]
            ) * (scale["bc_x1"] / jnp.sqrt(sizes["bc_x1"])),
            "bc_y0": jax.vmap(grad_bc_y, in_axes=(None, None, 0, 0))(
                p, 0.0, coll["bc_y0_s"], coll["bc_y0_t"]
            ) * (scale["bc_y0"] / jnp.sqrt(sizes["bc_y0"])),
            "bc_y1": jax.vmap(grad_bc_y, in_axes=(None, None, 0, 0))(
                p, 1.0, coll["bc_y1_s"], coll["bc_y1_t"]
            ) * (scale["bc_y1"] / jnp.sqrt(sizes["bc_y1"])),
        }
        return tuple(result[name] for name in order[1:])

    # --- Memory-bounded path ---------------------------------------------

    @jax.jit
    def pde_chunk_system(p, xs, ys, ts):
        """One interior chunk as an aligned (J_chunk, r_chunk) pair."""

        weight = scale["pde"] / jnp.sqrt(sizes["pde"])
        values = jax.vmap(lambda x, y, t: pde_point(p, x, y, t))(xs, ys, ts)
        gradients = jax.vmap(grad_pde, in_axes=(None, 0, 0, 0))(p, xs, ys, ts)
        return gradients * weight, values * weight

    @jax.jit
    def pde_residual_chunk(p, xs, ys, ts):
        weight = scale["pde"] / jnp.sqrt(sizes["pde"])
        values = jax.vmap(lambda x, y, t: pde_point(p, x, y, t))(xs, ys, ts)
        return values * weight

    @jax.jit
    def remaining_raw_blocks(p):
        return {
            "ic_u": jax.vmap(lambda x, y: ic_u_point(p, x, y))(
                coll["ic_u_x"], coll["ic_u_y"]
            ),
            "ic_ut": jax.vmap(lambda x, y: ic_ut_point(p, x, y))(
                coll["ic_ut_x"], coll["ic_ut_y"]
            ),
            "bc_x0": jax.vmap(lambda s, t: bc_x_point(p, 0.0, s, t))(
                coll["bc_x0_s"], coll["bc_x0_t"]
            ),
            "bc_x1": jax.vmap(lambda s, t: bc_x_point(p, 1.0, s, t))(
                coll["bc_x1_s"], coll["bc_x1_t"]
            ),
            "bc_y0": jax.vmap(lambda s, t: bc_y_point(p, 0.0, s, t))(
                coll["bc_y0_s"], coll["bc_y0_t"]
            ),
            "bc_y1": jax.vmap(lambda s, t: bc_y_point(p, 1.0, s, t))(
                coll["bc_y1_s"], coll["bc_y1_t"]
            ),
        }

    @jax.jit
    def remaining_residuals(p):
        raw = remaining_raw_blocks(p)
        return tuple(
            raw[name] * (scale[name] / jnp.sqrt(sizes[name])) for name in order[1:]
        )

    def _interior_slices():
        return chunk_slices(sizes["pde"], chunk)

    # --- Separable interior residual --------------------------------------

    grid_axes = None
    if separable and model is not None and getattr(model, "supports_separable_residual", False):
        grid_axes = tensor_grid_axes(points, ("pde_x", "pde_y", "pde_t"))

    def _forcing_grid(axes):
        x, y, t = (jnp.reshape(axis, (-1,)) for axis in axes)
        return formula_forcing(
            x[:, None, None], y[None, :, None], t[None, None, :], kappa_x, kappa_y
        )

    def _complete_interior(p, axes):
        """u_tt - kx u_xx - ky u_yy + u^2 - f, with time a separable axis."""

        value, second = separable_grid_terms(model, p, axes, field=0)
        return (
            second[2] - kappa_x * second[0] - kappa_y * second[1]
            + value**2 - _forcing_grid(axes)
        )

    def _modulated_interior(p, axes):
        """Same residual when time acts through the modulation network."""

        x_axis, y_axis, t_axis = axes

        def at_time(t):
            x_d = axis_derivatives(model, p, 0, x_axis, time=t, order=2)
            y_d = axis_derivatives(model, p, 1, y_axis, time=t, order=2)
            x_t = time_derivatives(model, p, 0, x_axis, t, order=2)
            y_t = time_derivatives(model, p, 1, y_axis, t, order=2)
            value = combine([x_d[0], y_d[0]], field=0)
            uxx = combine([x_d[2], y_d[0]], field=0)
            uyy = combine([x_d[0], y_d[2]], field=0)
            utt = (
                combine([x_t[2], y_t[0]], field=0)
                + 2.0 * combine([x_t[1], y_t[1]], field=0)
                + combine([x_t[0], y_t[2]], field=0)
            )
            return value, uxx, uyy, utt

        value, uxx, uyy, utt = jax.vmap(at_time)(jnp.reshape(t_axis, (-1,)))
        # vmap puts time first; move it back to the trailing grid axis.
        value, uxx, uyy, utt = (
            jnp.moveaxis(term, 0, -1) for term in (value, uxx, uyy, utt)
        )
        return utt - kappa_x * uxx - kappa_y * uyy + value**2 - _forcing_grid(axes)

    separable_interior_fn = None
    if grid_axes is not None:
        _interior = (
            _complete_interior if model.separable_time_axis else _modulated_interior
        )

        @jax.jit
        def separable_interior_fn(p):  # noqa: F811
            return jnp.reshape(_interior(p, grid_axes), (-1,))

    def interior_raw(p):
        """Raw interior residual, separable when the structure allows it."""

        if separable_interior_fn is not None:
            return separable_interior_fn(p)
        return jax.vmap(lambda x, y, t: pde_point(p, x, y, t))(
            coll["pde_x"], coll["pde_y"], coll["pde_t"]
        )

    def _interior_arguments(bounds):
        return (
            coll["pde_x"][bounds],
            coll["pde_y"][bounds],
            coll["pde_t"][bounds],
        )

    def normal_equations_fn(p):
        """Return ``(J^T J, J^T r, loss)`` without materialising J or r."""

        def blocks():
            for bounds in _interior_slices():
                yield pde_chunk_system(p, *_interior_arguments(bounds))
            # Both helpers emit their blocks in the same order.
            for jacobian, residual in zip(
                remaining_jacobians(p), remaining_residuals(p)
            ):
                yield jacobian, residual

        return accumulate_normal_equations(p, blocks())

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
        """Per-block mean squared raw residual (the Adam objective, SPINN)."""

        total = jnp.sum(interior_raw(p) ** 2)
        result = {"pde": total / sizes["pde"]}
        raw = remaining_raw_blocks(p)
        for name in order[1:]:
            result[name] = jnp.mean(raw[name] ** 2)
        return result

    return {
        "normal_equations_fn": normal_equations_fn,
        "loss_fn": loss_fn,
        "block_mse_fn": block_mse_fn,
        "residual_order": order,
        "residual_scales": scale,
    }



# --- Experiment defaults ----------------------------------------------------


def _tensor_grid(n: int) -> dict[str, Any]:
    """An ``n x n`` spatial grid times ``n`` time levels."""

    return {
        "sampling_strategy": CARTESIAN_TENSOR_GRID,
        "grid_nx": n,
        "grid_ny": n,
        "grid_nt": n,
        "n_pde": n * n * n,
        "n_ic_u": n * n,
        "n_ic_ut": n * n,
        "n_bc_each": n * n,
    }


SETUP = ExperimentSetup(
    problem={
        "formula_id": FORMULA_ID,
        "kappa_x": 1.0,
        "kappa_y": 1.0,
        "t_max": 10.0,
    },
    sampling={
        "sample_seed_offset": 3000,
        "jacobian_pde_chunk_size": 2048,
    },
    profiles={f"tensor_{n}sq_x_{n}": _tensor_grid(n) for n in (50, 100)},
    evaluation={"nx": 64, "ny": 64, "nt": 24},
    model={"layer_dims": [1, 10, 10, 1], "r": 5, "phi_m": 5, "phi_hidden": [10, 10]},
)


# --- Specification ----------------------------------------------------------


class KleinGordonProblem(ProblemSpec):
    key = "kg2d"
    spatial_dim = 2
    field_names = ("u",)
    residual_names = ("pde", "ic_u", "ic_ut", "bc_x0", "bc_x1", "bc_y0", "bc_y1")
    architecture_suffix = "2d"
    method_id_suffix = ""
    spinn_key_splits = 3
    setinn_key_splits = 4
    setup = SETUP

    def validate(self, problem, sampling, evaluation, residual_scales) -> None:
        if problem.get("formula_id") != FORMULA_ID:
            raise ValueError(f"kg2d requires formula_id={FORMULA_ID!r}")
        for name in ("kappa_x", "kappa_y", "t_max"):
            value = float(problem.get(name, float("nan")))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"problem.{name} must be finite and positive")
        for name in ("n_pde", "n_ic_u", "n_ic_ut", "n_bc_each"):
            value = sampling.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"sampling.{name} must be a positive integer")
        strategy = str(sampling.get("sampling_strategy"))
        if strategy == CARTESIAN_TENSOR_GRID:
            for name in ("grid_nx", "grid_ny", "grid_nt"):
                _positive_integer(sampling.get(name), name)
            nx, ny, nt = (
                int(sampling[name]) for name in ("grid_nx", "grid_ny", "grid_nt")
            )
            if nx != ny:
                raise ValueError("KG2D tensor grids require grid_nx = grid_ny")
            if sampling["n_pde"] != nx * ny * nt:
                raise ValueError("sampling.n_pde must equal grid_nx*grid_ny*grid_nt")
            if sampling["n_ic_u"] != nx * ny or sampling["n_ic_ut"] != nx * ny:
                raise ValueError("KG initial counts must equal grid_nx*grid_ny")
            if sampling["n_bc_each"] != nx * nt:
                raise ValueError("KG boundary count per face must equal grid_nx*grid_nt")
        else:
            raise ValueError(f"unsupported sampling_strategy {strategy!r}")
        self.validate_sampling_controls(sampling)
        self.validate_residual_scales(residual_scales)
        for name in ("nx", "ny", "nt"):
            value = evaluation.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"evaluation.{name} must be an integer >= 2")

    def sample_collocation(self, sampling, problem):
        return sample_collocation(sampling, t_max=float(problem["t_max"]))

    def formula_sha256(self, problem) -> str:
        return formula_sha256(str(problem["formula_id"]))

    def make_system(
        self, model, points, problem, sampling, residual_scales, *, dtype
    ):
        self.validate_model(model)
        self.validate_dtype(model, dtype)
        return make_system(
            model.field(0),
            points,
            model=model,
            kappa_x=float(problem["kappa_x"]),
            kappa_y=float(problem["kappa_y"]),
            residual_scales=residual_scales,
            dtype=dtype,
            jacobian_pde_chunk_size=int(sampling["jacobian_pde_chunk_size"]),
            formula_id=str(problem["formula_id"]),
        )

    def evaluate(
        self, model, parameters, problem, evaluation, *, dtype, chunk_size=8192
    ):
        self.validate_model(model)
        self.validate_dtype(model, dtype)
        nx, ny, nt = (int(evaluation[name]) for name in ("nx", "ny", "nt"))
        axes = (
            np.linspace(0.0, 1.0, nx, dtype=np.float64),
            np.linspace(0.0, 1.0, ny, dtype=np.float64),
            np.linspace(0.0, float(problem["t_max"]), nt, dtype=np.float64),
        )
        prediction = predict_grid_pointwise(
            model,
            parameters,
            axes,
            dtype=dtype,
            chunk_size=chunk_size,
        )[..., 0]
        mesh = np.meshgrid(*axes, indexing="ij")
        exact = exact_u_numpy(*mesh)
        error = prediction - exact
        denominator = float(np.linalg.norm(exact))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("Klein--Gordon reference norm must be finite and positive")
        metrics = {
            "relative_l2": float(np.linalg.norm(error) / denominator),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "max_abs_error": float(np.max(np.abs(error))),
            "evaluation_points": int(exact.size),
        }
        arrays = {
            "x": axes[0],
            "y": axes[1],
            "t": axes[2],
            "prediction": prediction,
            "exact": exact,
        }
        return EvaluationResult(metrics, arrays)


KLEIN_GORDON = KleinGordonProblem()
