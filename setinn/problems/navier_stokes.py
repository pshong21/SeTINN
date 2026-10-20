"""(2+1)-D incompressible Navier--Stokes, Taylor--Green vortex (``nstg``)."""

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
    tensor_grid_axes,
    time_derivatives,
)


# --- Physics ---------------------------------------------------------------


FORMULA_ID = "nstg_taylor_green_v1"
FORMULA_TEXT = (
    "F(t)=exp(-2*nu*t); u=-cos(x)*sin(y)*F; v=sin(x)*cos(y)*F; "
    "p=-0.25*(cos(2x)+cos(2y))*F^2; domain x,y in [0,2pi], t in [0,t_max]; "
    "mom_x=u_t+u*u_x+v*u_y+p_x-nu*(u_xx+u_yy); "
    "mom_y=v_t+u*v_x+v*v_y+p_y-nu*(v_xx+v_yy); cont=u_x+v_y; "
    "IC on (u,v) at t=0; Dirichlet (u,v) on four sides; p gauge-free"
)

BC_MODE = "four_sides_separate_dirichlet_uv"
BLOCKS = ("mom_x", "mom_y", "cont", "ic_u", "ic_v", "bc_x0", "bc_x1", "bc_y0", "bc_y1")

TWO_PI = 2.0 * math.pi


def exact_u(x, y, t, nu):
    return -jnp.cos(x) * jnp.sin(y) * jnp.exp(-2.0 * nu * t)


def exact_v(x, y, t, nu):
    return jnp.sin(x) * jnp.cos(y) * jnp.exp(-2.0 * nu * t)


def exact_p(x, y, t, nu):
    return -0.25 * (jnp.cos(2.0 * x) + jnp.cos(2.0 * y)) * jnp.exp(-4.0 * nu * t)


def exact_uvp(x, y, t, nu):
    return jnp.array([exact_u(x, y, t, nu), exact_v(x, y, t, nu), exact_p(x, y, t, nu)])


def exact_uvp_numpy(x, y, t, nu):
    """NumPy float64 analytic solution, shape (..., 3), for evaluation."""

    decay = np.exp(-2.0 * nu * t)
    u = -np.cos(x) * np.sin(y) * decay
    v = np.sin(x) * np.cos(y) * decay
    p = -0.25 * (np.cos(2.0 * x) + np.cos(2.0 * y)) * decay**2
    return np.stack([u, v, p], axis=-1)


def formula_sha256(formula_id: str = FORMULA_ID) -> str:
    if formula_id != FORMULA_ID:
        raise ValueError(f"unknown NSTG formula_id: {formula_id!r}")
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
    n_ic = _positive_integer(sampling["n_ic"], "n_ic")
    n_bc_each = _positive_integer(sampling["n_bc_each"], "n_bc_each")
    strategy = str(sampling.get("sampling_strategy"))
    if strategy == CARTESIAN_TENSOR_GRID:
        nx = _positive_integer(sampling.get("grid_nx"), "grid_nx")
        ny = _positive_integer(sampling.get("grid_ny"), "grid_ny")
        nt = _positive_integer(sampling.get("grid_nt"), "grid_nt")
        if n_pde != nx * ny * nt:
            raise ValueError("n_pde must equal grid_nx * grid_ny * grid_nt")
        if n_ic != nx * ny:
            raise ValueError("n_ic must equal grid_nx * grid_ny")
        if n_bc_each != ny * nt or n_bc_each != nx * nt:
            raise ValueError(
                "n_bc_each must equal both grid_ny*grid_nt and grid_nx*grid_nt"
            )
        x = np.linspace(0.0, TWO_PI, nx, dtype=np.float64)
        y = np.linspace(0.0, TWO_PI, ny, dtype=np.float64)
        t = np.linspace(0.0, t_max, nt, dtype=np.float64)
        pde_x, pde_y, pde_t = np.meshgrid(x, y, t, indexing="ij")
        ic_x, ic_y = np.meshgrid(x, y, indexing="ij")
        y_face, y_time = np.meshgrid(y, t, indexing="ij")
        x_face, x_time = np.meshgrid(x, t, indexing="ij")
        return {
            "pde_x": pde_x.reshape(-1),
            "pde_y": pde_y.reshape(-1),
            "pde_t": pde_t.reshape(-1),
            "ic_x": ic_x.reshape(-1),
            "ic_y": ic_y.reshape(-1),
            "bc_x0_s": y_face.reshape(-1),
            "bc_x0_t": y_time.reshape(-1),
            "bc_x1_s": y_face.reshape(-1),
            "bc_x1_t": y_time.reshape(-1),
            "bc_y0_s": x_face.reshape(-1),
            "bc_y0_t": x_time.reshape(-1),
            "bc_y1_s": x_face.reshape(-1),
            "bc_y1_t": x_time.reshape(-1),
        }
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


def make_system(
    uvp_flat,
    points: Mapping[str, Any],
    *,
    nu: float,
    bc_mode: str,
    residual_scales: Mapping[str, Any],
    dtype: Any,
    jacobian_pde_chunk_size: int,
    formula_id: str = FORMULA_ID,
    model: Any = None,
    separable: bool = True,
):
    """The residual system of one model; see ``ProblemSpec.make_system``."""

    if formula_id != FORMULA_ID:
        raise ValueError(f"unknown NSTG formula_id: {formula_id!r}")
    if bc_mode != BC_MODE:
        raise ValueError(
            f"bc_mode must be passed explicitly as {BC_MODE!r}, got {bc_mode!r}"
        )
    nu = float(nu)
    if not math.isfinite(nu) or nu <= 0:
        raise ValueError("nu must be finite and positive")
    scale = _scale_map(residual_scales)
    coll = {name: jnp.asarray(value, dtype=dtype) for name, value in points.items()}
    n_pde = int(coll["pde_x"].size)
    n_ic = int(coll["ic_x"].size)
    n_bc = {side: int(coll[f"bc_{side}_s"].size) for side in ("x0", "x1", "y0", "y1")}
    sizes = {
        "mom_x": n_pde,
        "mom_y": n_pde,
        "cont": n_pde,
        "ic_u": n_ic,
        "ic_v": n_ic,
        # Two constrained components (u, v) per boundary point.
        "bc_x0": 2 * n_bc["x0"],
        "bc_x1": 2 * n_bc["x1"],
        "bc_y0": 2 * n_bc["y0"],
        "bc_y1": 2 * n_bc["y1"],
    }
    chunk = min(_positive_integer(jacobian_pde_chunk_size, "jacobian_pde_chunk_size"), n_pde)

    zero = jnp.asarray(0.0, dtype=dtype)
    two_pi = jnp.asarray(TWO_PI, dtype=dtype)

    def u_(p, x, y, t):
        return uvp_flat(p, x, y, t)[0]

    def v_(p, x, y, t):
        return uvp_flat(p, x, y, t)[1]

    def pr_(p, x, y, t):
        return uvp_flat(p, x, y, t)[2]

    def pde_point(p, x, y, t):
        u = u_(p, x, y, t)
        v = v_(p, x, y, t)
        ux = jax.grad(u_, 1)(p, x, y, t)
        uy = jax.grad(u_, 2)(p, x, y, t)
        ut = jax.grad(u_, 3)(p, x, y, t)
        vx = jax.grad(v_, 1)(p, x, y, t)
        vy = jax.grad(v_, 2)(p, x, y, t)
        vt = jax.grad(v_, 3)(p, x, y, t)
        px = jax.grad(pr_, 1)(p, x, y, t)
        py = jax.grad(pr_, 2)(p, x, y, t)
        uxx = jax.grad(jax.grad(u_, 1), 1)(p, x, y, t)
        uyy = jax.grad(jax.grad(u_, 2), 2)(p, x, y, t)
        vxx = jax.grad(jax.grad(v_, 1), 1)(p, x, y, t)
        vyy = jax.grad(jax.grad(v_, 2), 2)(p, x, y, t)
        mom_x = ut + u * ux + v * uy + px - nu * (uxx + uyy)
        mom_y = vt + u * vx + v * vy + py - nu * (vxx + vyy)
        cont = ux + vy
        return jnp.array([mom_x, mom_y, cont])

    def ic_point(p, x, y):
        # (u, v) only: the pressure constant is a gauge freedom.
        prediction = uvp_flat(p, x, y, zero)[:2]
        return prediction - exact_uvp(x, y, zero, nu)[:2]

    def bc_point(p, x, y, t):
        prediction = uvp_flat(p, x, y, t)[:2]
        return prediction - exact_uvp(x, y, t, nu)[:2]

    bc_coordinates = {
        "x0": lambda s: (zero, s),
        "x1": lambda s: (two_pi, s),
        "y0": lambda s: (s, zero),
        "y1": lambda s: (s, two_pi),
    }
    sides = ("x0", "x1", "y0", "y1")

    pde_weight = jnp.asarray(
        [
            scale["mom_x"] / math.sqrt(sizes["mom_x"]),
            scale["mom_y"] / math.sqrt(sizes["mom_y"]),
            scale["cont"] / math.sqrt(sizes["cont"]),
        ],
        dtype=dtype,
    )
    ic_weight = jnp.asarray(
        [
            scale["ic_u"] / math.sqrt(sizes["ic_u"]),
            scale["ic_v"] / math.sqrt(sizes["ic_v"]),
        ],
        dtype=dtype,
    )
    bc_weight = {
        side: scale[f"bc_{side}"] / math.sqrt(sizes[f"bc_{side}"]) for side in sides
    }

    def bc_block(p, side):
        place = bc_coordinates[side]

        def one(s, t):
            x, y = place(s)
            return bc_point(p, x, y, t)

        return jax.vmap(one)(coll[f"bc_{side}_s"], coll[f"bc_{side}_t"])

    # --- Separable interior -----------------------------------------------

    grid_axes = None
    if (
        separable
        and model is not None
        and getattr(model, "supports_separable_residual", False)
    ):
        grid_axes = tensor_grid_axes(points, ("pde_x", "pde_y", "pde_t"))

    U, V, P = 0, 1, 2  # output indices of the three fields

    def _complete_terms(p, axes):
        """Every quantity the residual needs, with time a separable axis."""

        derivatives = [
            axis_derivatives(model, p, index, coordinates, order=2)
            for index, coordinates in enumerate(axes)
        ]

        def term(orders, field):
            return combine(
                [derivatives[a][orders[a]] for a in range(3)], field=field
            )

        return {
            "u": term((0, 0, 0), U), "v": term((0, 0, 0), V),
            "ux": term((1, 0, 0), U), "uy": term((0, 1, 0), U),
            "ut": term((0, 0, 1), U),
            "vx": term((1, 0, 0), V), "vy": term((0, 1, 0), V),
            "vt": term((0, 0, 1), V),
            "px": term((1, 0, 0), P), "py": term((0, 1, 0), P),
            "uxx": term((2, 0, 0), U), "uyy": term((0, 2, 0), U),
            "vxx": term((2, 0, 0), V), "vyy": term((0, 2, 0), V),
        }

    def _modulated_terms_at_time(p, x_axis, y_axis, t_value):
        """The same quantities on one time slice of a modulated model."""

        x_d = axis_derivatives(model, p, 0, x_axis, time=t_value, order=2)
        y_d = axis_derivatives(model, p, 1, y_axis, time=t_value, order=2)
        x_t = time_derivatives(model, p, 0, x_axis, t_value, order=1)
        y_t = time_derivatives(model, p, 1, y_axis, t_value, order=1)

        def space(ox, oy, field):
            return combine([x_d[ox], y_d[oy]], field=field)

        def d_time(field):
            return (
                combine([x_t[1], y_t[0]], field=field)
                + combine([x_t[0], y_t[1]], field=field)
            )

        return {
            "u": space(0, 0, U), "v": space(0, 0, V),
            "ux": space(1, 0, U), "uy": space(0, 1, U), "ut": d_time(U),
            "vx": space(1, 0, V), "vy": space(0, 1, V), "vt": d_time(V),
            "px": space(1, 0, P), "py": space(0, 1, P),
            "uxx": space(2, 0, U), "uyy": space(0, 2, U),
            "vxx": space(2, 0, V), "vyy": space(0, 2, V),
        }

    _TERM_NAMES = (
        "u", "v", "ux", "uy", "ut", "vx", "vy", "vt",
        "px", "py", "uxx", "uyy", "vxx", "vyy",
    )

    def _residual_from_terms(q):
        """The three interior residuals from the assembled derivative terms."""

        mom_x = (
            q["ut"] + q["u"] * q["ux"] + q["v"] * q["uy"] + q["px"]
            - nu * (q["uxx"] + q["uyy"])
        )
        mom_y = (
            q["vt"] + q["u"] * q["vx"] + q["v"] * q["vy"] + q["py"]
            - nu * (q["vxx"] + q["vyy"])
        )
        return mom_x, mom_y, q["ux"] + q["vy"]

    def _separable_interior(p):
        """Interior residuals as an (n_pde, 3) array, matching pde_point."""

        if model.separable_time_axis:
            terms = _complete_terms(p, grid_axes)
        else:
            x_axis, y_axis, t_axis = grid_axes

            def at_time(t_value):
                sliced = _modulated_terms_at_time(p, x_axis, y_axis, t_value)
                return tuple(sliced[name] for name in _TERM_NAMES)

            stacked = jax.vmap(at_time)(jnp.reshape(t_axis, (-1,)))
            # vmap puts time first; the grid convention is (x, y, t).
            terms = {
                name: jnp.moveaxis(value, 0, -1)
                for name, value in zip(_TERM_NAMES, stacked)
            }
        mom_x, mom_y, cont = _residual_from_terms(terms)
        return jnp.stack(
            [jnp.reshape(term, (-1,)) for term in (mom_x, mom_y, cont)], axis=1
        )

    separable_interior_fn = (
        jax.jit(_separable_interior) if grid_axes is not None else None
    )

    def interior_values(p):
        """Raw (n_pde, 3) interior residuals, separable when possible."""

        if separable_interior_fn is not None:
            return separable_interior_fn(p)
        return jax.vmap(lambda x, y, t: pde_point(p, x, y, t))(
            coll["pde_x"], coll["pde_y"], coll["pde_t"]
        )

    grad_pde = jax.jacrev(pde_point)
    grad_ic = jax.jacrev(ic_point)
    grad_bc = jax.jacrev(bc_point)

    @jax.jit
    def remaining_jacobians(p):
        ic = jax.vmap(grad_ic, in_axes=(None, 0, 0))(p, coll["ic_x"], coll["ic_y"])
        pieces = [(ic * ic_weight[None, :, None]).reshape(-1, p.shape[0])]
        for side in sides:
            place = bc_coordinates[side]

            def one(s, t):
                x, y = place(s)
                return grad_bc(p, x, y, t)

            block = jax.vmap(one)(coll[f"bc_{side}_s"], coll[f"bc_{side}_t"])
            pieces.append((block * bc_weight[side]).reshape(-1, p.shape[0]))
        return tuple(pieces)

    # --- Memory-bounded path ---------------------------------------------

    @jax.jit
    def pde_chunk_system(p, xs, ys, ts):
        """One interior chunk as an aligned (J_chunk, r_chunk) pair."""

        values = jax.vmap(lambda x, y, t: pde_point(p, x, y, t))(xs, ys, ts)
        gradients = jax.vmap(grad_pde, in_axes=(None, 0, 0, 0))(p, xs, ys, ts)
        residual = (values * pde_weight[None, :]).reshape(-1)
        jacobian = (gradients * pde_weight[None, :, None]).reshape(-1, p.shape[0])
        return jacobian, residual

    @jax.jit
    def pde_residual_chunk(p, xs, ys, ts):
        values = jax.vmap(lambda x, y, t: pde_point(p, x, y, t))(xs, ys, ts)
        return (values * pde_weight[None, :]).reshape(-1)

    @jax.jit
    def remaining_residuals(p):
        ic = jax.vmap(lambda x, y: ic_point(p, x, y))(coll["ic_x"], coll["ic_y"])
        pieces = [(ic * ic_weight[None, :]).reshape(-1)]
        for side in sides:
            pieces.append((bc_block(p, side) * bc_weight[side]).reshape(-1))
        return tuple(pieces)

    def _interior_slices():
        return chunk_slices(n_pde, chunk)

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
        """Per-block mean squared raw residual (the Adam objective, SPINN)."""

        totals = jnp.sum(interior_values(p) ** 2, axis=0)
        result = {
            "mom_x": totals[0] / sizes["mom_x"],
            "mom_y": totals[1] / sizes["mom_y"],
            "cont": totals[2] / sizes["cont"],
        }
        ic = jax.vmap(lambda x, y: ic_point(p, x, y))(coll["ic_x"], coll["ic_y"])
        result["ic_u"] = jnp.mean(ic[:, 0] ** 2)
        result["ic_v"] = jnp.mean(ic[:, 1] ** 2)
        for side in sides:
            result[f"bc_{side}"] = jnp.mean(bc_block(p, side) ** 2)
        return result

    return {
        "normal_equations_fn": normal_equations_fn,
        "loss_fn": loss_fn,
        "block_mse_fn": block_mse_fn,
        "residual_order": BLOCKS,
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
        "n_ic": n * n,
        "n_bc_each": n * n,
    }


SETUP = ExperimentSetup(
    problem={
        "formula_id": FORMULA_ID,
        "nu": 0.025,
        "t_max": 1.0,
        "bc_mode": BC_MODE,
    },
    sampling={
        "sample_seed_offset": 5000,
        "jacobian_pde_chunk_size": 2048,
    },
    profiles={f"tensor_{n}sq_x_{n}": _tensor_grid(n) for n in (50, 100)},
    evaluation={"nx": 24, "ny": 24, "nt": 12},
    model={"layer_dims": [1, 10, 10, 3], "r": 5, "phi_m": 5, "phi_hidden": [10, 10]},
    tinn={"temporal_layers": [1, 10, 10, 5], "spatial_layers": [2, 20, 20, 3]},
)


# --- Specification ----------------------------------------------------------


class NavierStokesProblem(ProblemSpec):
    key = "nstg"
    spatial_dim = 2
    field_names = ("u", "v", "p")
    residual_names = tuple(BLOCKS)
    architecture_suffix = "nstg"
    method_id_suffix = "_nstg"
    # Four keys for three axes; the count fixes the initial parameters.
    spinn_key_splits = 4
    setinn_key_splits = 4
    setup = SETUP

    def validate(self, problem, sampling, evaluation, residual_scales) -> None:
        if problem.get("formula_id") != FORMULA_ID:
            raise ValueError(f"nstg requires formula_id={FORMULA_ID!r}")
        if problem.get("bc_mode") != BC_MODE:
            raise ValueError(f"nstg requires bc_mode={BC_MODE!r}")
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
            nx = sampling.get("grid_nx")
            ny = sampling.get("grid_ny")
            nt = sampling.get("grid_nt")
            if not all(isinstance(value, int) and value > 0 for value in (nx, ny, nt)):
                raise ValueError(
                    "cartesian_tensor_grid requires positive integer grid_nx, "
                    "grid_ny and grid_nt"
                )
            if sampling["n_pde"] != nx * ny * nt:
                raise ValueError(
                    "sampling.n_pde must equal grid_nx * grid_ny * grid_nt"
                )
            if sampling["n_ic"] != nx * ny:
                raise ValueError("sampling.n_ic must equal grid_nx * grid_ny")
            if sampling["n_bc_each"] != ny * nt or sampling["n_bc_each"] != nx * nt:
                raise ValueError(
                    "sampling.n_bc_each must match each space-time boundary face"
                )
        else:
            raise ValueError(f"unknown sampling.sampling_strategy {strategy!r}")
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
            model.fields(),
            points,
            nu=float(problem["nu"]),
            bc_mode=str(problem["bc_mode"]),
            residual_scales=residual_scales,
            dtype=dtype,
            jacobian_pde_chunk_size=int(sampling["jacobian_pde_chunk_size"]),
            formula_id=str(problem["formula_id"]),
            model=model,
        )

    def evaluate(
        self, model, parameters, problem, evaluation, *, dtype, chunk_size=8192
    ):
        self.validate_model(model)
        self.validate_dtype(model, dtype)
        nu = float(problem["nu"])
        nx, ny, nt = (int(evaluation[name]) for name in ("nx", "ny", "nt"))
        axes = (
            np.linspace(0.0, TWO_PI, nx, dtype=np.float64),
            np.linspace(0.0, TWO_PI, ny, dtype=np.float64),
            np.linspace(0.0, float(problem["t_max"]), nt, dtype=np.float64),
        )
        prediction = predict_grid_pointwise(
            model,
            parameters,
            axes,
            dtype=dtype,
            chunk_size=chunk_size,
        )
        mesh = np.meshgrid(*axes, indexing="ij")
        exact = exact_uvp_numpy(*mesh, nu)
        error = prediction - exact

        def rel(err, ref):
            denominator = float(np.linalg.norm(ref))
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise ValueError("reference norm must be finite and positive")
            return float(np.linalg.norm(err) / denominator)

        flat_error = error.reshape(-1, 3)
        flat_exact = exact.reshape(-1, 3)
        pressure_gauge_shift = float(np.mean(flat_error[:, 2]))
        pressure_error = flat_error[:, 2] - pressure_gauge_shift
        pressure_exact = flat_exact[:, 2] - np.mean(flat_exact[:, 2])
        metrics = {
            "relative_l2": rel(flat_error[:, :2], flat_exact[:, :2]),
            "relative_l2_u": rel(flat_error[:, 0], flat_exact[:, 0]),
            "relative_l2_v": rel(flat_error[:, 1], flat_exact[:, 1]),
            "relative_l2_p_zero_mean": rel(pressure_error, pressure_exact),
            "pressure_gauge_shift": pressure_gauge_shift,
            "rmse_uv": float(np.sqrt(np.mean(flat_error[:, :2] ** 2))),
            "max_abs_error_uv": float(np.max(np.abs(flat_error[:, :2]))),
            "evaluation_points": int(flat_exact.shape[0]),
        }
        arrays = {
            "x": axes[0],
            "y": axes[1],
            "t": axes[2],
            "prediction": prediction,
            "exact": exact,
        }
        return EvaluationResult(metrics, arrays)


NAVIER_STOKES = NavierStokesProblem()

