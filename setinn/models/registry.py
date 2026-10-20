"""Build a model from its architecture name and settings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .interface import ModelInterface, spatial_axis_names

if TYPE_CHECKING:
    from ..problems.interface import ProblemSpec

FAMILIES = ("tinn", "spinn", "setinn", "setinnall")
INDUCED_LAYERS = {"setinn": "first", "setinnall": "all"}
SETTINGS = {
    "tinn": {"temporal_layers", "spatial_layers"},
    "spinn": {"layer_dims", "r"},
    "setinn": {"layer_dims", "r", "phi_m", "phi_hidden", "mod_axes", "wa_init_scale"},
    "setinnall": {"layer_dims", "r", "phi_m", "phi_hidden", "mod_axes", "wa_init_scale"},
}


def architectures_for(problem: ProblemSpec) -> tuple[str, ...]:
    """Every architecture name valid for one problem."""

    return tuple(f"{family}_{problem.architecture_suffix}" for family in FAMILIES)


def _hidden_widths(
    values: Sequence[Any], *, setting: str, input_width: int, output_width: int
) -> tuple[int, ...]:
    layers = tuple(int(value) for value in values)
    if (
        len(layers) < 3
        or layers[0] != input_width
        or layers[-1] != output_width
        or any(value <= 0 for value in layers)
    ):
        raise ValueError(
            f"{setting} must contain positive widths beginning at "
            f"{input_width} and ending at {output_width}"
        )
    return layers[1:-1]


def build_model(
    architecture: str,
    *,
    problem: ProblemSpec,
    config: Mapping[str, Any],
    seed: int,
    dtype: Any,
) -> ModelInterface:
    """Build one model, with its input and output widths taken from ``problem``."""

    family, _, suffix = str(architecture).partition("_")
    if family not in FAMILIES or suffix != problem.architecture_suffix:
        raise ValueError(
            f"architecture {architecture!r} is not valid for problem "
            f"{problem.key!r}; choose one of {architectures_for(problem)}"
        )
    config = dict(config)
    unknown = set(config) - SETTINGS[family]
    if unknown:
        raise ValueError(f"unknown {family} model settings: {sorted(unknown)}")
    spatial_dim, n_out = problem.spatial_dim, problem.n_out

    if family == "tinn":
        from .tinn import TINN

        hidden = (
            _hidden_widths(
                config["spatial_layers"],
                setting="spatial_layers",
                input_width=spatial_dim,
                output_width=n_out,
            )
            if "spatial_layers" in config
            else (20, 20)
        )
        temporal = tuple(
            int(value)
            for value in config.get("temporal_layers", (1, 10, 10, 2 * len(hidden) + 1))
        )
        builder = TINN(
            temporal_layers=temporal, spatial_layers=(spatial_dim, *hidden, n_out)
        )
    else:
        hidden = (
            _hidden_widths(
                config["layer_dims"],
                setting="layer_dims",
                input_width=1,
                output_width=n_out,
            )
            if "layer_dims" in config
            else (10, 10)
        )
        rank = int(config.get("r", 10))
        if rank < 1:
            raise ValueError("r must be positive")
        axis_layers = (1, *hidden, n_out)
        if family == "spinn":
            from .spinn import SPINN

            builder = SPINN(
                spatial_dim=spatial_dim,
                n_out=n_out,
                rank=rank,
                axis_layers=axis_layers,
                key_splits=problem.spinn_key_splits,
            )
        else:
            from .phi import PhiNetwork
            from .setinn import SeTINN

            names = spatial_axis_names(spatial_dim)
            mod_axes = tuple(str(axis) for axis in config.get("mod_axes", names))
            if (
                not mod_axes
                or len(set(mod_axes)) != len(mod_axes)
                or any(axis not in names for axis in mod_axes)
            ):
                raise ValueError(f"mod_axes must be distinct axes from {names}")
            phi_m = int(config.get("phi_m", 8))
            if phi_m < 1:
                raise ValueError("phi_m must be positive")
            builder = SeTINN(
                spatial_dim=spatial_dim,
                n_out=n_out,
                rank=rank,
                axis_layers=axis_layers,
                induced_layers=INDUCED_LAYERS[family],
                modulated_axes=tuple(names.index(axis) for axis in mod_axes),
                phi=PhiNetwork(
                    output_dim=phi_m,
                    hidden_layers=tuple(
                        int(value) for value in config.get("phi_hidden", (16, 16))
                    ),
                ),
                wa_init_scale=float(config.get("wa_init_scale", 0.0)),
                key_splits=problem.setinn_key_splits,
            )
    model = builder.build(seed=seed, dtype=dtype)
    problem.validate_model(model)
    return model
