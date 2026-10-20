"""The protocol: a sealed, hashed record of one experiment."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from .models import architectures_for, build_model
from .models.interface import spatial_axis_names
from .problems.interface import ProblemSpec
from .training.adam import AdamHParams
from .training.lm import LMHParams

SCHEMA_VERSION = 3
OPTIMIZERS = ("lm", "adam")


class ProtocolError(ValueError):
    """A protocol is structurally invalid or names settings a problem lacks."""


def protocol_hash(protocol: Mapping[str, Any]) -> str:
    """sha256 of the protocol's JSON content (keys sorted)."""

    clean = {key: value for key, value in protocol.items() if not key.startswith("_")}
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_seed_for(protocol: Mapping[str, Any], seed: int) -> int:
    return int(protocol["sampling"]["sample_seed_offset"]) + int(seed)


def wa_token(value: float) -> str:
    """Method-id token for ``wa_init_scale``: 0.0 -> ``wa0``, 0.01 -> ``wa1em2``."""

    if value == 0.0:
        return "wa0"
    mantissa, _, exponent = f"{value:.0e}".partition("e")
    return f"wa{mantissa}e{int(exponent)}".replace("-", "m")


def count_parameters(problem: ProblemSpec, method: Mapping[str, Any]) -> int:
    """Build the method's model once in float64 and return its size."""

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    model = build_model(
        str(method["architecture"]),
        problem=problem,
        config=dict(method.get("model", {})),
        seed=0,
        dtype=jnp.float64,
    )
    return model.n_params


def _cast_like(default: Any, value: Any) -> Any:
    """Keep an override in the type of its default, so the JSON is stable."""

    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, str):
        return str(value)
    return value


def _training(optimizer: str) -> dict[str, Any]:
    if optimizer == "lm":
        return {"semantics": LMHParams.SEMANTICS, "mu0": LMHParams().mu0}
    hp = AdamHParams()
    return {
        "learning_rate": hp.learning_rate,
        "b1": hp.b1,
        "b2": hp.b2,
        "eps": hp.eps,
        "nesterov": hp.nesterov,
    }


def make_method(
    problem: ProblemSpec,
    family: str,
    *,
    optimizer: str = "lm",
    mod_axes: Sequence[str] | None = None,
    wa_init_scale: float = 0.0,
    layer_dims: Sequence[int] | None = None,
    rank: int | None = None,
) -> dict[str, Any]:
    """One method: its id, architecture, model settings and optimizer."""

    if optimizer not in OPTIMIZERS:
        raise ProtocolError(f"optimizer must be one of {OPTIMIZERS}")
    if family == "tinn":
        method_id, model = "tinn", copy.deepcopy(dict(problem.setup.tinn))
    elif family in ("spinn", "setinn", "setinnall"):
        model = copy.deepcopy(dict(problem.setup.model))
        if layer_dims is not None:
            model["layer_dims"] = [int(value) for value in layer_dims]
        if rank is not None:
            model["r"] = int(rank)
        if family == "spinn":
            model = {k: v for k, v in model.items() if k not in ("phi_m", "phi_hidden")}
            method_id = f"spinn_{problem.architecture_suffix}"
        else:
            axes = mod_axes or spatial_axis_names(problem.spatial_dim)
            model["mod_axes"] = [str(axis) for axis in axes]
            model["wa_init_scale"] = float(wa_init_scale)
            method_id = (
                f"{family}_{''.join(model['mod_axes'])}_"
                f"{wa_token(model['wa_init_scale'])}{problem.method_id_suffix}"
            )
    else:
        raise ProtocolError(f"unknown model family {family!r}")
    return {
        "id": method_id,
        "architecture": f"{family}_{problem.architecture_suffix}",
        "optimizer": optimizer,
        "precision": "float64",
        "model": model,
        "training": _training(optimizer),
    }


def build_protocol(
    problem: ProblemSpec,
    method: Mapping[str, Any],
    *,
    seeds: Sequence[int],
    profile: str,
    settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the settings and return the protocol for ``method``."""

    setup = problem.setup
    seeds = [int(seed) for seed in seeds]
    problem_block = copy.deepcopy(dict(setup.problem))
    for key, value in (settings or {}).items():
        if key not in problem_block:
            raise ProtocolError(f"unknown setting {key!r} for problem {problem.key!r}")
        problem_block[key] = _cast_like(problem_block[key], value)

    sampling = copy.deepcopy(dict(setup.sampling))
    if profile not in setup.profiles:
        raise ProtocolError(
            f"unknown profile {profile!r} for {problem.key}; "
            f"choose one of {sorted(setup.profiles)}"
        )
    sampling.update(copy.deepcopy(dict(setup.profiles[profile])))
    sampling["profile"] = str(profile)

    method = copy.deepcopy(dict(method))
    try:
        method["expected_parameter_count"] = count_parameters(problem, method)
    except ValueError as error:
        raise ProtocolError(f"invalid model for {method.get('id')}: {error}") from error

    protocol: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "",
        "problem": {"kind": problem.key, **problem_block},
        "precision": "float64",
        "seeds": seeds,
        "sampling": sampling,
        "residual_scales": {name: 1.0 for name in problem.residual_names},
        "evaluation": copy.deepcopy(dict(setup.evaluation)),
        "methods": [method],
    }
    settings_only = {
        key: protocol[key]
        for key in ("seeds", "problem", "sampling", "residual_scales", "evaluation", "methods")
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            settings_only, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()[:10]
    protocol["protocol_id"] = f"{problem.key}_s{len(seeds)}_cfg{fingerprint}"
    validate_protocol(protocol, problem)
    return protocol


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _safe_name(value: Any, field: str) -> None:
    if not isinstance(value, str) or value in {".", ".."} or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", value
    ):
        raise ProtocolError(f"{field} must be a single safe path component")


def validate_protocol(protocol: Mapping[str, Any], problem: ProblemSpec) -> None:
    """Reject a protocol that could not be run as written."""

    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError(f"schema_version must be {SCHEMA_VERSION}")
    _safe_name(protocol.get("protocol_id"), "protocol_id")
    if protocol["problem"].get("kind") != problem.key:
        raise ProtocolError(f"problem.kind must be {problem.key!r}")
    seeds = protocol.get("seeds")
    if not seeds or not all(_is_int(seed) for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ProtocolError("seeds must be a non-empty list of distinct integers")
    if protocol.get("precision") not in {"float32", "float64"}:
        raise ProtocolError("precision must be float32 or float64")
    sampling = protocol["sampling"]
    if "seed" in sampling:
        raise ProtocolError(
            "sampling.seed is not allowed: the sample seed is "
            "sample_seed_offset + seed, so every method of a seed shares its points"
        )
    if not _is_int(sampling.get("sample_seed_offset")):
        raise ProtocolError("sampling.sample_seed_offset must be an integer")
    for name, scale in protocol["residual_scales"].items():
        if not (math.isfinite(float(scale)) and float(scale) >= 0):
            raise ProtocolError(f"residual_scales.{name} must be finite and non-negative")

    methods = protocol.get("methods") or []
    if not methods:
        raise ProtocolError("a protocol needs at least one method")
    ids = [method.get("id") for method in methods]
    if len(set(ids)) != len(ids):
        raise ProtocolError("method ids must be unique")
    valid = architectures_for(problem)
    for method in methods:
        _safe_name(method.get("id"), "method id")
        if "seed" in method:
            raise ProtocolError(f"{method['id']}: a method may not carry its own seed")
        if method.get("architecture") not in valid:
            raise ProtocolError(f"{method['id']}: architecture must be one of {valid}")
        if method.get("optimizer") not in OPTIMIZERS:
            raise ProtocolError(f"{method['id']}: optimizer must be one of {OPTIMIZERS}")
        if method.get("precision") not in {"float32", "float64"}:
            raise ProtocolError(f"{method['id']}: invalid precision")
        training = method.get("training", {})
        if method["optimizer"] == "lm":
            if method["precision"] != "float64":
                raise ProtocolError(f"{method['id']}: LM requires float64")
            if not training.get("semantics"):
                raise ProtocolError(f"{method['id']}: LM needs training.semantics")
        elif not float(training.get("learning_rate", 0.0)) > 0.0:
            raise ProtocolError(f"{method['id']}: Adam needs a positive learning_rate")
        count = method.get("expected_parameter_count")
        if not _is_int(count) or count <= 0:
            raise ProtocolError(f"{method['id']}: expected_parameter_count must be positive")
    try:
        problem.validate(
            protocol["problem"],
            sampling,
            protocol["evaluation"],
            protocol["residual_scales"],
        )
    except ValueError as error:
        raise ProtocolError(str(error)) from error
