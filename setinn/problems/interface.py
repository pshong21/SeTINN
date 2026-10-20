"""What a PDE module provides: sampling, the residual system and evaluation."""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from ..models.interface import ModelInterface


@dataclass(frozen=True)
class ExperimentSetup:
    """The defaults :func:`setinn.protocol.build_protocol` starts from."""

    problem: Mapping[str, Any]
    sampling: Mapping[str, Any]
    profiles: Mapping[str, Mapping[str, Any]]
    evaluation: Mapping[str, Any]
    model: Mapping[str, Any]
    tinn: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    metrics: Mapping[str, Any]
    arrays: Mapping[str, Any]


class ProblemSpec(ABC):
    """One PDE: physics, collocation, residual system and evaluation."""

    key: str
    spatial_dim: int
    field_names: tuple[str, ...]
    residual_names: tuple[str, ...]
    architecture_suffix: str
    method_id_suffix: str
    spinn_key_splits: int
    setinn_key_splits: int
    setup: ExperimentSetup

    @property
    def n_out(self) -> int:
        return len(self.field_names)

    def validate_model(self, model: ModelInterface) -> None:
        if model.spatial_dim != self.spatial_dim:
            raise ValueError(
                f"{self.key} requires spatial_dim={self.spatial_dim}, "
                f"got {model.spatial_dim}"
            )
        if model.n_out != self.n_out:
            raise ValueError(
                f"{self.key} requires fields {self.field_names} "
                f"(n_out={self.n_out}), got n_out={model.n_out}"
            )

    @staticmethod
    def validate_dtype(model: ModelInterface, dtype: Any) -> None:
        requested = np.dtype(dtype)
        actual = np.dtype(model.dtype)
        if requested != actual:
            raise ValueError(
                f"evaluation/system dtype {requested} does not match "
                f"model dtype {actual}"
            )

    def collocation_sha256(self, points: Mapping[str, Any]) -> str:
        """Hash one point set: names, shapes and float64 bytes, sorted by name."""

        digest = hashlib.sha256()
        for name in sorted(points):
            value = np.ascontiguousarray(np.asarray(points[name], dtype=np.float64))
            digest.update(name.encode("ascii") + b"\0")
            digest.update(str(value.shape).encode("ascii") + b"\0")
            digest.update(value.tobytes(order="C"))
        return digest.hexdigest()

    def validate_residual_scales(self, scales: Mapping[str, Any]) -> None:
        if set(scales) != set(self.residual_names):
            raise ValueError(
                f"residual_scales must contain exactly {self.residual_names}"
            )
        for name in self.residual_names:
            value = float(scales[name])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"residual scale {name!r} must be finite and non-negative"
                )

    @staticmethod
    def validate_sampling_controls(sampling: Mapping[str, Any]) -> None:
        seed = sampling.get("seed", sampling.get("sample_seed_offset"))
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("sampling must declare integer seed or sample_seed_offset")
        chunk = sampling.get("jacobian_pde_chunk_size")
        if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk <= 0:
            raise ValueError(
                "sampling.jacobian_pde_chunk_size must be a positive integer"
            )

    @abstractmethod
    def validate(
        self,
        problem: Mapping[str, Any],
        sampling: Mapping[str, Any],
        evaluation: Mapping[str, Any],
        residual_scales: Mapping[str, Any],
    ) -> None:
        """Reject settings this PDE cannot run, before anything is built."""

    @abstractmethod
    def sample_collocation(
        self, sampling: Mapping[str, Any], problem: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """The float64 collocation points; ``sampling["seed"]`` is the sample seed."""

    @abstractmethod
    def formula_sha256(self, problem: Mapping[str, Any]) -> str:
        """Hash of the closed-form solution and forcing this PDE uses."""

    @abstractmethod
    def make_system(
        self,
        model: ModelInterface,
        points: Mapping[str, Any],
        problem: Mapping[str, Any],
        sampling: Mapping[str, Any],
        residual_scales: Mapping[str, Any],
        *,
        dtype: Any,
    ) -> Mapping[str, Any]:
        """The residual system the optimizers use."""

    @abstractmethod
    def evaluate(
        self,
        model: ModelInterface,
        parameters: Any,
        problem: Mapping[str, Any],
        evaluation: Mapping[str, Any],
        *,
        dtype: Any,
        chunk_size: int = 8192,
    ) -> EvaluationResult:
        """Metrics and fields on the evaluation grid, through the point function."""
