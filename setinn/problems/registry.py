"""The benchmark PDEs, by key."""

from __future__ import annotations

from .interface import ProblemSpec
from .klein_gordon import KLEIN_GORDON
from .klein_gordon_3d import KLEIN_GORDON_3D
from .navier_stokes import NAVIER_STOKES
from .navier_stokes_3d import NAVIER_STOKES_3D

PROBLEMS: dict[str, ProblemSpec] = {
    problem.key: problem
    for problem in (KLEIN_GORDON, NAVIER_STOKES, KLEIN_GORDON_3D, NAVIER_STOKES_3D)
}
