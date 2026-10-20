"""TINN, SPINN and SeTINN behind one point-model interface (``interface.py``)."""

from .registry import FAMILIES, architectures_for, build_model

__all__ = ["FAMILIES", "architectures_for", "build_model"]
