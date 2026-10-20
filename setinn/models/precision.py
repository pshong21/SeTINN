"""Explicit floating-point precision for every model."""

from __future__ import annotations

import os
from typing import Any

# Set before JAX is imported.
os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def require_floating_dtype(dtype: Any) -> Any:
    """Return a concrete JAX floating dtype and reject silent downcasts."""

    if dtype is None:
        raise ValueError("dtype must be selected explicitly")
    result = jnp.dtype(dtype)
    if not jnp.issubdtype(result, jnp.floating):
        raise ValueError(f"dtype must be floating, got {result}")
    if result == jnp.dtype(jnp.float64) and not jax.config.x64_enabled:
        raise RuntimeError(
            "float64 was requested after JAX started with x64 disabled; "
            "set JAX_ENABLE_X64=True before importing JAX"
        )
    return result


def cast_floating(tree: Any, dtype: Any) -> Any:
    """Cast every floating leaf in a pytree to one explicit dtype."""

    target = require_floating_dtype(dtype)

    def cast(value: Any) -> Any:
        array = jnp.asarray(value)
        return array.astype(target) if jnp.issubdtype(array.dtype, jnp.inexact) else array

    result = jax.tree_util.tree_map(cast, tree)
    mismatched = {
        str(jnp.asarray(value).dtype)
        for value in jax.tree_util.tree_leaves(result)
        if jnp.issubdtype(jnp.asarray(value).dtype, jnp.inexact)
        and jnp.asarray(value).dtype != target
    }
    if mismatched:
        raise TypeError(
            f"floating parameter leaves must all use {target}; found {sorted(mismatched)}"
        )
    return result
