"""Chunked accumulation of the LM normal equations, shared by every PDE."""

from __future__ import annotations

from typing import Any, Iterable, Iterator

import jax
import jax.numpy as jnp


@jax.jit
def _absorb(product, rhs, loss, jacobian_block, residual_block):
    """Fold one (J_block, r_block) pair into the running accumulators."""

    return (
        product + jacobian_block.T @ jacobian_block,
        rhs + jacobian_block.T @ residual_block,
        loss + jnp.sum(residual_block**2),
    )


def accumulate_normal_equations(
    p, blocks: Iterable[tuple[Any, Any]]
) -> tuple[Any, Any, Any]:
    """Return ``(J^T J, J^T r, loss)`` from (J_block, r_block) pairs."""

    n_params = p.shape[0]
    product = jnp.zeros((n_params, n_params), dtype=p.dtype)
    rhs = jnp.zeros((n_params,), dtype=p.dtype)
    loss = jnp.zeros((), dtype=p.dtype)
    for jacobian_block, residual_block in blocks:
        product, rhs, loss = _absorb(
            product, rhs, loss, jacobian_block, residual_block
        )
    return product, rhs, loss


def accumulate_loss(p, residual_blocks: Iterable[Any]):
    """Return ``||r||^2`` from an iterable of weighted residual blocks."""

    loss = jnp.zeros((), dtype=p.dtype)
    for residual_block in residual_blocks:
        loss = loss + jnp.sum(residual_block**2)
    return loss


def chunk_slices(total: int, chunk: int) -> Iterator[slice]:
    """Yield the slices of a chunked sweep over ``total`` collocation points."""

    if chunk <= 0:
        raise ValueError("chunk must be a positive integer")
    for start in range(0, total, chunk):
        yield slice(start, min(start + chunk, total))