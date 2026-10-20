"""Adam, as SPINN is trained: Optax Adam on the block-MSE objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax


@dataclass(frozen=True)
class AdamHParams:
    """All arguments of ``optax.adam`` that affect numerical updates."""

    learning_rate: float = 1e-3
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    eps_root: float = 0.0
    mu_dtype: Any = None
    nesterov: bool = True

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.b1 < 1 or not 0 <= self.b2 < 1:
            raise ValueError("b1 and b2 must be in [0, 1)")
        if self.eps < 0 or self.eps_root < 0:
            raise ValueError("eps and eps_root must be non-negative")
        if not isinstance(self.nesterov, bool):
            raise ValueError("nesterov must be boolean")


def block_mse_loss_and_grad(system):
    """The objective ``sum_b scale_b^2 mean(r_b^2)`` and its gradient."""

    scales = system["residual_scales"]
    order = system["residual_order"]
    block_mse_fn = system["block_mse_fn"]

    def objective(parameters):
        blocks = block_mse_fn(parameters)
        return sum(scales[name] ** 2 * blocks[name] for name in order)

    return jax.jit(jax.value_and_grad(objective))


def make_optimizer(hparams: AdamHParams):
    """Construct exactly the declared Optax transformation."""

    return optax.adam(
        learning_rate=hparams.learning_rate,
        b1=hparams.b1,
        b2=hparams.b2,
        eps=hparams.eps,
        eps_root=hparams.eps_root,
        mu_dtype=hparams.mu_dtype,
        nesterov=hparams.nesterov,
    )


def _tree_is_finite(tree: Any) -> Any:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


def train_adam(
    loss_and_grad: Callable[[Any], tuple[Any, Any]],
    parameters: Any,
    n_steps: int,
    *,
    hparams: AdamHParams | None = None,
    initial_optimizer_state: Any | None = None,
    initial_history: list[float] | None = None,
    log_every: int = 1000,
    log: Callable[[str], None] = print,
    checkpoint_callback: Callable[[int, Any, Any, list[float]], None] | None = None,
):
    """Run Adam until ``n_steps`` updates have been made in total."""

    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    if isinstance(log_every, bool) or not isinstance(log_every, int) or log_every <= 0:
        raise ValueError("log_every must be a positive integer")
    hp = hparams or AdamHParams()
    optimizer = make_optimizer(hp)
    optimizer_state = (
        optimizer.init(parameters)
        if initial_optimizer_state is None
        else initial_optimizer_state
    )
    history = list(initial_history or [])
    completed = len(history)
    if completed > n_steps:
        raise ValueError(
            f"Adam start step {completed} exceeds requested steps {n_steps}"
        )

    @jax.jit
    def update_once(current_parameters, current_state):
        loss, gradient = loss_and_grad(current_parameters)
        updates, next_state = optimizer.update(gradient, current_state)
        next_parameters = optax.apply_updates(current_parameters, updates)
        finite = (
            jnp.isfinite(loss)
            & _tree_is_finite(gradient)
            & _tree_is_finite(updates)
            & _tree_is_finite(next_parameters)
            & _tree_is_finite(next_state)
        )
        return next_parameters, next_state, loss, finite

    if not bool(jax.device_get(_tree_is_finite(parameters))):
        raise FloatingPointError("initial Adam parameters contain NaN or Inf")
    if not bool(jax.device_get(_tree_is_finite(optimizer_state))):
        raise FloatingPointError("initial Adam optimizer state contains NaN or Inf")

    for index in range(completed, n_steps):
        parameters, optimizer_state, loss, finite = update_once(
            parameters, optimizer_state
        )
        if not bool(jax.device_get(finite)):
            raise FloatingPointError(
                f"Adam produced NaN or Inf while attempting update {index + 1}"
            )
        loss_value = float(jax.device_get(loss))
        history.append(loss_value)
        if checkpoint_callback is not None:
            checkpoint_callback(index + 1, parameters, optimizer_state, history)
        if index % log_every == 0 or index == n_steps - 1:
            log(f"  Adam step {index:6d} | loss {loss_value:.4e}")

    return parameters, optimizer_state, history
