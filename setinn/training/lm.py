"""Levenberg-Marquardt on a flat parameter vector, as used to train TINN."""

from dataclasses import dataclass
import warnings

import jax
import jax.numpy as jnp


@dataclass
class LMHParams:
    mu0: float = 10.0
    mul_factor: float = 1.7
    div_factor: float = 1.3
    min_mu: float = 1e-12
    max_mu: float = 1e8
    safeguard_ratio: float = 1e5   # loss / mu > 1e5  ->  mu = loss / 10

    #: The acceptance rule, as recorded in a protocol.
    SEMANTICS = "classic_reject_uphill"

    def __post_init__(self):
        if self.mu0 <= 0 or self.min_mu <= 0 or self.max_mu < self.min_mu:
            raise ValueError("LM damping bounds must satisfy 0 < min_mu <= max_mu")
        if self.mul_factor <= 1 or self.div_factor <= 1:
            raise ValueError("LM damping factors must be greater than one")

    @classmethod
    def from_semantics(cls, semantics: str, **kwargs):
        """Hyperparameters for the acceptance rule a protocol names."""
        if semantics != cls.SEMANTICS:
            raise ValueError(f"unknown LM semantics: {semantics}")
        return cls(**kwargs)


@dataclass
class LMState:
    mu: float
    loss_old: float
    step: int = 0


def init_lm_state(hp: LMHParams) -> LMState:
    return LMState(mu=hp.mu0, loss_old=1e5, step=0)


def _check_x64():
    if not jax.config.read("jax_enable_x64"):
        warnings.warn("The LM path requires x64 (jax.config.update('jax_enable_x64', True)). "
                      "Running in x32 risks an unstable linear solve.", RuntimeWarning)


def lm_step(p, state: LMState, hp: LMHParams, normal_equations_fn, loss_fn):
    """One LM step: ``(p, state) -> (p_new, state_new, loss of p_new)``."""

    J_product, Jt_residual, loss_value = normal_equations_fn(p)
    loss = float(loss_value)
    n_params = J_product.shape[0]
    matrix_dtype = J_product.dtype
    rhs = -Jt_residual

    I = jnp.eye(n_params, dtype=matrix_dtype)
    dp = jnp.linalg.solve(J_product + state.mu * I, rhs)
    p_cand = p + dp

    loss_cand = float(loss_fn(p_cand))
    if loss_cand < loss:
        p_new = p_cand
        mu = max(state.mu / hp.div_factor, hp.min_mu)
        reported = loss_cand
    else:
        p_new = p                                      # reject the step
        mu = min(hp.mul_factor * state.mu, hp.max_mu)
        reported = loss
    if reported / mu > hp.safeguard_ratio:
        mu = min(max(reported / 10.0, hp.min_mu), hp.max_mu)
    return p_new, LMState(mu=mu, loss_old=reported, step=state.step + 1), reported


def train_lm(
    p0,
    n_steps,
    *,
    normal_equations_fn,
    loss_fn,
    hp=None,
    log_every=50,
    log=print,
    initial_state=None,
    initial_history=None,
    checkpoint_callback=None,
    return_state=False,
):
    """Run LM until ``n_steps`` steps have been taken in total."""
    hp = hp or LMHParams()
    state = initial_state or init_lm_state(hp)
    p = p0
    history = list(initial_history or [])
    if state.step != len(history):
        raise ValueError(
            f"LM state/history mismatch: state.step={state.step}, history={len(history)}"
        )
    if state.step > n_steps:
        raise ValueError(
            f"LM start step {state.step} exceeds requested steps {n_steps}"
        )
    for step in range(state.step, n_steps):
        p, state, loss = lm_step(p, state, hp, normal_equations_fn, loss_fn)
        finite = jnp.all(jnp.isfinite(p))
        finite = finite & jnp.isfinite(loss)
        finite = finite & jnp.isfinite(state.mu) & jnp.isfinite(state.loss_old)
        if not bool(jax.device_get(finite)):
            raise FloatingPointError(
                f"LM produced NaN or Inf at completed step {step + 1}"
            )
        history.append(loss)
        if checkpoint_callback is not None:
            checkpoint_callback(step + 1, p, state, history)
        if step % log_every == 0 or step == n_steps - 1:
            log(f"  LM step {step:4d} | loss {loss:.4e} | mu {state.mu:.2e}")
    if return_state:
        return p, state, history
    return p, history


_check_x64()  # warn once on module import
