#!/usr/bin/env python
"""Train one architecture on one PDE under a wall-clock budget."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# CUDA graphs need memory outside the allocator pool.
if "xla_gpu_enable_command_buffer" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
    ).strip()

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from setinn.models import FAMILIES, build_model  # noqa: E402
from setinn.problems import PROBLEMS  # noqa: E402
from setinn.protocol import (  # noqa: E402
    ProtocolError,
    build_protocol,
    make_method,
    protocol_hash,
    sample_seed_for,
)
from setinn.provenance import (  # noqa: E402
    atomic_json,
    implementation_hash,
    runtime_fingerprint,
    sha256_file,
    source_hashes,
)
from setinn.training.adam import (  # noqa: E402
    AdamHParams,
    block_mse_loss_and_grad,
    make_optimizer,
    train_adam,
)
from setinn.training.lm import (  # noqa: E402
    LMHParams,
    LMState,
    init_lm_state,
    train_lm,
)

# Calibration: a discarded warm-up, then a timed sample.
CALIBRATION_STEPS = 500
CALIBRATION_WARMUP_FRACTION = 0.5
# Wall-clock cap on calibration.
CALIBRATION_SECONDS = 300.0


# --- protocol ---------------------------------------------------------------


def _with_output_width(layer_dims, n_out: int) -> list[int]:
    """Set the final entry of ``layer_dims`` to the problem's output width."""

    dims = [int(value) for value in layer_dims]
    if len(dims) < 3:
        raise SystemExit(f"layer_dims needs at least three entries, got {dims}")
    return dims[:-1] + [int(n_out)]


def protocol_for(args, spec):
    """The protocol this command line describes."""

    family, _, axes = args.arm.partition("_")
    if family not in FAMILIES:
        raise SystemExit(
            f"unknown arm {args.arm!r}; choose one of {', '.join(FAMILIES)}"
        )
    if axes and family not in ("setinn", "setinnall"):
        raise SystemExit(f"only SeTINN arms take induced axes, got {args.arm!r}")
    method = make_method(
        spec,
        family,
        optimizer=args.optimizer,
        mod_axes=list(axes) if axes else None,
        wa_init_scale=args.wa_init_scale,
        layer_dims=(
            _with_output_width(args.layer_dims.split(","), spec.n_out)
            if args.layer_dims
            else None
        ),
        rank=args.rank,
    )
    settings = {}
    if args.t_max is not None:
        settings["t_max"] = float(args.t_max)
    if args.nu is not None:
        # Re = 1/nu.
        settings["nu"] = float(args.nu)
    return build_protocol(
        spec,
        method,
        seeds=args.seeds,
        profile=args.profile,
        settings=settings,
    )


# --- checkpointing ----------------------------------------------------------


def save(run_dir: Path, manifest, parameters, state, history, trace):
    """Write the parameters, optimizer state, loss history, trace and manifest."""

    np.save(run_dir / "parameters.npy", np.asarray(jax.device_get(parameters)))
    np.save(run_dir / "loss_history.npy", np.asarray(history, dtype=np.float64))
    if isinstance(state, LMState):
        atomic_json(
            run_dir / "lm_state.json",
            {"mu": float(state.mu), "loss_old": float(state.loss_old),
             "step": int(state.step)},
        )
    else:
        # Adam's state is stored as its pytree leaves.
        leaves = jax.tree_util.tree_leaves(state)
        np.savez(
            run_dir / "adam_state.npz",
            **{f"leaf_{index}": np.asarray(jax.device_get(leaf))
               for index, leaf in enumerate(leaves)},
        )
    atomic_json(run_dir / "trace.json", trace)
    atomic_json(run_dir / "manifest.json", manifest)


def _settings_hash(protocol) -> str:
    """Protocol hash without ``seeds`` and ``protocol_id``."""

    return protocol_hash(
        {key: value for key, value in protocol.items()
         if key not in ("seeds", "protocol_id")}
    )


def already_complete(run_dir: Path, hours: float, protocol) -> bool:
    """True when this directory holds a finished run of at least ``hours``."""

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if "total_steps" not in manifest:
        return False
    try:
        stored = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        stored = None
    if stored is None or _settings_hash(stored) != _settings_hash(protocol):
        raise SystemExit(
            f"{run_dir} holds a finished run with other settings; "
            "choose another --out"
        )
    return sum(manifest.get("budget_hours", [])) + 1e-9 >= hours


# --- one seed ---------------------------------------------------------------


def run_seed(args, spec, protocol, method, seed, out_dir):
    run_dir = out_dir / method["id"] / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training.log"

    def log(message):
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    if args.skip_finished and already_complete(run_dir, args.hours, protocol):
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        final = manifest.get("final", {})
        print(f"[{method['id']} seed {seed}] already finished "
              f"({manifest['total_steps']:,} steps, "
              f"rel_l2 {final.get('relative_l2', float('nan')):.4e}); skipping",
              flush=True)
        return final

    protocol_path = run_dir / "protocol.json"
    atomic_json(protocol_path, protocol)

    dtype = jnp.float64
    points = spec.sample_collocation(
        {**protocol["sampling"], "seed": sample_seed_for(protocol, seed)},
        protocol["problem"],
    )
    collocation_sha = spec.collocation_sha256(points)

    model = build_model(
        method["architecture"],
        problem=spec,
        config=method.get("model", {}),
        seed=seed,
        dtype=dtype,
    )
    expected = int(method["expected_parameter_count"])
    if model.n_params != expected:
        raise SystemExit(
            f"parameter-count drift for {method['id']}: {model.n_params} != {expected}"
        )

    system = spec.make_system(
        model, points, protocol["problem"], protocol["sampling"],
        protocol["residual_scales"], dtype=dtype,
    )
    normal_equations_fn = system["normal_equations_fn"]
    loss_fn = system["loss_fn"]

    def loss_at(p):
        return float(jax.device_get(loss_fn(p)))

    def score(p, keep_fields=False):
        result = spec.evaluate(
            model, p, protocol["problem"], protocol["evaluation"], dtype=dtype
        )
        metrics = {
            k: float(v) for k, v in result.metrics.items() if isinstance(v, (int, float))
        }
        if not keep_fields:
            return metrics
        return metrics, {
            name: np.asarray(value, dtype=np.float64)
            for name, value in result.arrays.items()
        }

    training = method["training"]
    use_adam = method["optimizer"] == "adam"
    if use_adam:
        hp = AdamHParams(
            learning_rate=float(training["learning_rate"]),
            b1=float(training["b1"]),
            b2=float(training["b2"]),
            eps=float(training["eps"]),
            nesterov=bool(training["nesterov"]),
        )
        loss_and_grad = block_mse_loss_and_grad(system)
        optimizer_template = make_optimizer(hp).init(model.p0)
    else:
        hp = LMHParams.from_semantics(
            training["semantics"], mu0=float(training["mu0"])
        )
        loss_and_grad = None
        optimizer_template = None

    sources = source_hashes()
    manifest = {
        "schema_version": 2,
        "problem": spec.key,
        "method": method,
        "seed": seed,
        "sample_seed": sample_seed_for(protocol, seed),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash(protocol),
        "protocol_path": str(protocol_path),
        "collocation_sha256": collocation_sha,
        "formula_sha256": spec.formula_sha256(protocol["problem"]),
        "source_sha256": sources,
        "runtime": runtime_fingerprint(jax),
        "parameter_count": int(model.n_params),
        "budget_hours": [],
        "optimizer": method["optimizer"],
    }

    log(f"[{method['id']} seed {seed}] P={model.n_params:,}  collocation={collocation_sha[:12]}")

    # Compile only what the optimizer calls.
    warm = time.perf_counter()
    if use_adam:
        jax.block_until_ready(loss_and_grad(model.p0))
    else:
        jax.block_until_ready(normal_equations_fn(model.p0))
    log(f"  warmup {time.perf_counter() - warm:.1f} s")

    def take(steps, start_parameters, start_state, start_history):
        """Train up to ``steps`` total; return (elapsed, parameters, state, history)."""

        begin = time.perf_counter()
        if use_adam:
            result = train_adam(
                loss_and_grad, start_parameters, steps, hparams=hp,
                initial_optimizer_state=start_state,
                initial_history=start_history,
                log_every=steps + 1, log=lambda _m: None,
            )
        else:
            result = train_lm(
                start_parameters, steps,
                normal_equations_fn=normal_equations_fn, loss_fn=loss_fn, hp=hp,
                log_every=steps + 1, log=lambda _m: None,
                initial_state=start_state, initial_history=start_history,
                return_state=True,
            )
        return (time.perf_counter() - begin, *result)

    def fresh_state():
        return optimizer_template if use_adam else init_lm_state(hp)

    # A short probe sizes the timed stretches.
    probe = max(1, CALIBRATION_STEPS // 25)
    probe_seconds, probe_parameters, probe_state, probe_history = take(
        probe, model.p0, fresh_state(), []
    )
    per_step = probe_seconds / probe
    affordable = int(CALIBRATION_SECONDS / max(per_step, 1e-9))
    # At least two probes.
    planned = max(2 * probe, min(CALIBRATION_STEPS, affordable))
    warmup_steps = int(planned * CALIBRATION_WARMUP_FRACTION)
    sample_steps = planned - warmup_steps

    if sample_steps <= 0:
        seconds_per_step = per_step
        warmup_steps, sample_steps = probe, 0
    else:
        parameters_warm, state_warm, history_warm = (
            probe_parameters, probe_state, probe_history
        )
        if warmup_steps > probe:
            _, parameters_warm, state_warm, history_warm = take(
                warmup_steps, model.p0, fresh_state(), []
            )
        completed = len(history_warm)
        # Two stretch lengths cancel the compile time: t(n) = C + n s.
        short_steps = max(1, sample_steps // 3)
        short_seconds, *_ = take(
            completed + short_steps,
            parameters_warm, state_warm, history_warm,
        )
        long_seconds, *_ = take(
            completed + sample_steps,
            parameters_warm, state_warm, history_warm,
        )
        span = sample_steps - short_steps
        if span > 0 and long_seconds > short_seconds:
            seconds_per_step = (long_seconds - short_seconds) / span
        else:
            seconds_per_step = long_seconds / sample_steps
        manifest["calibration_compile_seconds"] = max(
            0.0, short_seconds - short_steps * seconds_per_step
        )

    del probe_parameters, probe_state, probe_history
    manifest["calibration_warmup_steps"] = int(warmup_steps)
    manifest["calibration_sample_steps"] = int(sample_steps)
    calibration_steps = int(warmup_steps + sample_steps)
    manifest["calibration_steps"] = calibration_steps
    parameters, state, history = model.p0, fresh_state(), []
    # Time one evaluation.
    score_start = time.perf_counter()
    first = score(model.p0)
    seconds_per_eval = time.perf_counter() - score_start
    trace = [{"step": 0, "elapsed_seconds": 0.0,
              "weighted_loss": loss_at(model.p0), **first}]
    manifest["calibrated_seconds_per_step"] = seconds_per_step
    manifest["calibrated_seconds_per_eval"] = seconds_per_eval
    log(f"  calibration: {seconds_per_step:.4f} s/step from {sample_steps} steps "
        f"after a {warmup_steps}-step warm-up")

    # Evaluations come out of the budget first.
    training_seconds = args.hours * 3600.0 - args.evals * seconds_per_eval
    if training_seconds <= 0:
        raise SystemExit(
            f"budget of {args.hours} h is consumed by {args.evals} evaluations "
            f"at {seconds_per_eval:.1f} s each"
        )
    target = max(1, int(round(training_seconds / seconds_per_step)))
    manifest["budget_hours"].append(args.hours)
    manifest["evaluation_overhead_seconds"] = args.evals * seconds_per_eval
    manifest["implementation_sha256"] = implementation_hash(sources)
    log(f"  calibration: {seconds_per_eval:.2f} s/eval x {args.evals} evals "
        f"= {args.evals * seconds_per_eval / 60:.1f} min reserved")
    log(f"  budget {args.hours} h -> {target:,} training steps")

    started = time.perf_counter()
    every = max(1, target // args.evals)

    def record(step, loss):
        metrics = score(parameters_ref[0])
        entry = {"step": int(step), "elapsed_seconds": time.perf_counter() - started,
                 "weighted_loss": float(loss), **metrics}
        trace.append(entry)
        log(f"  step {step:>9,}  {entry['elapsed_seconds']/3600:6.2f} h  "
            f"loss {float(loss):.4e}  rel_l2 {metrics.get('relative_l2', float('nan')):.4e}")

    parameters_ref = [parameters]

    def checkpoint(step, p, st, hist):
        parameters_ref[0] = p
        # Evaluate and save every ``every`` steps before the final evaluation.
        if step % every == 0 and step < every * args.evals:
            record(step, hist[-1])
            save(run_dir, manifest, p, st, hist, trace)

    if use_adam:
        parameters, state, history = train_adam(
            loss_and_grad, parameters, target, hparams=hp,
            initial_optimizer_state=state, initial_history=history,
            log_every=max(1, target // 20), log=log,
            checkpoint_callback=checkpoint,
        )
    else:
        parameters, state, history = train_lm(
            parameters, target,
            normal_equations_fn=normal_equations_fn, loss_fn=loss_fn, hp=hp,
            log_every=max(1, target // 20), log=log,
            initial_state=state, initial_history=history,
            checkpoint_callback=checkpoint, return_state=True,
        )
    parameters_ref[0] = parameters
    if not trace or trace[-1]["step"] != target:
        record(target, history[-1])

    # Fields on the evaluation grid, from the final parameters.
    _, fields = score(parameters, keep_fields=True)
    truth = fields.get("exact", fields.get("reference"))
    prediction = fields["prediction"]
    absolute_error = np.abs(prediction - truth)
    export = dict(fields)
    export.setdefault("exact", truth)
    export.setdefault("reference", truth)
    export["absolute_error"] = absolute_error
    if prediction.ndim >= 2:
        # Time-mean error and per-entry relative L2 over the last field axis.
        export["absolute_error_time_mean"] = absolute_error.mean(axis=-1)
        denominators = np.sqrt((truth**2).sum(axis=tuple(range(truth.ndim - 1))))
        numerators = np.sqrt(
            (absolute_error**2).sum(axis=tuple(range(truth.ndim - 1)))
        )
        export["l2_of_time"] = np.where(
            denominators > 0, numerators / np.maximum(denominators, 1e-300), np.nan
        )
    for key in ("step", "elapsed_seconds", "weighted_loss", "relative_l2"):
        values = [entry[key] for entry in trace if key in entry]
        if values:
            export[f"trace_{key}"] = np.asarray(values, dtype=np.float64)
    export["loss_history"] = np.asarray(history, dtype=np.float64)
    np.savez_compressed(run_dir / "evaluation.npz", **export)

    manifest["final"] = trace[-1]
    manifest["total_steps"] = int(len(history) if use_adam else state.step)
    manifest["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(run_dir, manifest, parameters, state, history, trace)
    log(f"  done: {manifest['total_steps']:,} steps, "
        f"rel_l2 {trace[-1].get('relative_l2', float('nan')):.4e}")
    manifest["artifacts"] = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    atomic_json(run_dir / "manifest.json", manifest)
    return trace[-1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--problem", default="kg2d", choices=sorted(PROBLEMS))
    parser.add_argument(
        "--arm",
        required=True,
        help="tinn, spinn, setinn (first layer induced) or setinnall (every "
             "layer); SeTINN may name its induced axes, e.g. setinn_xy",
    )
    parser.add_argument("--layer-dims", default=None,
                        help="one SPINN/SeTINN axis body (default 1,10,10,1, the "
                             "paper's SeTINN); the final entry is replaced by the "
                             "number of outputs")
    parser.add_argument("--rank", type=int, default=None,
                        help="separation rank r (default 5, the paper's SeTINN)")
    parser.add_argument("--wa-init-scale", type=float, default=0.0,
                        help="initial scale of the SeTINN lift A (0: start as SPINN)")
    parser.add_argument("--profile", required=True,
                        help="tensor grid: tensor_50sq_x_50 or tensor_100sq_x_100 "
                             "(2-D), tensor_16cubed_x_16 or tensor_25cubed_x_25 (3-D)")
    parser.add_argument("--t-max", type=float, default=None,
                        help="time horizon (default: the paper's, e.g. 10 for KG2D)")
    parser.add_argument("--nu", type=float, default=None,
                        help="viscosity of the Taylor-Green problem, Re = 1/nu")
    parser.add_argument(
        "--optimizer",
        choices=("lm", "adam"),
        default="lm",
        help="lm as in TINN; adam as in SPINN, whose network is too large for "
             "LM (an Adam step costs O(P), an LM step O(M P^2))",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[997, 998, 999])
    parser.add_argument("--hours", type=float, required=True, help="wall-clock budget per seed")
    parser.add_argument("--evals", type=int, default=6, help="evaluations during the run")
    parser.add_argument(
        "--skip-finished",
        action="store_true",
        help="leave runs that already have this budget alone instead of "
             "training them again; stop if a finished run there has other "
             "settings",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    spec = PROBLEMS[args.problem]
    try:
        protocol = protocol_for(args, spec)
    except ProtocolError as error:
        parser.error(str(error))
    method = protocol["methods"][0]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"problem   {spec.key}")
    print(f"protocol  {protocol['protocol_id']}")
    print(f"hash      {protocol_hash(protocol)}")
    print(f"method    {method['id']}  ({method['architecture']})")
    print(f"model     {method.get('model', {})}")
    print(f"sampling  {protocol['sampling']}")
    separable = build_model(
        method["architecture"], problem=spec, config=method.get("model", {}),
        seed=args.seeds[0], dtype=jnp.float64,
    ).supports_separable_residual
    strategy = protocol["sampling"]["sampling_strategy"]
    print(f"separable {separable and strategy == 'cartesian_tensor_grid'}"
          f"  (model={separable}, grid={strategy})")
    print(f"optimizer {method['optimizer']}")
    print(f"seeds     {args.seeds}   budget {args.hours} h each")
    print(f"output    {out_dir}")
    print()

    finals = {}
    for seed in args.seeds:
        finals[seed] = run_seed(args, spec, protocol, method, seed, out_dir)
        print()

    print("=== summary ===")
    for seed, entry in finals.items():
        print(f"  seed {seed}  step {entry.get('step', 0):>9,}  "
              f"rel_l2 {entry.get('relative_l2', float('nan')):.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
