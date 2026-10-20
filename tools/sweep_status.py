#!/usr/bin/env python
"""Report on a sweep of runs, one block per equation."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

DEFAULT_ROOT = "results/main"

EQUATION_ORDER = ("kg2d", "nstg40", "nstg200", "kg3d", "ns3d")
EQUATION_LABEL = {
    "kg2d": "(2+1) Klein-Gordon",
    "kg3d": "(3+1) Klein-Gordon",
    "nstg40": "(2+1) Navier-Stokes  Re 40",
    "nstg200": "(2+1) Navier-Stokes  Re 200",
    "ns3d": "(3+1) Navier-Stokes",
}


def grid_shape(profile: str) -> tuple[str, int]:
    """Turn a profile name into a printable shape and a point count."""

    cubed = re.search(r"(\d+)cubed_x_(\d+)", profile)
    if cubed:
        n, nt = int(cubed.group(1)), int(cubed.group(2))
        return f"{n}x{n}x{n}x{nt}", n**3 * nt
    square = re.search(r"(\d+)sq_x_(\d+)", profile)
    if square:
        n, nt = int(square.group(1)), int(square.group(2))
        return f"{n}x{n}x{nt}", n**2 * nt
    numbers = [int(v) for v in re.findall(r"\d+", profile)]
    return profile, (numbers[0] if numbers else 0)


def describe(manifest) -> str:
    """What a panel actually ran, from its manifest rather than from a guess."""

    method = manifest.get("method", {})
    model = method.get("model", {}) or {}
    bits = []
    optimizer = manifest.get("optimizer", "lm")
    if optimizer != "lm":
        bits.append(optimizer)
    if "layer_dims" in model:
        bits.append(",".join(str(v) for v in model["layer_dims"]))
    if "r" in model:
        bits.append(f"r{model['r']}")
    if not bits:
        bits.append(method.get("architecture", "?"))
    return " ".join(bits)


def to_six(trace, evals: int):
    """Reduce a trace to ``evals`` points, keeping the final evaluation."""

    if len(trace) <= evals:
        return trace
    final = trace[-1]
    candidates = trace[:-1]
    chosen = []
    for index in range(1, evals):
        wanted = final["step"] * index / evals
        nearest = min(candidates, key=lambda e: abs(e["step"] - wanted))
        if nearest not in chosen:
            chosen.append(nearest)
    chosen.sort(key=lambda e: e["step"])
    return chosen + [final]


def average(traces, metric: str):
    """Average the seeds column by column, over the columns they all reached."""

    width = min(len(t) for t in traces)
    averaged = []
    for index in range(width):
        entries = [t[index] for t in traces]
        values = [e[metric] for e in entries if e.get(metric) is not None]
        averaged.append({
            "step": sum(e["step"] for e in entries) // len(entries),
            "elapsed_seconds": sum(e["elapsed_seconds"] for e in entries) / len(entries),
            "weighted_loss": sum(e.get("weighted_loss", float("nan"))
                                 for e in entries) / len(entries),
            metric: (sum(values) / len(values)) if values else None,
            "n_seeds": len(entries),
            "spread": (max(values) / min(values)
                       if len(values) > 1 and min(values) > 0 else 1.0),
        })
    return averaged


def read_run(directory: Path, evals: int = 6, metric: str = "relative_l2"):
    """One panel's runs, averaged across whatever seeds it holds."""

    trace_paths = sorted(directory.glob("*/seed_*/trace.json"))
    if not trace_paths:
        return None
    traces, manifests = [], []
    for trace_path in trace_paths:
        raw = json.loads(trace_path.read_text(encoding="utf-8"))
        # Skip step 0, the untrained model.
        traces.append(to_six([e for e in raw if e["step"] > 0], evals))
        manifest_path = trace_path.parent / "manifest.json"
        manifests.append(
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file() else {}
        )
    seeds = [int(p.parent.name.split("_")[1]) for p in trace_paths]
    first = manifests[0]
    finished = [m for m in manifests if "total_steps" in m]
    return {
        "trace": average(traces, metric),
        "manifest": first,
        "seeds": seeds,
        "parameters": first.get("parameter_count", 0),
        # Done only when every seed is.
        "finished": len(finished) == len(manifests),
        "total_steps": (
            sum(m["total_steps"] for m in finished) // len(finished)
            if finished else None
        ),
        "seconds_per_step": first.get("calibrated_seconds_per_step"),
        "seconds_per_eval": first.get("calibrated_seconds_per_eval", 0.0),
        "budget_hours": sum(first.get("budget_hours", [])) or None,
        "mtime": max(p.stat().st_mtime for p in trace_paths),
    }


def target_steps(run):
    if run["total_steps"]:
        return run["total_steps"]
    hours = run["budget_hours"]
    if not run["seconds_per_step"] or not hours:
        return None
    training = hours * 3600.0 - 6 * run["seconds_per_eval"]
    return max(1, int(round(training / run["seconds_per_step"])))


def duration(seconds):
    if seconds is None or seconds != seconds or seconds < 0:
        return "-"
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h{rest // 60:02d}m"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=DEFAULT_ROOT)
    parser.add_argument("--metric", default="relative_l2")
    parser.add_argument("--evals", type=int, default=6,
                        help="columns to show; a trace with more is reduced to "
                             "this many, keeping the final evaluation")
    parser.add_argument("--spread", action="store_true",
                        help="under each averaged row, the max/min ratio across "
                             "seeds; above about 3 the average is hiding a "
                             "disagreement larger than the gaps being compared")
    parser.add_argument("--times", action="store_true",
                        help="print the elapsed hours of each evaluation")
    parser.add_argument("--loss", action="store_true")
    parser.add_argument("--curves", action="store_true")
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print(f"no such directory: {root}")
        return 1

    cells = {}
    for panel_dir in sorted(root.glob("*/*/*/")):
        equation, profile, panel = panel_dir.parts[-3:]
        run = read_run(panel_dir, args.evals, args.metric)
        if run:
            cells[(equation, profile, panel)] = run
    if not cells:
        print(f"no runs under {root}")
        return 1

    equations = sorted(
        {key[0] for key in cells},
        key=lambda e: (EQUATION_ORDER.index(e) if e in EQUATION_ORDER else 99, e),
    )
    panels = sorted({key[2] for key in cells})
    finished = sum(1 for run in cells.values() if run["finished"])
    now = time.time()

    print("=" * 100)
    total_seeds = sum(len(run["seeds"]) for run in cells.values())
    print(f"{root}   {len(cells)} panels / {total_seeds} runs, "
          f"{finished} panels complete   metric: {args.metric}")
    print("values are averaged over the seeds a panel holds; "
          "'x3' after a panel name is the seed count")
    print("=" * 100)

    for equation in equations:
        profiles = sorted(
            {key[1] for key in cells if key[0] == equation},
            key=lambda p: grid_shape(p)[1],
        )
        width = max(
            len(cells[(equation, p, panel)]["trace"])
            for p in profiles for panel in panels
            if (equation, p, panel) in cells
        )
        print()
        print(f"═══ {EQUATION_LABEL.get(equation, equation)} ═══")
        print(f"  {'panel':7s} {'configuration':20s} {'P':>7s} "
              f"{'step':>9s}/{'target':<9s} {'state':>7s}"
              + "".join(f"{'eval ' + str(i + 1):>12s}" for i in range(width)))
        for profile in profiles:
            here = {
                panel: cells[(equation, profile, panel)]
                for panel in panels if (equation, profile, panel) in cells
            }
            if not here:
                continue
            shape, points = grid_shape(profile)
            budget = next(
                (r["budget_hours"] for r in here.values() if r["budget_hours"]), None
            )
            line = f"  {shape}   {points:,} points"
            if budget:
                line += f"   budget {budget:g} h"
            print(f"  {'-' * 4} {line.strip()} {'-' * 4}")
            for panel in panels:
                run = here.get(panel)
                if run is None:
                    print(f"  {panel:7s} {'(queued)':>20s}")
                    continue
                trace = run["trace"]
                total = target_steps(run)
                last = trace[-1] if trace else {"step": 0, "elapsed_seconds": 0.0}
                if run["finished"]:
                    state = "done"
                elif now - run["mtime"] > 3600:
                    state = "stalled"
                elif total:
                    state = f"{100 * last['step'] / total:3.0f}%"
                else:
                    state = "?"
                # Mark the seed count.
                label = panel if len(run["seeds"]) < 2 else f"{panel}x{len(run['seeds'])}"
                row = (f"  {label:7s} {describe(run['manifest'])[:20]:20s} "
                       f"{run['parameters']:7,} "
                       f"{last['step']:>9,}/{total or 0:<9,} {state:>7s}")
                for entry in trace:
                    value = entry.get(args.metric)
                    row += f"{value:12.3e}" if value is not None else " " * 12
                print(row)
                if len(run["seeds"]) > 1 and args.spread:
                    print(" " * 57 + "".join(
                        f"{e['spread']:11.2f}x" for e in trace
                    ))
                if args.times:
                    print(" " * 57 + "".join(
                        f"{e['elapsed_seconds'] / 3600:11.2f}h" for e in trace
                    ))
                if args.loss:
                    loss_row = " " * 57
                    for entry in trace:
                        loss_row += f"{entry.get('weighted_loss', float('nan')):12.3e}"
                    print(loss_row)

    print()
    print("═══ workers ═══")
    for panel in panels:
        runs = [(key, run) for key, run in cells.items() if key[2] == panel]
        active = [(key, run) for key, run in runs if not run["finished"]]
        done = sum(1 for _, run in runs if run["finished"])
        if not active:
            print(f"  {panel:7s} {done} finished, nothing active")
            continue
        for (equation, profile, _), run in active:
            total = target_steps(run)
            trace = run["trace"]
            last = trace[-1] if trace else {"step": 0, "elapsed_seconds": 0.0}
            left = (
                (total - last["step"]) * last["elapsed_seconds"] / last["step"]
                if last["step"] and total else None
            )
            age = int((now - run["mtime"]) / 60)
            print(f"  {panel:7s} {equation}  {grid_shape(profile)[0]}  "
                  f"step {last['step']:,}/{total or 0:,}  "
                  f"{duration(last['elapsed_seconds'])} in, {duration(left)} left"
                  + ("   STALLED" if age > 60 else f"  (wrote {age} min ago)"))

    if args.curves:
        for key in sorted(cells):
            run = cells[key]
            print()
            print(f"═══ {key[0]} / {grid_shape(key[1])[0]} / {key[2]}   "
                  f"{describe(run['manifest'])}   P={run['parameters']:,} ═══")
            for entry in run["trace"]:
                value = entry.get(args.metric, float("nan"))
                print(f"  {entry['step']:>10,} {entry['elapsed_seconds']/3600:6.2f} h  "
                      f"loss {entry['weighted_loss']:.4e}  {args.metric} {value:.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())