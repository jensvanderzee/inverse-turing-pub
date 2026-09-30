# -*- coding: utf-8 -*-
"""
Train the original PyTorch invRietkerk model on a SINGLE real-data site.

The published real-data run (`realdata_train_invPDE.py`) fits nine coefficients
against four subsites at once. This driver runs the identical model, loss,
optimiser and schedule against one site, so the fits can be compared to the
four-site ensemble.

Nothing in the original code is modified or copied. `RealDataLoader` and
`train_model_real_data` are imported from `realdata_train_invPDE` and called
directly; the only reason this file exists rather than a one-line edit is that
the four-site driver hard-codes `num_epochs=7500` and `learning_rate=0.2` inside
`train_models_real_data`, so there is no way to reach them from its signature.

Why single-site is worth running. The four-site loss is a mean over
site x year-pair comparisons, so a one-site loss is in the same units but is not
the same objective: four sites impose four sets of constraints on one shared
parameter vector. If the coefficients are genuinely identifiable, dropping to one
site should loosen them; if the four-site agreement was an artefact of the
optimiser rather than of the data, it should persist.

Usage
-----
    python python_1site/train_1site_realdata.py --site b --models 10 --epochs 7500

    # every site in turn, 5 restarts each
    for s in a b c e f i j k; do
        python python_1site/train_1site_realdata.py --site $s --models 5
    done

Outputs (per site, under --out/site_<x>/) mirror the original layout so the
project's existing analysis scripts can read them:
    models/model_NN.pth          state dicts
    parameters/model_NN_params.pkl   parameter history (pickle, as original)
    results/training_summary.json
    progress.csv                 appended per model, so a killed run keeps its work
    trajectories/model_NN.csv    per-snapshot parameters, for plotting
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

# The original scripts live at the repository root and import each other by bare
# module name, so the root has to be importable regardless of where this is run
# from.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import realdata_train_invPDE as rt  # noqa: E402
from realdata_train_invPDE import (  # noqa: E402
    RealDataLoader,
    train_model_real_data,
)


def install_live_tracer(csv_path, comparisons_per_epoch, every, model_id):
    """
    Stream parameter values to `csv_path` while a fit is running.

    `train_model_real_data` keeps its snapshots in a local list and returns them
    only at the end, so a 7500-epoch fit is otherwise a black box for hours. It
    also prints all nine parameters every 100 epochs, but scraping stdout is
    fragile and too coarse to watch a fit turn over.

    The hook is observation only: `invRietkerk` is subclassed, `super()` does all
    the arithmetic, and the override merely counts calls. `compute_loss` invokes
    `simulate_year_weekly` exactly once per year-pair comparison, so call count
    divided by comparisons is the epoch, and the parameters seen at the first
    call of an epoch are the ones the previous optimiser step produced — the same
    convention as the original `parameter_history`.

    Returns a restore() that puts the original class back.
    """
    original = rt.invRietkerk
    state = {"calls": 0}

    # Appended, never truncated: one file carries every model's live trace, so a
    # later model cannot erase the record of an earlier one.
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as fh:
            csv.writer(fh).writerow(["model_id", "epoch", "t"] + PARAM_NAMES)
    t0 = time.time()

    class TracedRietkerk(original):
        def simulate_year_weekly(self, *a, **kw):
            if state["calls"] % comparisons_per_epoch == 0:
                epoch = state["calls"] // comparisons_per_epoch
                if epoch % every == 0:
                    vals = {n: p.item() for n, p in self.named_parameters()
                            if p.data.shape == torch.Size([1])}
                    with open(csv_path, "a", newline="") as fh:
                        csv.writer(fh).writerow(
                            [model_id, epoch, f"{time.time() - t0:.1f}"]
                            + [vals.get(n, float("nan")) for n in PARAM_NAMES])
            state["calls"] += 1
            return super().simulate_year_weekly(*a, **kw)

    rt.invRietkerk = TracedRietkerk
    return lambda: setattr(rt, "invRietkerk", original)

PARAM_NAMES = [
    "surface_water_diffusion_coeff",
    "soil_water_diffusion_coeff",
    "biomass_diffusion_coeff",
    "evaporation_rate",
    "seepage_rate",
    "mortality_rate",
    "infiltration_rate",
    "plant_uptake_rate",
    "water_use_efficiency",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", required=True,
                   help="subsite letter, e.g. b (directory data/subsite_b)")
    p.add_argument("--models", type=int, default=10, help="random restarts")
    p.add_argument("--epochs", type=int, default=7500,
                   help="Adam epochs per model (published schedule is 7500)")
    p.add_argument("--lr", type=float, default=0.2, help="Adam learning rate")
    p.add_argument("--steps-per-week", type=int, default=3,
                   help="Euler sub-steps per forcing week (published run uses 3)")
    p.add_argument("--multiplier", type=float, default=1500.0,
                   help="NDVI -> biomass scale factor")
    p.add_argument("--save-interval", type=int, default=10,
                   help="epochs between parameter snapshots")
    p.add_argument("--data-dir", default=os.path.join(REPO, "data"))
    p.add_argument("--out", default=os.path.join(REPO, "python_1site", "results"))
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--absolute-loss", action="store_true",
                   help="use absolute-value MSE instead of the delta loss")
    p.add_argument("--trace-every", type=int, default=10,
                   help="stream parameters to live.csv every N epochs (0 = off)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    outdir = os.path.join(args.out, f"site_{args.site}")
    for sub in ("models", "parameters", "results", "trajectories"):
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)

    print("=" * 72)
    print(f"Single-site training: subsite_{args.site}")
    print("=" * 72)
    print(f"device            {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"models            {args.models}")
    print(f"epochs            {args.epochs}")
    print(f"learning rate     {args.lr}")
    print(f"steps per week    {args.steps_per_week}")
    print(f"loss              {'absolute MSE' if args.absolute_loss else 'delta MSE'}")
    print(f"output            {outdir}")

    loader = RealDataLoader(args.data_dir, selected_sites=[args.site],
                            device=device, use_weekly_precip=True)
    training_data = loader.get_training_data(
        ndvi_to_biomass_multiplier=args.multiplier)

    series = training_data["location_time_series"]
    if len(series) == 0:
        raise SystemExit(f"no usable data for subsite_{args.site}")
    if len(series) != 1:
        raise SystemExit(f"expected 1 location, loaded {len(series)}: {list(series)}")

    name, ts = next(iter(series.items()))
    # Fields are (batch, channel, H, W); the spatial extent is the trailing pair.
    grid = tuple(ts[0]["biomass"].shape)[-2:]
    # One comparison per consecutive year pair; this is the denominator the loss
    # is averaged over, and it is what makes a one-site loss comparable in units
    # to the four-site one.
    print(f"\nloaded {name}: {len(ts)} observations, grid {grid[0]}x{grid[1]}, "
          f"{len(ts) - 1} year-pair comparisons")
    print(f"biomass range {float(ts[0]['biomass'].min()):.2f} - "
          f"{float(max(t['biomass'].max() for t in ts)):.2f}")

    progress_path = os.path.join(outdir, "progress.csv")
    with open(progress_path, "w", newline="") as fh:
        csv.writer(fh).writerow(
            ["model_id", "seed", "final_loss", "best_loss", "epochs", "minutes"]
            + PARAM_NAMES)

    results = []
    t_start = time.time()
    for idx in range(args.models):
        seed = 77 + idx * 102          # same seed ladder as the four-site driver
        print(f"\n--- model {idx + 1}/{args.models}  (seed {seed}) ---")
        restore = None
        if args.trace_every > 0:
            restore = install_live_tracer(
                os.path.join(outdir, "live.csv"), len(ts) - 1, args.trace_every, idx)
        t0 = time.time()
        try:
            loss_history, model, init_params, param_history = train_model_real_data(
                training_data=training_data,
                num_epochs=args.epochs,
                learning_rate=args.lr,
                device=device,
                seed=seed,
                save_interval=args.save_interval,
                steps_per_week=args.steps_per_week,
                use_delta_loss=not args.absolute_loss,
            )
        except Exception as exc:                       # one bad restart must not
            print(f"model {idx} FAILED: {exc}")        # take the batch down
            continue
        finally:
            if restore is not None:
                restore()
        minutes = (time.time() - t0) / 60.0

        final = {n: p.item() for n, p in model.named_parameters()
                 if p.data.shape == torch.Size([1])}
        finite = [l for l in loss_history if np.isfinite(l)]
        row = {
            "model_id": idx,
            "seed": seed,
            "final_loss": loss_history[-1] if loss_history else float("nan"),
            "best_loss": min(finite) if finite else float("nan"),
            "epochs": len(loss_history),
            "minutes": minutes,
            "initial_params": init_params,
            "final_params": final,
            "loss_history": loss_history,
        }
        results.append(row)

        torch.save(model.state_dict(),
                   os.path.join(outdir, "models", f"model_{idx:02d}.pth"))
        with open(os.path.join(outdir, "parameters",
                               f"model_{idx:02d}_params.pkl"), "wb") as fh:
            pickle.dump(param_history, fh)

        # Snapshot trajectory, same column layout as the Julia runs so the same
        # plotting script reads either.
        with open(os.path.join(outdir, "trajectories", f"model_{idx:02d}.csv"),
                  "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["phase", "step", "t", "loss", "gnorm"] + PARAM_NAMES)
            for snap in param_history:
                ep = int(snap.get("epoch", -1))
                lv = loss_history[ep] if 0 <= ep < len(loss_history) else float("nan")
                w.writerow(["adam-raw", ep, "", lv, ""]
                           + [snap.get(n, float("nan")) for n in PARAM_NAMES])

        # Appended immediately: a long batch that dies at model 8 keeps 0-7.
        with open(progress_path, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [idx, seed, row["final_loss"], row["best_loss"],
                 row["epochs"], f"{minutes:.2f}"]
                + [final.get(n, float("nan")) for n in PARAM_NAMES])

        print(f"model {idx}: final loss {row['final_loss']:.6f}  "
              f"(best {row['best_loss']:.6f})  {minutes:.1f} min")

    ok = [r for r in results if np.isfinite(r["final_loss"])]
    print("\n" + "=" * 72)
    print(f"{len(ok)}/{args.models} usable in {(time.time() - t_start) / 60:.1f} min")
    if ok:
        losses = [r["final_loss"] for r in ok]
        print(f"loss {min(losses):.4f} - {max(losses):.4f}")
        print(f"\n{'parameter':<32}{'median':>12}{'CV':>10}{'max/min':>12}")
        for n in PARAM_NAMES:
            v = np.array([r["final_params"][n] for r in ok])
            print(f"{n:<32}{np.median(v):>12.5g}{v.std() / v.mean():>10.3f}"
                  f"{v.max() / v.min():>12.5g}")

    with open(os.path.join(outdir, "results", "training_summary.json"), "w") as fh:
        json.dump({
            "site": args.site,
            "location": name,
            "grid": list(grid),
            "observations": len(ts),
            "comparisons": len(ts) - 1,
            "num_models": args.models,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "steps_per_week": args.steps_per_week,
            "multiplier": args.multiplier,
            "use_delta_loss": not args.absolute_loss,
            "device": str(device),
            "results": results,
        }, fh, indent=2)
    print(f"\nwrote {outdir}")


if __name__ == "__main__":
    main()
