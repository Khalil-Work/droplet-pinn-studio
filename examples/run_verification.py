"""Produce the recorded comparison, animation, and raw data.

Equal-budget comparison of the hard-IC ansatz and the soft-IC network on
the normalized droplet d^2 problem (z' = -1, z(0) = 1, exact z = 1 - tau),
three deterministic seeds each, identical architecture / collocation /
optimizer / schedule / steps. Errors are absolute metrics on the stated
pre-exhaustion interval [0, 0.8].

Outputs under docs/results/: comparison figure (cover), training curves,
41-frame animation built from recorded optimizer snapshots (prediction vs
exact with the iteration count visible), CSVs, run_summary.json.
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import sys

# Windows (EngineeringPython) loads two OpenMP runtimes when NumPy/MKL and
# torch are imported together; allow it for this plotting-only process.
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from droplet_pinn_studio import exact_solution, train_pin

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

BLUE, RED, GREEN, GRAY = "#1f4e79", "#b23a48", "#0b6e4f", "#555555"
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)

SEEDS = (0, 1, 2)
STEPS, COLLOCATION, LR, TAU_MAX = 2000, 40, 5e-3, 0.8


def run_suite():
    results = {"hard": [], "soft": []}
    for formulation in ("hard", "soft"):
        for seed in SEEDS:
            r = train_pin(formulation=formulation, seed=seed, steps=STEPS,
                          collocation=COLLOCATION, lr=LR, tau_max=TAU_MAX,
                          snapshot_every=STEPS // 40 if formulation == "hard"
                          and seed == 0 else None)
            results[formulation].append(r)
    return results


def make_cover(results):
    hard = results["hard"][0]
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.2))
    ax[0].plot(hard["tau_eval"], hard["z_pred"], "-", color=BLUE, lw=1.8,
               label="hard-IC ansatz (seed 0)")
    ax[0].plot(hard["tau_eval"], exact_solution(hard["tau_eval"]), "--",
               color=GRAY, lw=1.6, label="exact 1 − τ")
    ax[0].set_xlabel("τ [-]")
    ax[0].set_ylabel("z = q/q₀ [-]")
    ax[0].set_title(f"Learned vs exact (seed 0)\nmax |error| {hard['max_abs_error']:.1e}")
    ax[0].legend(frameon=False, fontsize=8)

    for i, r in enumerate(results["hard"]):
        h = np.array(r["history"])
        ax[1].semilogy(h[:, 0], h[:, 1], color=BLUE, alpha=0.5 + 0.25 * i,
                       label=f"hard seed {r['seed']}")
    for i, r in enumerate(results["soft"]):
        h = np.array(r["history"])
        ax[1].semilogy(h[:, 0], h[:, 1], color=RED, alpha=0.5 + 0.25 * i,
                       ls="--", label=f"soft seed {r['seed']}")
    ax[1].set_xlabel("Adam step [-]")
    ax[1].set_ylabel("residual loss [-]")
    ax[1].set_title("Training curves (equal budgets)")
    ax[1].legend(frameon=False, fontsize=7, ncol=2)

    labels, means, stds = [], [], []
    for formulation in ("hard", "soft"):
        errs = [r["held_out_max_abs_error"] for r in results[formulation]]
        labels.append(formulation)
        means.append(np.mean(errs))
        stds.append(np.std(errs))
    ax[2].bar(labels, means, yerr=stds, color=[BLUE, RED], width=0.5,
              capsize=6)
    ax[2].set_ylabel("held-out max |error| [-]")
    ax[2].set_title("3 seeds: mean ± std")
    fig.suptitle("Droplet PINN Studio — learning z' = −1, z(0) = 1 "
                 "(normalized droplet d² law)", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison.png", dpi=150, bbox_inches="tight")
    fig.savefig(RESULTS / "comparison.svg", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_gif(results):
    snap = results["hard"][0]["snapshots"]
    tau = results["hard"][0]["tau_eval"]
    exact = exact_solution(tau)
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=140)
    (line,) = ax.plot([], [], color=BLUE, lw=2.0, label="network prediction")
    ax.plot(tau, exact, "--", color=GRAY, lw=1.6, label="exact 1 − τ")
    ax.set_xlim(0.0, TAU_MAX)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("τ [-]")
    ax.set_ylabel("z = q/q₀ [-]")
    title = ax.set_title("hard-IC ansatz — Adam step 0")
    ax.legend(frameon=False, fontsize=9, loc="lower left")

    writer = mpl_animation.PillowWriter(fps=8)
    with writer.saving(fig, RESULTS / "training_evolution.gif", dpi=140):
        for step, pred in snap:
            line.set_data(tau, pred)
            title.set_text(f"hard-IC ansatz — Adam step {step}")
            writer.grab_frame()
    plt.close(fig)
    return len(snap)


def write_csvs(results):
    rows = ["formulation,seed,final_loss,max_abs_error,l2_error,held_out_max_abs_error"]
    for formulation in ("hard", "soft"):
        for r in results[formulation]:
            rows.append(
                f"{formulation},{r['seed']},{r['final_loss']:.9e},"
                f"{r['max_abs_error']:.9e},{r['l2_error']:.9e},"
                f"{r['held_out_max_abs_error']:.9e}"
            )
    (RESULTS / "summary.csv").write_text("\n".join(rows), encoding="utf-8")
    hard = results["hard"][0]
    rows = ["tau,z_hard_seed0,z_exact"]
    for t, z, e in zip(hard["tau_eval"], hard["z_pred"], exact_solution(hard["tau_eval"])):
        rows.append(f"{t:.6f},{z:.9e},{e:.9e}")
    (RESULTS / "best_run_profile.csv").write_text("\n".join(rows), encoding="utf-8")


def main():
    results = run_suite()
    make_cover(results)
    frames = make_gif(results)
    write_csvs(results)

    summary = {
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "system": platform.platform(),
        },
        "budget": {
            "steps": STEPS, "collocation": COLLOCATION, "lr": LR,
            "tau_max": TAU_MAX, "seeds": list(SEEDS),
            "note": "identical architecture/optimizer/schedule for both formulations",
        },
        "hard_held_out_max_abs": [r["held_out_max_abs_error"] for r in results["hard"]],
        "soft_held_out_max_abs": [r["held_out_max_abs_error"] for r in results["soft"]],
        "animation": {"file": "training_evolution.gif", "frames": frames},
        "scope_note": "illustrative droplet submodel with an exact solution; "
                      "no atomization/spray-CFD/experimental claims",
    }
    (RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
