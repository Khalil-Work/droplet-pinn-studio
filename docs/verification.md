# Verification evidence

Reproduced by `python -m pytest tests -q` (6 tests; run with a
torch-equipped environment) and `python examples/run_verification.py`
(figures + GIF + CSV/JSON). Verified on Windows 11, Python 3.13.5,
torch 2.14.1+cpu, NumPy 2.1.3.

## What each test establishes

| Test | Property established |
| --- | --- |
| `test_hard_ansatz_satisfies_ic_identically` | The ansatz meets z(0)=1 for arbitrary untrained networks (structural IC). |
| `test_exact_solution_residual_is_zero` | The posed problem and its exact solution are consistent (autograd residual −1; z(0)=1; positivity on the interval). |
| `test_training_reduces_loss_both_formulations` | Both formulations actually learn under the stated budget. |
| `test_soft_formulation_learns_the_initial_condition` | Untrained soft nets violate z(0)=1; with the penalty term they approach it — the two formulations genuinely differ in IC handling. |
| `test_deterministic_reproducibility` | Same seed + budget → bitwise-identical loss history and predictions. |
| `test_errors_use_absolute_metrics_on_pre_exhaustion_interval` | Errors are finite, non-negative, absolute metrics with the grid capped at τ=0.8 (no division by the exact zero at τ=1). |

## Recorded run (`docs/results/`)

Equal-budget comparison (2000 Adam steps, 40 collocation points, identical
architecture/optimizer/schedule), three seeds each, held-out max |error|:

| Formulation | seed 0 | seed 1 | seed 2 | mean | std |
| --- | --- | --- | --- | --- | --- |
| hard-IC ansatz | 7.7e-5 | 8.1e-5 | 2.3e-4 | ~1.3e-4 | ~7e-5 |
| soft-IC network | 7.5e-4 | 5.9e-5 | 1.3e-3 | ~7.2e-4 | ~5e-4 |

Observed on this budget: the hard-IC ansatz has both a lower mean error
and noticeably lower seed-to-seed variability. This is a measured
observation on one small problem, not a general claim about PINN
formulations. Files: `comparison.png/.svg` (cover), `summary.csv`,
`best_run_profile.csv`, `training_evolution.gif` (40 frames of recorded
optimizer snapshots with the exact curve and iteration count visible),
`run_summary.json`.

## Limits of this evidence

- One deliberately trivial ODE; conclusions do not transfer to
  high-dimensional or stiff PDEs.
- Equal step budgets ≠ equal FLOPs (recorded caveat); CPU-only.
- No comparison against classical numerical integration (the exact
  solution is free — PINNs here are a study vehicle, not a tool of choice).
- No atomization, spray, or experimental scope anywhere.
