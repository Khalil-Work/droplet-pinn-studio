# API reference

Module `droplet_pinn_studio.studio` (re-exported at package level).

- `exact_solution(tau)` — z = 1 − τ (NumPy or torch input).
- `make_network(hidden=20, layers=2, activation=torch.nn.Tanh, seed=0)` —
  the MLP N(τ); Xavier-initialized weights, default biases.
- `hard_ansatz(net, tau)` — z = 1 + τ·N(τ); z(0) = 1 identically.
- `train_pin(formulation="hard"|"soft", seed=0, steps=2000, collocation=40,
  lr=5e-3, ic_weight=1.0, tau_max=0.8, hidden=20, layers=2,
  snapshot_every=None) -> dict` — trains one PINN with a fixed torch seed
  (deterministic given the arguments) and returns:

  - `history`: [(step, loss)] every 50 steps (plus step 1);
  - `snapshots`: [(step, z_prediction_on_eval_grid)] when
    `snapshot_every` is set (the animation source);
  - `tau_eval`, `z_pred`: the 81-point evaluation grid/prediction;
  - `max_abs_error`, `l2_error`: absolute errors vs the exact solution on
    the evaluation grid (τ ∈ [0, tau_max]);
  - `held_out_max_abs_error`: max |error| on a distinct 41-point grid;
  - `final_loss`.

Raises nothing beyond torch's own errors; inputs are validated by range
conventions (tau_max < 1 keeps errors division-free).

## Example

```python
from droplet_pinn_studio import train_pin

result = train_pin(formulation="hard", seed=0, steps=2000)
print(result["held_out_max_abs_error"], result["final_loss"])
```

Windows note: importing NumPy/MKL and torch in one process can trigger a
duplicate-OpenMP error; `examples/run_verification.py` sets
`KMP_DUPLICATE_LIB_OK` before importing torch (plotting-only process).
