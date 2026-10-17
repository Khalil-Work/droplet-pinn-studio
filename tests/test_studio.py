"""Verification tests for droplet_pinn_studio.

Small, fast budgets (CPU): each test trains a few hundred steps at most.

Properties pinned:
- the hard ansatz satisfies the initial condition identically (z(0)=1 for
  an untrained random network);
- the normalized problem is correctly posed (exact solution residual zero);
- training reduces the residual loss on both formulations;
- the soft formulation actually needs its IC term (untrained soft nets do
  not satisfy z(0)=1; after training with the penalty they approach it);
- deterministic reproducibility: same seed + same budget -> identical loss
  history;
- errors are reported on the stated pre-exhaustion interval with absolute
  metrics (no division by the exact zero at tau=1).
"""

import numpy as np
import pytest
import torch

from droplet_pinn_studio import exact_solution, hard_ansatz, make_network, train_pin


def test_hard_ansatz_satisfies_ic_identically():
    for seed in range(3):
        net = make_network(seed=seed)
        z0 = hard_ansatz(net, torch.zeros(4, 1))
        torch.testing.assert_close(z0, torch.ones(4, 1))


def test_exact_solution_residual_is_zero():
    tau = torch.linspace(0.0, 0.8, 33, requires_grad=True).reshape(-1, 1)
    z = exact_solution(tau)
    dz = torch.autograd.grad(z.sum(), tau)[0]
    torch.testing.assert_close(dz, -torch.ones_like(dz))
    # z(0) = 1 and z stays positive on [0, 0.8] (pre-exhaustion interval).
    assert exact_solution(torch.tensor(0.0)).item() == 1.0
    assert exact_solution(torch.tensor(0.8)).item() == pytest.approx(0.2)


def test_training_reduces_loss_both_formulations():
    for formulation in ("hard", "soft"):
        result = train_pin(formulation=formulation, seed=1, steps=300,
                           collocation=20, lr=1e-2)
        first = result["history"][0][1]
        last = result["final_loss"]
        assert last < first, f"{formulation}: loss {first} -> {last} did not fall"
        assert np.isfinite(result["max_abs_error"])


def test_soft_formulation_learns_the_initial_condition():
    # Untrained soft nets do not satisfy z(0) = 1 ...
    torch.manual_seed(0)
    raw = make_network(seed=0)
    assert abs(raw(torch.zeros(1, 1)).item() - 1.0) > 1e-6
    # ... but with the IC penalty the trained soft network approaches it.
    result = train_pin(formulation="soft", seed=0, steps=400, collocation=20,
                       lr=1e-2)
    # z(0) from the recorded prediction grid (tau_eval starts at 0).
    assert abs(result["z_pred"][0] - 1.0) < 5e-2


def test_deterministic_reproducibility():
    a = train_pin(formulation="hard", seed=7, steps=150, collocation=16,
                  lr=1e-2)
    b = train_pin(formulation="hard", seed=7, steps=150, collocation=16,
                  lr=1e-2)
    assert a["history"] == b["history"]
    np.testing.assert_array_equal(a["z_pred"], b["z_pred"])


def test_errors_use_absolute_metrics_on_pre_exhaustion_interval():
    result = train_pin(formulation="hard", seed=3, steps=200, collocation=16,
                       lr=1e-2)
    # Interval excludes tau = 1: no division by an exact zero anywhere.
    assert result["tau_eval"][-1] == pytest.approx(0.8)
    for key in ("max_abs_error", "l2_error", "held_out_max_abs_error"):
        assert np.isfinite(result[key]) and result[key] >= 0.0
