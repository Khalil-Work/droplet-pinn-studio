"""Droplet PINN Studio: learning the normalized droplet d^2 ODE with a PINN.

Normalized problem (constant-condition d^2 law): with z = q/q0 and
tau = K t / q0 the governing ODE and its exact solution are

    dz/dtau = -1,   z(0) = 1,   z_exact(tau) = 1 - tau,   0 <= tau <= 1.

Two formulations are compared under equal budgets:

- hard-IC ansatz:  z(tau) = 1 + tau * N(tau)   (satisfies z(0) = 1 for any
  network N — the initial condition is built in);
- soft-IC network: z = N(tau) with the IC added as a loss penalty term.

Both use the same architecture, collocation points, optimizer, learning
rate, and step budget; several deterministic seeds quantify variability.
All errors are evaluated on a held-out tau grid inside the stated
pre-exhaustion interval [0, tau_max] (tau_max < 1 avoids the exact zero at
exhaustion, so no relative-error division by zero).

This is an illustrative droplet submodel with an inexpensive exact
solution: no atomization, no spray CFD, no experimental agreement, and no
claim that PINNs are superior to the analytic solution.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "exact_solution",
    "make_network",
    "hard_ansatz",
    "train_pin",
    "TRAIN_DEFAULTS",
]


def exact_solution(tau):
    """Exact normalized solution z = 1 - tau (numpy or torch input)."""
    return 1.0 - tau


TRAIN_DEFAULTS = {
    "hidden": 20,
    "layers": 2,
    "activation": torch.nn.Tanh,
    "collocation": 40,
    "steps": 2000,
    "lr": 5.0e-3,
    "ic_weight": 1.0,
    "tau_max": 0.8,
}


def make_network(hidden=20, layers=2, activation=torch.nn.Tanh, seed=0):
    """Small MLP N(tau) -> z (1 input, 1 output)."""
    torch.manual_seed(seed)
    mods = [torch.nn.Linear(1, hidden), activation()]
    for _ in range(layers - 1):
        mods += [torch.nn.Linear(hidden, hidden), activation()]
    mods += [torch.nn.Linear(hidden, 1)]
    net = torch.nn.Sequential(*mods)
    for p in net.parameters():
        if p.dim() == 2:  # weight matrices only; biases keep their default
            torch.nn.init.xavier_normal_(p)
    return net


def hard_ansatz(net, tau):
    """z(tau) = 1 + tau * N(tau): satisfies z(0) = 1 identically."""
    return 1.0 + tau * net(tau)


def _residual(net, tau, formulation):
    tau = tau.requires_grad_(True)
    z = hard_ansatz(net, tau) if formulation == "hard" else net(tau)
    dz = torch.autograd.grad(z.sum(), tau, create_graph=True)[0]
    return dz + 1.0, z


def train_pin(formulation="hard", seed=0, steps=2000, collocation=40,
              lr=5.0e-3, ic_weight=1.0, tau_max=0.8, hidden=20, layers=2,
              snapshot_every=None, device="cpu"):
    """Train one PINN; returns a result dict.

    The residual enforces dz/dtau = -1 at collocation points drawn from
    U(0, tau_max) with a fixed torch seed (reproducible); soft-IC adds
    ic_weight * (z(0) - 1)^2. Snapshots (network predictions on a fixed
    evaluation grid) are recorded every ``snapshot_every`` steps for the
    animation, each tagged with its step count.
    """
    torch.manual_seed(seed)
    net = make_network(hidden=hidden, layers=layers, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=steps // 2,
                                            gamma=0.5)
    tau_coll = torch.rand(collocation, 1) * tau_max
    tau_eval = torch.linspace(0.0, tau_max, 81).reshape(-1, 1)
    history = []
    snapshots = []
    for step in range(1, steps + 1):
        opt.zero_grad()
        residual, z0 = _residual(net, tau_coll.clone(), formulation)
        loss = (residual**2).mean()
        if formulation == "soft":
            loss = loss + ic_weight * (net(torch.zeros(1, 1)) - 1.0) ** 2
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == 1:
            history.append((step, float(loss.detach())))
        if snapshot_every and (step % snapshot_every == 0 or step == steps):
            with torch.no_grad():
                pred = hard_ansatz(net, tau_eval) if formulation == "hard" \
                    else net(tau_eval)
            snapshots.append((step, pred.squeeze(-1).numpy().copy()))
    with torch.no_grad():
        pred = hard_ansatz(net, tau_eval) if formulation == "hard" else net(tau_eval)
    z_pred = pred.squeeze(-1).numpy()
    z_exact = exact_solution(tau_eval.squeeze(-1).numpy())
    error = z_pred - z_exact
    held_out = torch.linspace(0.0, tau_max, 41)
    with torch.no_grad():
        zh = (hard_ansatz(net, held_out.reshape(-1, 1)) if formulation == "hard"
              else net(held_out.reshape(-1, 1))).squeeze(-1).numpy()
    ze = exact_solution(held_out.numpy())
    return {
        "formulation": formulation,
        "seed": seed,
        "history": history,
        "snapshots": snapshots,
        "tau_eval": tau_eval.squeeze(-1).numpy(),
        "z_pred": z_pred,
        "max_abs_error": float(np.max(np.abs(error))),
        "l2_error": float(np.sqrt(np.mean(error**2))),
        "held_out_max_abs_error": float(np.max(np.abs(zh - ze))),
        "final_loss": history[-1][1] if history else float("nan"),
    }
