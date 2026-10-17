# Model and method

## Problem

Learning the constant-condition droplet d² law with a physics-informed
neural network (PINN), in normalized variables. With q = D², z = q/q₀ and
τ = K t/q₀, the law becomes the simplest possible ODE system:

    dz/dτ = −1,   z(0) = 1,   exact solution  z(τ) = 1 − τ.

The normalization removes all physical parameters from the learning
problem (they re-enter through the scales q₀ and K); the exact solution is
inexpensive, which is precisely what makes honest error measurement
possible. **This is an illustrative droplet submodel**: no atomization, no
breakup, no spray-CFD validation, no experimental agreement, and no claim
that PINNs beat the analytic solution.

## Formulations compared (equal budgets)

- **Hard-IC ansatz:** z(τ) = 1 + τ·N(τ) — satisfies z(0) = 1 *identically*
  for any network N; the initial condition is structural, not penalized.
- **Soft-IC network:** z = N(τ) with the IC added as a penalty term
  w·(N(0) − 1)² in the loss.

Both share: architecture (1→20→20→1 tanh MLP), Xavier initialization,
collocation points (40 drawn U(0, τ_max) with a fixed torch seed), Adam
(2000 steps, lr 5e-3, StepLR halving at mid-budget), CPU. Equal step
budgets are not exactly equal FLOPs — recorded as a caveat. Three
deterministic seeds per formulation quantify variability.

## Loss

    L = mean((dz/dτ + 1)²)  [+ w·(N(0) − 1)² for soft-IC],

with dz/dτ from reverse-mode autodifferentiation of the network.

## Error measurement

All errors are **absolute** metrics (max |error|, L2, held-out max) on the
stated pre-exhaustion interval τ ∈ [0, 0.8]. The interval excludes τ = 1,
where the exact solution reaches zero and relative error would divide by
it. Held-out errors use a 41-point grid distinct from the training
collocation draws.

## Scope and limits

The problem is deliberately trivial: the value of the study is the clean
formulation comparison (structural vs penalized IC) under controlled,
seed-reproducible conditions — a template for posing droplet submodels
learnably, not a claim about PINN performance on real spray physics.
