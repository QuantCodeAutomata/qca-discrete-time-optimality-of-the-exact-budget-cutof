# AGENTS.md — Repository-Specific Knowledge

## Project Purpose
Numerical reproduction of discrete-time optimality of the exact-budget
cutoff/capping allocation rule for admissible loss allocation (no empirical data;
all computations are on synthetic laws over [0, ∞)).

## Core Architecture (src/)

### `laws.py` — `DiscreteLaw`
- Represents a measure with a point mass `q` at 0 and a grid-discretized
  positive part on an `x_grid` with weights `w`.
- Key methods: `call_function(a)`, `truncated_moment(a)`, `mean()`, `default_mass()`.
- Initial law constructors: `make_uniform_law()`, `make_mixture_uniform_law()`,
  `make_atomic_law()`.

### `semigroup.py` — `apply_semigroup(law, dt)`
- Propagates a `DiscreteLaw` under the absorbed Brownian semigroup for time `dt`.
- Uses the exact transition density
  `p_t^0(x,y) = (2πt)^{-1/2}[exp(-(x-y)²/2t) − exp(-(x+y)²/2t)]`.
- Default probability for mass at position x: `P_x(τ≤t) = 2Φ(−x/√t)`.
- Mass at zero is accumulated (absorbed) and never re-emitted.

### `cutoff.py` — `apply_cap(law, c_budget)`, `run_cutoff_flow(law0, alpha, T, delta)`
- `find_threshold(law, c)`: Brent root-finding for κ s.t. C_μ(κ) = c.
- `apply_cap(law, c)`: returns (capped_law, kappa) where the cap operator
  `T_c μ = (x↦x∧κ)# μ` is applied by redistributing mass above κ to κ.
- `run_cutoff_flow`: full discrete evolution; returns dict with kappas, q/h/m paths, law sequence.
- Cap identity: `C_{T_c μ}(a) = (C_μ(a) − c)⁺`, verified numerically after each step.

### `policies.py` — Benchmark admissible controls
- `proportional_haircut_law(law, c_budget)`: Y = (1−λ)X, λ = c/m.
- `flat_haircut_law(law, c_budget)`: uniform reduction Y = X − d.
- `bottom_stripping_law(law, c_budget)`: removes mass from the bottom of the
  distribution using a partial boundary-cell strip (Brent-corrected).
- All policies preserve q_μ (default mass) and satisfy m_ν = m_μ − c (exact budget).

### `simulate.py` — `simulate_pathwise(law0, kappas, delta, T, n_paths, seed)`
- Independent pathwise discrete reflection construction:
  `K_t = max_{j: jΔ≤t}(X_0 + B_{jΔ} − κ_j)⁺`, absorbed at 0.
- Uses shared random seeds for common-random-number comparison to law recursion.

### `metrics.py`
- `evaluate_payoffs(law, payoffs)`: dict of E[f(X)] for payoff functions f.
- `compute_d_curve(law, a_grid)`: D_μ(a) = m_μ − C_μ(a) on a strike grid.
- `check_exact_balance(h_path, q_path, alpha)`: residual of E[L] = α P(τ≤t).
- `check_resource_invariance(m_path, q_path, m0, alpha)`: residual of m + αq = m₀.

## Test Suite (84 tests, ~16 s)
- `tests/test_laws.py`: law construction, call function, mean, quadrature accuracy.
- `tests/test_semigroup.py`: semigroup zero-time, mass conservation, default accumulation.
- `tests/test_cutoff.py`: threshold finding, cap identities, flow invariants.
- `tests/test_metrics_policies.py`: D-curve, payoffs, policy budget balance.

## Experiment Scripts (exp/)
- `exp1_discrete_optimality.py`: one-step + dynamic cutoff vs. benchmarks.
- `exp2_continuous_limit.py`: mesh refinement Δ→0, payoff/boundary convergence.
- `exp3_stability.py`: Block A payoff invariance, Block B robustness, Block C perturbation.

## Key Numerical Tolerances
- Root-finding (Brent): xtol=1e-12, rtol=1e-10.
- Mass conservation after semigroup: absolute tolerance 1e-8.
- Budget equality after cap: absolute tolerance 1e-9.
- Resource invariance check: tolerance 1e-6.
- x-grid: 2000 points on [0, 3.0] unless overridden.

## Known Edge Cases
- `apply_cap` with c=0 returns (law, inf) — threshold is ∞ when no budget.
- For mu1 uniform[0.5, 1.5]: first threshold κ₁ = inf (rendered as NaN in plots)
  because the positive support is far from zero; consistent with Λ_t → a+b = 2.0.
- At t=0, initial laws with atoms (mu3) produce singular density; PDE residuals
  are only meaningful for t ≥ τ ≈ 0.05.
- `bottom_stripping_law` uses a partial boundary-cell correction (Brent's method)
  to achieve O(dx²) mean error rather than O(dx).

## Results Location
All plots (PNG), summary tables (CSV), and `RESULTS.md` are in `results/`.
