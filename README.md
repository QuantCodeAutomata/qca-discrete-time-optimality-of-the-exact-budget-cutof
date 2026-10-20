# Discrete-Time Optimality of the Exact-Budget Cutoff Allocation Rule

Numerical reproduction of the theorem-level claims in the paper on discrete-time
optimal admissible loss allocation via the exact-budget cutoff/capping rule.

## Overview

This repository implements and validates three experiments:

| Experiment | Topic |
|-----------|-------|
| **Exp 1** | Discrete-time one-step and dynamic optimality of T_c μ |
| **Exp 2** | Continuous-time limit and free-boundary characterization |
| **Exp 3** | Uniqueness, stability, and robustness across initial laws |

**No empirical data is required.** All experiments use synthetic initial laws
on [0,∞) with compact support and no atom at zero.

## Repository Structure

```
.
├── data/                       # Reserved for data artefacts (synthetic only)
├── src/
│   ├── laws.py                 # DiscreteLaw, call/truncated-moment functions
│   ├── semigroup.py            # Absorbed Brownian semigroup S_t
│   ├── cutoff.py               # Budget inversion and cap operator T_c
│   ├── policies.py             # Benchmark admissible controls
│   ├── simulate.py             # Pathwise discrete reflection construction
│   └── metrics.py              # Payoffs, D-curves, invariant checks
├── exp/
│   ├── exp1_discrete_optimality.py
│   ├── exp2_continuous_limit.py
│   └── exp3_stability.py
├── tests/
│   ├── test_laws.py
│   ├── test_semigroup.py
│   ├── test_cutoff.py
│   └── test_metrics_policies.py
├── results/
│   └── RESULTS.md              # Full experimental findings
├── README.md
└── .gitignore
```

## Key Mathematical Objects

| Symbol | Meaning |
|--------|---------|
| S_t μ | Absorbed Brownian semigroup: propagates law μ for time t |
| C_μ(a) | Call function ∫(x−a)⁺ μ(dx) |
| D_μ(a) | Truncated-moment ∫(x∧a) μ(dx) = m_μ − C_μ(a) |
| q_μ | Default mass μ({0}) |
| κ_c(μ) | Threshold: C_μ(κ) = c |
| T_c μ | Capped law (x↦x∧κ_c(μ))# μ |
| G_Δ(μ) | One-step optimal operator: T_{c_Δ(μ)} S_Δ μ |

The absorbed Brownian semigroup uses the exact transition density:

```
p_t^0(x,y) = (2πt)^{-1/2} [exp(-(x-y)²/2t) - exp(-(x+y)²/2t)]
```

with default probability P_x(τ₀ ≤ t) = 2Φ(−x/√t).

## Running the Experiments

```bash
# Run all three experiments
python -m exp.exp1_discrete_optimality
python -m exp.exp2_continuous_limit
python -m exp.exp3_stability

# Run the test suite (84 tests)
python -m pytest tests/ -q
```

## Results Summary

- Cutoff dominates all admissible benchmarks (D-curve dominance) in 8/9 cases strictly,
  1/9 as a numerical tie (< 10⁻⁹ difference).
- Dynamic cutoff wins for **every** tested payoff (survival, linear, sqrt, log1p, x∧1)
  across all 9 (case, α) settings.
- Exact balance E[L_{t_n}] = α P(τ≤t_n): residual = **0.00** (machine precision).
- Resource invariance m_{t_n} + α q_{t_n} = m₀: residual ≤ 7.44 × 10⁻¹⁰.
- Payoff values V_f^Δ converge as Δ→0 (std < 10⁻³ across 5 mesh refinements).
- Perturbation stability is approximately **Lipschitz** in d_T# metric.

See [`results/RESULTS.md`](results/RESULTS.md) for complete tables and analysis.

## Dependencies

```
numpy >= 1.26
scipy >= 1.14
pandas >= 2.2
matplotlib
seaborn
pytest
```

No external market data or API access required.
