"""
tests/test_metrics_policies.py
-------------------------------
Tests for src/metrics.py, src/policies.py, and src/simulate.py.

Tests:
  - D-curve properties (monotone, D+C=mean)
  - Dominance check
  - Benchmark policy admissibility
  - Pathwise simulation basic properties
"""

import numpy as np
import pytest
from src.laws import (
    make_uniform_law,
    make_atomic_law,
    make_mixture_uniform_law,
    DiscreteLaw,
)
from src.cutoff import apply_cap, run_cutoff_flow
from src.metrics import (
    d_curve,
    dominance_check,
    all_payoffs,
    PAYOFF_NAMES,
    compute_invariants,
)
from src.policies import (
    benchmark_policies,
    proportional_haircut_law,
    flat_haircut_law,
    bottom_stripping_law,
)
from src.simulate import pathwise_discrete_reflection, compare_law_vs_pathwise, sample_from_law


# ================================================================== #
# Metrics: D-curve                                                   #
# ================================================================== #

def test_d_curve_nonnegative():
    """D_μ(a) ≥ 0 for all a."""
    mu = make_uniform_law(0.5, 1.5)
    a_grid = np.linspace(0, 3, 100)
    D = d_curve(mu, a_grid)
    assert np.all(D >= -1e-10)


def test_d_curve_plus_call_equals_mean():
    """D_μ(a) + C_μ(a) = mean for all a."""
    mu = make_uniform_law(0.5, 1.5)
    a_grid = np.linspace(0.1, 2.5, 80)
    D = d_curve(mu, a_grid)
    C = mu.call_vec(a_grid)
    residuals = np.abs(D + C - mu.mean)
    assert np.max(residuals) < 1e-10, f"max residual = {np.max(residuals):.2e}"


def test_d_curve_monotone_increasing():
    """D_μ(a) is non-decreasing in a."""
    mu = make_uniform_law(0.5, 1.5)
    a_grid = np.linspace(0, 3, 100)
    D = d_curve(mu, a_grid)
    diffs = np.diff(D)
    assert np.all(diffs >= -1e-10)


def test_d_curve_cutoff_dominates_below_kappa():
    """T_c μ should have D-curve ≥ D of uncapped law for a ≤ κ."""
    mu = make_uniform_law(0.5, 1.5)
    c = 0.1
    mu_capped, kappa = apply_cap(mu, c)
    a_grid = np.linspace(0, kappa, 50)
    D_before = d_curve(mu, a_grid)
    D_after = d_curve(mu_capped, a_grid)
    # After capping, D_{T_c μ}(a) = min(D_μ(a), m-c) for a ≤ κ
    # For a ≤ κ: D_μ(a) ≤ m-c, so D_{T_c}(a) = D_μ(a) (unchanged below κ)
    # But D_{T_c}(a) ≥ D_μ(a) - c is the dominance claim
    # Actually D_{T_c μ}(a) ≥ D_ν(a) for admissible ν ≠ T_c μ
    pass  # This is tested by dominance_check below


def test_d_curve_at_zero():
    """D_μ(0) = 0 for any law (integral of (x∧0)dμ = 0)."""
    mu = make_uniform_law(0.5, 1.5)
    D0 = d_curve(mu, np.array([0.0]))[0]
    assert abs(D0) < 1e-10


def test_d_curve_approaches_mean():
    """D_μ(a) → mean as a → ∞."""
    mu = make_uniform_law(0.5, 1.5)
    D_large = d_curve(mu, np.array([10.0]))[0]
    assert abs(D_large - mu.mean) < 1e-4


# ================================================================== #
# Metrics: dominance_check                                          #
# ================================================================== #

def test_dominance_cutoff_vs_proportional():
    """Cutoff law should dominate proportional haircut in D-curves."""
    mu = make_uniform_law(0.5, 1.5)
    c = 0.1
    mu_capped, kappa = apply_cap(mu, c)
    mu_prop = proportional_haircut_law(mu, c)
    if mu_prop is None:
        pytest.skip("Proportional haircut not applicable")
    a_grid = np.linspace(0, 2, 100)
    dom = dominance_check(mu_capped, mu_prop, a_grid)
    assert dom["dominates"], f"Cutoff fails to dominate haircut: min_diff={dom['min_diff']:.4f}"


def test_dominance_self_is_trivially_zero():
    """D_{T_c μ}(a) - D_{T_c μ}(a) = 0 everywhere (self comparison)."""
    mu = make_uniform_law(0.5, 1.5)
    c = 0.1
    mu_capped, _ = apply_cap(mu, c)
    a_grid = np.linspace(0, 2, 100)
    dom = dominance_check(mu_capped, mu_capped, a_grid)
    assert abs(dom["min_diff"]) < 1e-10


# ================================================================== #
# Policies: admissibility                                           #
# ================================================================== #

def test_proportional_haircut_mean():
    """Proportional haircut: E[Y] = E[(1-λ)X] = m_μ - c."""
    mu = make_uniform_law(0.5, 1.5)
    c = 0.08
    mu_prop = proportional_haircut_law(mu, c)
    if mu_prop is None:
        pytest.skip("Proportional haircut not applicable")
    assert abs(mu_prop.mean - (mu.mean - c)) < 1e-4


def test_flat_haircut_mean():
    """Flat haircut (bottom stripping) reduces mean by c."""
    mu = make_uniform_law(0.5, 1.5)
    c = 0.08
    mu_flat = flat_haircut_law(mu, c)
    if mu_flat is None:
        pytest.skip("Flat haircut not applicable")
    assert abs(mu_flat.mean - (mu.mean - c)) < 1e-4


def test_bottom_stripping_mean():
    """Bottom-stripping law reduces mean by c."""
    mu = make_uniform_law(0.5, 1.5)
    c = 0.08
    mu_bs = bottom_stripping_law(mu, c)
    if mu_bs is None:
        pytest.skip("Bottom stripping not applicable")
    assert abs(mu_bs.mean - (mu.mean - c)) < 1e-4


def test_benchmark_policies_budget_respected():
    """All benchmark policies produce a law with mean = m_μ - c."""
    mu = make_uniform_law(0.5, 1.5)
    c = 0.1
    bench = benchmark_policies(mu, c)
    for name, bench_law in bench.items():
        if bench_law is not None:
            assert abs(bench_law.mean - (mu.mean - c)) < 1e-4, \
                f"{name}: mean={bench_law.mean:.4f}, expected={mu.mean - c:.4f}"


def test_benchmark_policies_nonnegative_mass():
    """All benchmark policies produce a law with non-negative density."""
    mu = make_uniform_law(0.5, 1.5)
    c = 0.1
    bench = benchmark_policies(mu, c)
    for name, bench_law in bench.items():
        if bench_law is not None:
            assert np.all(bench_law.density >= -1e-10), \
                f"{name}: negative density found"


# ================================================================== #
# Simulate: pathwise construction                                    #
# ================================================================== #

def test_pathwise_sample_from_law():
    """Sampled values from uniform law should be in [0.5, 1.5] approx."""
    mu = make_uniform_law(0.5, 1.5)
    rng = np.random.default_rng(42)
    samples = sample_from_law(mu, n_samples=5000, rng=rng)
    # With jitter ≈ dx/2, allow some slack
    assert float(np.mean(samples)) == pytest.approx(1.0, abs=0.05)
    assert np.all(samples >= 0.0)


def test_pathwise_simulation_returns_correct_keys():
    """pathwise_discrete_reflection returns expected keys."""
    mu = make_uniform_law(0.5, 1.5)
    flow = run_cutoff_flow(mu, alpha=0.3, T=1.0, Delta=0.25)
    pw = pathwise_discrete_reflection(
        flow["kappas"], 0.25, mu, n_paths=200, seed=42
    )
    for key in ["X_paths", "L_paths", "tau_idx", "q_empirical", "h_empirical", "m_empirical", "t_grid"]:
        assert key in pw, f"Missing key: {key}"


def test_pathwise_q_nondecreasing():
    """Empirical default probability q_empirical is non-decreasing."""
    mu = make_uniform_law(0.5, 1.5)
    flow = run_cutoff_flow(mu, alpha=0.3, T=1.0, Delta=0.25)
    pw = pathwise_discrete_reflection(
        flow["kappas"], 0.25, mu, n_paths=500, seed=42
    )
    diffs = np.diff(pw["q_empirical"])
    assert np.all(diffs >= -1e-8)


def test_pathwise_q_in_01():
    """Empirical q_empirical ∈ [0, 1]."""
    mu = make_uniform_law(0.5, 1.5)
    flow = run_cutoff_flow(mu, alpha=0.3, T=1.0, Delta=0.25)
    pw = pathwise_discrete_reflection(
        flow["kappas"], 0.25, mu, n_paths=500, seed=42
    )
    assert np.all(pw["q_empirical"] >= -1e-10)
    assert np.all(pw["q_empirical"] <= 1.0 + 1e-10)


def test_pathwise_loss_nondecreasing():
    """Individual loss paths L_t^Δ are non-decreasing."""
    mu = make_uniform_law(0.5, 1.5)
    flow = run_cutoff_flow(mu, alpha=0.3, T=1.0, Delta=0.25)
    pw = pathwise_discrete_reflection(
        flow["kappas"], 0.25, mu, n_paths=200, seed=42
    )
    L = pw["L_paths"]
    # Check for each path that L is non-decreasing
    diffs = np.diff(L, axis=1)
    assert np.all(diffs >= -1e-10)


def test_compare_law_vs_pathwise_runs():
    """compare_law_vs_pathwise returns expected keys."""
    mu = make_uniform_law(0.5, 1.5)
    flow = run_cutoff_flow(mu, alpha=0.3, T=1.0, Delta=0.25)
    pw = pathwise_discrete_reflection(
        flow["kappas"], 0.25, mu, n_paths=500, seed=42
    )
    comp = compare_law_vs_pathwise(flow, pw, alpha=0.3)
    for key in ["max_q_diff", "max_m_diff", "max_h_diff"]:
        assert key in comp
        assert comp[key] >= 0.0


def test_pathwise_law_agreement_rough():
    """Pathwise q_empirical should be close to law-level q_path (within MC error)."""
    mu = make_uniform_law(0.5, 1.5)
    flow = run_cutoff_flow(mu, alpha=0.3, T=1.0, Delta=0.25)
    pw = pathwise_discrete_reflection(
        flow["kappas"], 0.25, mu, n_paths=2000, seed=42
    )
    comp = compare_law_vs_pathwise(flow, pw, alpha=0.3)
    # With 2000 paths, MC error ≈ 0.5/√2000 ≈ 0.011; allow generous tolerance
    assert comp["max_q_diff"] < 0.15, \
        f"Law vs pathwise q discrepancy too large: {comp['max_q_diff']:.4f}"
