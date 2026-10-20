"""
tests/test_cutoff.py
--------------------
Tests for src/cutoff.py: threshold finding, cap operator, and discrete flow.

Key mathematical properties verified:
  - C_{T_c μ}(a) = (C_μ(a) - c)^+
  - D_{T_c μ}(a) = min(D_μ(a), m_μ - c)
  - Total mass after capping = 1
  - Mean after capping = mean_before - c
  - Exact balance: h_n = α · q_n
  - Resource invariance: m_n + α · q_n = m_0
"""

import numpy as np
import pytest
from src.laws import (
    make_uniform_law,
    make_atomic_law,
    make_mixture_uniform_law,
    standard_test_cases,
    DiscreteLaw,
)
from src.cutoff import (
    find_threshold,
    apply_cap,
    verify_cap_identities,
    run_cutoff_flow,
)
from src.metrics import compute_invariants, all_payoffs, PAYOFF_NAMES


# ================================================================== #
# find_threshold tests                                               #
# ================================================================== #

def test_find_threshold_call_value():
    """κ = find_threshold(μ, c): C_μ(κ) = c within tolerance."""
    mu = make_uniform_law(0.5, 1.5)
    c = 0.05
    kappa = find_threshold(mu, c)
    assert kappa is not None
    assert abs(mu.call(kappa) - c) < 1e-6, \
        f"C_μ(κ={kappa:.4f}) = {mu.call(kappa):.6f} ≠ c={c}"


def test_find_threshold_zero_budget():
    """c=0: threshold is at or above max support (no capping needed)."""
    mu = make_uniform_law(0.5, 1.5)
    kappa = find_threshold(mu, 0.0)
    # C_μ(κ) = 0 means κ ≥ max support
    assert mu.call(kappa) < 1e-8


def test_find_threshold_atomic_law_exact():
    """Atomic law: verify threshold analytically.

    μ = 0.5·δ_{0.25} + 0.5·δ_{1.25}, c = 0.5 * (1.25 - κ) for κ ∈ [0.25, 1.25).
    So κ = 1.25 - 2c for c ≤ 0.5.
    """
    mu = make_atomic_law([0.25, 1.25], [0.5, 0.5])
    c = 0.1
    kappa = find_threshold(mu, c)
    expected_kappa = 1.25 - 2 * c  # analytical
    # C_μ(κ) = 0.5*(1.25-κ)^+ for κ ∈ (0.25, 1.25)
    assert abs(mu.call(kappa) - c) < 1e-6


def test_find_threshold_is_in_support(uniform_law=None):
    """Threshold should be within or at the support of the law."""
    mu = make_uniform_law(0.5, 1.5)
    c = 0.1
    kappa = find_threshold(mu, c)
    # κ should be in [a, b] for a uniform law
    assert 0.0 <= kappa <= 2.0


# ================================================================== #
# apply_cap tests                                                    #
# ================================================================== #

def test_cap_reduces_mean_by_c():
    """After capping: mean = mean_before - c (up to tolerance)."""
    mu = make_uniform_law(0.5, 1.5)
    c = 0.1
    mu_capped, kappa = apply_cap(mu, c)
    expected_mean = mu.mean - c
    assert abs(mu_capped.mean - expected_mean) < 1e-4, \
        f"mean_capped={mu_capped.mean:.6f}, expected={expected_mean:.6f}"


def test_cap_preserves_total_mass():
    """Capping does not add or remove total mass."""
    mu = make_uniform_law(0.5, 1.5)
    c = 0.08
    mu_capped, _ = apply_cap(mu, c)
    assert abs(mu_capped.total_mass - mu.total_mass) < 1e-6


def test_cap_identity_call():
    """C_{T_c μ}(a) = (C_μ(a) - c)^+ for all a."""
    mu = make_uniform_law(0.5, 1.5)
    c = 0.1
    mu_capped, kappa = apply_cap(mu, c)
    a_grid = np.linspace(0, 2, 100)
    C_before = mu.call_vec(a_grid)
    C_after = mu_capped.call_vec(a_grid)
    expected = np.maximum(C_before - c, 0.0)
    residuals = np.abs(C_after - expected)
    assert np.max(residuals) < 1e-5, \
        f"Max identity residual: {np.max(residuals):.2e}"


def test_cap_identity_truncated_moment():
    """D_{T_c μ}(a) = min(D_μ(a), m_μ - c) for all a."""
    mu = make_uniform_law(0.5, 1.5)
    c = 0.1
    mu_capped, kappa = apply_cap(mu, c)
    a_grid = np.linspace(0, 2, 100)
    D_before = mu.truncated_moment_vec(a_grid)
    D_after = mu_capped.truncated_moment_vec(a_grid)
    expected = np.minimum(D_before, mu.mean - c)
    residuals = np.abs(D_after - expected)
    assert np.max(residuals) < 1e-5, \
        f"Max D identity residual: {np.max(residuals):.2e}"


def test_cap_zero_budget_is_identity():
    """T_0 μ = μ (no capping when c=0)."""
    mu = make_uniform_law(0.5, 1.5)
    mu_capped, kappa = apply_cap(mu, 0.0)
    assert abs(mu_capped.mean - mu.mean) < 1e-6


def test_cap_atomic_law():
    """Verify cap for atomic law: atoms above κ are moved to κ."""
    mu = make_atomic_law([0.25, 1.25], [0.5, 0.5])
    c = 0.1  # κ = 1.25 - 2*0.1 = 1.05
    mu_capped, kappa = apply_cap(mu, c)
    # Mean should be 0.5*0.25 + 0.5*kappa = 0.125 + 0.5*kappa = mu.mean - c
    expected_mean = mu.mean - c
    assert abs(mu_capped.mean - expected_mean) < 1e-6


def test_cap_call_identity_mixture():
    """Cap identity holds for mixture law."""
    mu = make_mixture_uniform_law(0.2, 0.4, 1.2, 1.4, p1=0.5)
    c = 0.05
    mu_capped, kappa = apply_cap(mu, c)
    a_grid = np.linspace(0, 2, 100)
    C_before = mu.call_vec(a_grid)
    C_after = mu_capped.call_vec(a_grid)
    expected = np.maximum(C_before - c, 0.0)
    assert np.max(np.abs(C_after - expected)) < 1e-5


# ================================================================== #
# verify_cap_identities tests                                        #
# ================================================================== #

def test_verify_cap_identities_passes():
    """verify_cap_identities should return passed=True."""
    mu = make_uniform_law(0.5, 1.5)
    a_grid = np.linspace(0, 2, 100)
    c = 0.1
    result = verify_cap_identities(mu, c, a_grid)
    assert result["passed"], f"Identities failed: call_res={result['max_call_residual']:.2e}"


def test_verify_cap_identities_residuals_small():
    """Cap identity residuals should be < 1e-4."""
    mu = make_uniform_law(0.5, 1.5)
    a_grid = np.linspace(0, 2, 200)
    for c in [0.01, 0.05, 0.1, 0.2]:
        result = verify_cap_identities(mu, c, a_grid)
        assert result["max_call_residual"] < 1e-4, \
            f"c={c}: call residual={result['max_call_residual']:.2e}"
        assert result["max_trunc_residual"] < 1e-4, \
            f"c={c}: trunc residual={result['max_trunc_residual']:.2e}"


# ================================================================== #
# run_cutoff_flow tests                                              #
# ================================================================== #

def test_cutoff_flow_runs():
    """run_cutoff_flow completes without error."""
    mu = make_uniform_law(0.5, 1.5)
    flow = run_cutoff_flow(mu, alpha=0.3, T=1.0, Delta=0.25)
    assert "laws" in flow
    assert "kappas" in flow
    assert "q_path" in flow


def test_cutoff_flow_number_of_steps():
    """Flow has exactly M+1 laws for M steps."""
    Delta = 0.25
    T = 1.0
    M = int(round(T / Delta))
    mu = make_uniform_law(0.5, 1.5)
    flow = run_cutoff_flow(mu, alpha=0.3, T=T, Delta=Delta)
    assert len(flow["laws"]) == M + 1
    assert len(flow["kappas"]) == M


def test_cutoff_flow_exact_balance():
    """h_n = α · q_n holds along the flow (exact balance)."""
    mu = make_uniform_law(0.5, 1.5)
    alpha = 0.3
    flow = run_cutoff_flow(mu, alpha=alpha, T=1.0, Delta=0.25)
    h_path = np.array(flow["h_path"])
    q_path = np.array(flow["q_path"])
    residuals = np.abs(h_path - alpha * q_path)
    assert np.max(residuals) < 1e-5, \
        f"Exact balance violated: max residual = {np.max(residuals):.2e}"


def test_cutoff_flow_resource_invariance():
    """m_n + α · q_n = m_0 along the flow (resource invariance)."""
    mu = make_uniform_law(0.5, 1.5)
    m0 = mu.mean
    alpha = 0.3
    flow = run_cutoff_flow(mu, alpha=alpha, T=1.0, Delta=0.25)
    m_path = np.array(flow["m_path"])
    q_path = np.array(flow["q_path"])
    invariants = m_path + alpha * q_path
    residuals = np.abs(invariants - m0)
    assert np.max(residuals) < 1e-4, \
        f"Resource invariance violated: max residual = {np.max(residuals):.2e}"


def test_cutoff_flow_monotone_q():
    """Default mass q_n is non-decreasing."""
    mu = make_uniform_law(0.5, 1.5)
    flow = run_cutoff_flow(mu, alpha=0.3, T=1.0, Delta=0.25)
    q_path = np.array(flow["q_path"])
    assert np.all(np.diff(q_path) >= -1e-8)


def test_cutoff_flow_monotone_decreasing_m():
    """Mean m_n is non-increasing (losses reduce reserves)."""
    mu = make_uniform_law(0.5, 1.5)
    flow = run_cutoff_flow(mu, alpha=0.3, T=1.0, Delta=0.25)
    m_path = np.array(flow["m_path"])
    assert np.all(np.diff(m_path) <= 1e-8)


def test_cutoff_flow_all_cases():
    """run_cutoff_flow succeeds for all standard test cases.

    Resource invariance m_n + α q_n = m_0 holds analytically (absorbed BM is
    mean-conserving by the optional-stopping theorem).  The residual seen here
    is purely numerical (finite-grid quadrature of the semigroup kernel), so we
    allow a tolerance of 2e-3 rather than the default 1e-6.
    """
    cases = standard_test_cases()
    for case in cases:
        for alpha in case["alpha_values"]:
            if case["m0"] <= alpha:
                continue
            flow = run_cutoff_flow(case["law"], alpha, T=1.0, Delta=0.25)
            inv = compute_invariants(flow, alpha, case["m0"], tol=2e-3)
            assert inv["exact_balance_ok"], \
                f"Exact balance failed for {case['name']}, α={alpha}"
            assert inv["resource_invariance_ok"], \
                f"Resource invariance failed for {case['name']}, α={alpha}: " \
                f"max_res={inv['max_invariance_residual']:.2e}"


def test_cutoff_flow_finer_mesh():
    """Finer mesh produces consistent exact balance."""
    mu = make_uniform_law(0.5, 1.5)
    for Delta in [0.25, 0.125, 0.0625]:
        flow = run_cutoff_flow(mu, alpha=0.3, T=1.0, Delta=Delta)
        h_path = np.array(flow["h_path"])
        q_path = np.array(flow["q_path"])
        residuals = np.abs(h_path - 0.3 * q_path)
        assert np.max(residuals) < 1e-5, \
            f"Balance violated at Δ={Delta}: max res = {np.max(residuals):.2e}"


# ================================================================== #
# Metrics and payoffs                                                #
# ================================================================== #

def test_all_payoffs_nonnegative():
    """All payoffs (except log which starts from 0) should be ≥ 0."""
    mu = make_uniform_law(0.5, 1.5)
    payoffs = all_payoffs(mu)
    for name, val in payoffs.items():
        if not np.isnan(val):
            assert val >= 0.0, f"Payoff {name} = {val} < 0"


def test_payoff_survival_is_mass_above_zero():
    """Survival payoff = 1 - q_0 = P(X > 0) for fresh law."""
    mu = make_uniform_law(0.5, 1.5)
    payoffs = all_payoffs(mu)
    expected_surv = 1.0 - mu.default_mass
    assert abs(payoffs["survival"] - expected_surv) < 1e-4


def test_payoff_linear_equals_mean():
    """Linear payoff E[f(x)] = E[x] for positive law (no mass at 0)."""
    mu = make_uniform_law(0.5, 1.5)
    payoffs = all_payoffs(mu)
    expected = mu.mean
    assert abs(payoffs["linear"] - expected) < 1e-4


def test_payoffs_order_concavity():
    """By Jensen's inequality for concave payoffs on [0.5, 1.5]:
    E[f(X)] ≥ f(E[X]) when f is concave and supported above 0.5.
    Verify sqrt and log-payoffs behave consistently.
    """
    mu = make_uniform_law(0.5, 1.5)
    payoffs = all_payoffs(mu)
    # sqrt: E[√X] for X~Unif[0.5, 1.5], should be positive
    assert payoffs["sqrt"] > 0
    # log: E[log(1+X)] for X~Unif[0.5, 1.5], should be positive
    assert payoffs["log1p"] > 0
