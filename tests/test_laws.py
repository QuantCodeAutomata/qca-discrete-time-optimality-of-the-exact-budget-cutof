"""
tests/test_laws.py
------------------
Tests for src/laws.py: DiscreteLaw, factory functions, call/truncated-moment functions,
and the d_T^# stability metric.
"""

import numpy as np
import pytest
from src.laws import (
    DiscreteLaw,
    make_uniform_law,
    make_mixture_uniform_law,
    make_atomic_law,
    standard_test_cases,
    uncontrolled_default_curve,
    d_T_sharp,
)


# ================================================================== #
# Fixtures                                                            #
# ================================================================== #

@pytest.fixture
def uniform_law():
    return make_uniform_law(0.5, 1.5)


@pytest.fixture
def atomic_law():
    return make_atomic_law([0.25, 1.25], [0.5, 0.5])


@pytest.fixture
def mixture_law():
    return make_mixture_uniform_law(0.2, 0.4, 1.2, 1.4, p1=0.5)


# ================================================================== #
# Basic property tests                                               #
# ================================================================== #

def test_uniform_law_total_mass(uniform_law):
    assert abs(uniform_law.total_mass - 1.0) < 1e-4


def test_uniform_law_mean(uniform_law):
    assert abs(uniform_law.mean - 1.0) < 1e-3, f"Expected mean≈1.0, got {uniform_law.mean}"


def test_atomic_law_total_mass(atomic_law):
    assert abs(atomic_law.total_mass - 1.0) < 1e-10


def test_atomic_law_mean(atomic_law):
    expected_mean = 0.5 * 0.25 + 0.5 * 1.25
    assert abs(atomic_law.mean - expected_mean) < 1e-10, \
        f"Expected mean={expected_mean}, got {atomic_law.mean}"


def test_atomic_law_default_mass_zero(atomic_law):
    assert abs(atomic_law.default_mass) < 1e-10


def test_mixture_law_mean(mixture_law):
    expected_mean = 0.5 * (0.2 + 0.4) / 2 + 0.5 * (1.2 + 1.4) / 2
    assert abs(mixture_law.mean - expected_mean) < 1e-3


def test_uniform_law_no_default_mass(uniform_law):
    assert uniform_law.default_mass == 0.0


# ================================================================== #
# Call function tests                                                #
# ================================================================== #

def test_call_at_zero_equals_mean(uniform_law):
    """C_μ(0) = ∫(x-0)^+ dμ = mean"""
    assert abs(uniform_law.call(0.0) - uniform_law.mean) < 1e-4


def test_call_above_support_is_zero(uniform_law):
    """C_μ(a) ≈ 0 for a well above support"""
    assert uniform_law.call(10.0) < 1e-8


def test_call_monotone_decreasing(uniform_law):
    """C_μ(a) is monotone non-increasing"""
    a_grid = np.linspace(0, 3, 50)
    calls = uniform_law.call_vec(a_grid)
    diffs = np.diff(calls)
    assert np.all(diffs <= 1e-10), f"Non-monotone call: min diff = {diffs.min():.4e}"


def test_call_non_negative(uniform_law):
    """C_μ(a) ≥ 0 for all a"""
    a_grid = np.linspace(0, 5, 100)
    calls = uniform_law.call_vec(a_grid)
    assert np.all(calls >= -1e-10)


def test_atomic_law_call_exact():
    """For atomic law with atoms at 0.25 and 1.25: C_μ(1.0) = 0.5*(1.25-1.0)=0.125"""
    law = make_atomic_law([0.25, 1.25], [0.5, 0.5])
    expected = 0.5 * max(0.25 - 1.0, 0) + 0.5 * max(1.25 - 1.0, 0)
    assert abs(law.call(1.0) - expected) < 1e-10


def test_truncated_moment_plus_call_equals_mean(uniform_law):
    """D_μ(a) + C_μ(a) = mean for all a"""
    a_grid = np.linspace(0.1, 2.0, 20)
    d_vals = uniform_law.truncated_moment_vec(a_grid)
    c_vals = uniform_law.call_vec(a_grid)
    residuals = np.abs(d_vals + c_vals - uniform_law.mean)
    assert np.max(residuals) < 1e-10, f"max residual = {np.max(residuals):.2e}"


def test_truncated_moment_non_decreasing(uniform_law):
    """D_μ(a) is non-decreasing"""
    a_grid = np.linspace(0, 3, 50)
    d_vals = uniform_law.truncated_moment_vec(a_grid)
    diffs = np.diff(d_vals)
    assert np.all(diffs >= -1e-10)


def test_truncated_moment_bounded_by_mean(uniform_law):
    """D_μ(a) ≤ mean for all a"""
    a_grid = np.linspace(0, 5, 50)
    d_vals = uniform_law.truncated_moment_vec(a_grid)
    assert np.all(d_vals <= uniform_law.mean + 1e-10)


# ================================================================== #
# Edge cases                                                         #
# ================================================================== #

def test_call_at_support_boundary():
    """C_μ(b) ≈ 0 for uniform[a,b]: all mass is at or below b."""
    law = make_uniform_law(0.5, 1.5)
    assert law.call(1.5) < 1e-6  # very small


def test_single_atom_law():
    """Single atom at x=1.0: call(a) = max(1.0-a, 0)."""
    law = make_atomic_law([1.0], [1.0])
    assert abs(law.call(0.5) - 0.5) < 1e-10
    assert abs(law.call(1.0) - 0.0) < 1e-10
    assert abs(law.call(2.0) - 0.0) < 1e-10
    assert abs(law.mean - 1.0) < 1e-10


def test_three_atom_law():
    """Three-atom law: verify call at intermediate strike."""
    law = make_atomic_law([0.5, 1.0, 1.5], [1/3, 1/3, 1/3])
    # call(1.0) = (1/3)*(0.5-1.0)^+ + (1/3)*(1.0-1.0)^+ + (1/3)*(1.5-1.0)^+ = 1/3*0.5
    expected = (1/3) * 0.5
    assert abs(law.call(1.0) - expected) < 1e-10


# ================================================================== #
# Standard test cases                                                #
# ================================================================== #

def test_standard_test_cases_structure():
    cases = standard_test_cases()
    assert len(cases) == 3
    for case in cases:
        for key in ["name", "law", "m0", "a_inf", "b_sup", "alpha_values"]:
            assert key in case, f"Missing key '{key}'"


def test_standard_test_case_means():
    cases = standard_test_cases()
    expected_means = [1.0, 0.8, 0.75]
    for case, m_exp in zip(cases, expected_means):
        assert abs(case["law"].mean - m_exp) < 1e-2, \
            f"Case {case['name']}: mean={case['law'].mean:.4f}, expected≈{m_exp}"


def test_standard_test_cases_admissibility():
    cases = standard_test_cases()
    for case in cases:
        for alpha in case["alpha_values"]:
            if alpha < case["m0"]:
                assert case["law"].mean > alpha, \
                    f"Case {case['name']}: mean={case['law'].mean} ≤ alpha={alpha}"


# ================================================================== #
# Uncontrolled default curve                                         #
# ================================================================== #

def test_uncontrolled_default_curve_monotone():
    """q_μ^h(t) must be non-decreasing in t."""
    law = make_uniform_law(0.5, 1.5)
    t_grid = np.linspace(0.01, 2.0, 30)
    q_curve = uncontrolled_default_curve(law, t_grid)
    diffs = np.diff(q_curve)
    assert np.all(diffs >= -1e-10), f"Non-monotone q_curve: min diff = {diffs.min():.4e}"


def test_uncontrolled_default_curve_in_01():
    """q_μ^h(t) ∈ [0, 1] for all t."""
    law = make_uniform_law(0.5, 1.5)
    t_grid = np.linspace(0.01, 5.0, 50)
    q_curve = uncontrolled_default_curve(law, t_grid)
    assert np.all(q_curve >= -1e-10)
    assert np.all(q_curve <= 1.0 + 1e-10)


def test_uncontrolled_default_curve_approaches_one():
    """For large t, q_μ^h(t) grows toward 1 (all mass eventually absorbed).

    Analytic value for Unif[0.5, 1.5] at t=100:
      E[2Φ(-X/√100)] = ∫_{0.5}^{1.5} 2Φ(-x/10) dx ≈ 0.920.

    A threshold of 0.99 requires t > 1500; we use 0.88 which is well below the
    analytic result and tests that the curve is substantially elevated.
    """
    law = make_uniform_law(0.5, 1.5)
    t_grid = np.array([100.0])
    q_val = float(uncontrolled_default_curve(law, t_grid)[0])
    assert q_val > 0.88, f"q at t=100 should be >0.88 (analytic≈0.92), got {q_val:.4f}"


def test_atomic_default_curve_exact():
    """Atomic law: q_μ^h(t) = Σ w_i · 2Φ(-x_i/√t)."""
    from scipy.stats import norm
    law = make_atomic_law([0.25, 1.25], [0.5, 0.5])
    t_val = 0.5
    t_grid = np.array([t_val])
    q_computed = float(uncontrolled_default_curve(law, t_grid)[0])
    q_exact = 0.5 * 2 * norm.cdf(-0.25 / np.sqrt(t_val)) + 0.5 * 2 * norm.cdf(-1.25 / np.sqrt(t_val))
    assert abs(q_computed - q_exact) < 1e-10


# ================================================================== #
# d_T^# metric                                                       #
# ================================================================== #

def test_d_T_sharp_self_is_zero():
    """d_T^#(μ, μ) = 0."""
    law = make_uniform_law(0.5, 1.5)
    a_grid = np.linspace(0, 3, 100)
    t_grid = np.linspace(0.01, 1.0, 30)
    d = d_T_sharp(law, law, a_grid, t_grid)
    assert abs(d) < 1e-12, f"d_T^#(μ,μ) = {d:.2e} ≠ 0"


def test_d_T_sharp_nonnegative():
    """d_T^# ≥ 0."""
    law1 = make_uniform_law(0.5, 1.5)
    law2 = make_uniform_law(0.6, 1.6)
    a_grid = np.linspace(0, 3, 100)
    t_grid = np.linspace(0.01, 1.0, 20)
    d = d_T_sharp(law1, law2, a_grid, t_grid)
    assert d >= 0.0


def test_d_T_sharp_small_perturbation():
    """Nearby laws should have small d_T^#."""
    law1 = make_uniform_law(0.5, 1.5)
    law2 = make_uniform_law(0.5 + 0.01, 1.5 + 0.01)
    a_grid = np.linspace(0, 3, 100)
    t_grid = np.linspace(0.01, 1.0, 20)
    d = d_T_sharp(law1, law2, a_grid, t_grid)
    assert d < 0.1, f"d_T^# = {d:.4f} too large for ε=0.01 shift"
