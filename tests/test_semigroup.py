"""
tests/test_semigroup.py
-----------------------
Tests for src/semigroup.py: absorbed Brownian motion semigroup S_t.

Key mathematical properties:
  - Mass conservation: total mass ≤ initial total mass (mass is absorbed at 0)
  - Default mass q_{S_t μ} ≥ q_μ (monotone increasing defaults)
  - q_{S_t μ} = ∫ 2Φ(-x/√t) μ(dx) for μ with no mass at 0
  - Semigroup property: S_{t+s} μ = S_t (S_s μ)
"""

import numpy as np
import pytest
from scipy.stats import norm
from src.laws import (
    make_uniform_law,
    make_atomic_law,
    make_mixture_uniform_law,
    DiscreteLaw,
)
from src.semigroup import apply_semigroup, absorbed_bm_default_prob


# ================================================================== #
# Basic propagation tests                                            #
# ================================================================== #

def test_semigroup_preserves_total_mass():
    """Total mass ≤ 1 after propagation (some absorbed at 0)."""
    mu = make_uniform_law(0.5, 1.5)
    mu_t = apply_semigroup(mu, 0.1)
    assert mu_t.total_mass <= 1.0 + 1e-6


def test_semigroup_default_mass_increases():
    """Default mass grows monotonically in t."""
    mu = make_uniform_law(0.5, 1.5)
    q0 = mu.default_mass
    for t in [0.01, 0.1, 0.5, 1.0]:
        mu_t = apply_semigroup(mu, t)
        assert mu_t.default_mass >= q0 - 1e-10, \
            f"Default mass decreased at t={t}: {mu_t.default_mass:.4f} < {q0:.4f}"


def test_atomic_law_default_prob_exact():
    """For atomic law, q_{S_t μ} = Σ w_i · 2Φ(-x_i/√t) exactly."""
    law = make_atomic_law([0.25, 1.25], [0.5, 0.5])
    t = 0.5
    mu_t = apply_semigroup(law, t)
    q_computed = mu_t.default_mass
    q_exact = 0.5 * 2 * norm.cdf(-0.25 / np.sqrt(t)) + 0.5 * 2 * norm.cdf(-1.25 / np.sqrt(t))
    assert abs(q_computed - q_exact) < 1e-5, \
        f"q computed={q_computed:.6f}, expected={q_exact:.6f}"


def test_single_atom_default_prob():
    """For δ_x, P(τ_0 ≤ t) = 2Φ(-x/√t)."""
    for x in [0.5, 1.0, 2.0]:
        for t in [0.1, 0.5, 1.0]:
            law = make_atomic_law([x], [1.0])
            mu_t = apply_semigroup(law, t)
            q_expected = 2.0 * norm.cdf(-x / np.sqrt(t))
            assert abs(mu_t.default_mass - q_expected) < 1e-5, \
                f"x={x}, t={t}: q={mu_t.default_mass:.6f} vs {q_expected:.6f}"


def test_absorbed_bm_default_prob_formula():
    """Direct function test: absorbed_bm_default_prob(x, t) = 2Φ(-x/√t)."""
    x_vals = np.array([0.5, 1.0, 2.0])
    t_vals = np.array([0.1, 0.5, 1.0])
    for x in x_vals:
        for t in t_vals:
            computed = absorbed_bm_default_prob(x, t)
            expected = 2.0 * norm.cdf(-x / np.sqrt(t))
            assert abs(computed - expected) < 1e-12


def test_semigroup_mean_decreases():
    """Mean of the law decreases as absorbed mass escapes to 0."""
    mu = make_uniform_law(0.5, 1.5)
    m0 = mu.mean
    for t in [0.1, 0.5, 1.0]:
        mu_t = apply_semigroup(mu, t)
        assert mu_t.mean <= m0 + 1e-8, f"Mean increased at t={t}: {mu_t.mean:.4f} > {m0:.4f}"


def test_semigroup_mass_conservation_with_default():
    """q_{S_t μ} + ∫ density(x) dx = 1 (total mass conserved)."""
    mu = make_uniform_law(0.5, 1.5)
    for t in [0.1, 0.5, 1.0]:
        mu_t = apply_semigroup(mu, t)
        assert abs(mu_t.total_mass - 1.0) < 2e-3, \
            f"Total mass = {mu_t.total_mass:.6f} at t={t}"


def test_semigroup_zero_time():
    """S_dt μ ≈ μ for small dt (near-identity).

    Use dt=0.001 so the kernel width sqrt(dt) ≈ 0.032 >> grid spacing dx=0.002,
    ensuring the Gaussian kernel is well-resolved on the discrete grid.
    At dt=1e-6 the kernel is sub-grid and the numerical integration is
    unreliable — that is a grid-resolution limitation, not a code error.
    """
    mu = make_uniform_law(0.5, 1.5)
    mu_t = apply_semigroup(mu, 0.001)
    assert abs(mu_t.mean - mu.mean) < 1e-3
    assert abs(mu_t.default_mass - mu.default_mass) < 1e-6


def test_semigroup_supports_mixture_law():
    """Mixture law propagates without error."""
    mu = make_mixture_uniform_law(0.2, 0.4, 1.2, 1.4, p1=0.5)
    mu_t = apply_semigroup(mu, 0.5)
    assert mu_t.total_mass > 0
    assert mu_t.default_mass >= 0


def test_semigroup_default_absorbed_boundary():
    """At long times, most mass is absorbed at 0.

    For Unif[0.5, 1.5] at t=100:
      E[2Φ(-X/√100)] = ∫_{0.5}^{1.5} 2Φ(-x/10) dx ≈ 0.920  (analytic value).

    Reaching >0.99 requires t > 1500 which is impractical numerically; the
    threshold 0.88 is strict but definitely satisfied by the analytic result.
    """
    mu = make_uniform_law(0.5, 1.5)
    mu_t = apply_semigroup(mu, 100.0)
    assert mu_t.default_mass > 0.88, \
        f"Expected >0.88 absorption at t=100, got {mu_t.default_mass:.4f}"


# ================================================================== #
# Semigroup approximation tests                                      #
# ================================================================== #

def test_semigroup_mass_balance_atomic():
    """For atomic law: S_t mass balance is exact."""
    law = make_atomic_law([0.5, 1.0], [0.5, 0.5])
    t = 0.3
    mu_t = apply_semigroup(law, t)
    # Expected default mass
    q_expected = 0.5 * 2 * norm.cdf(-0.5 / np.sqrt(t)) + 0.5 * 2 * norm.cdf(-1.0 / np.sqrt(t))
    assert abs(mu_t.default_mass - q_expected) < 1e-4
