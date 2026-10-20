"""
cutoff.py
---------
Exact-budget cutoff / capping operator T_c and the one-step optimality operator.

The capping operator (Custom — Context7 found no library equivalent):
  T_c μ = (x ↦ x ∧ κ_c(μ))# μ
where κ_c(μ) is the unique threshold satisfying C_μ(κ) = c, and C_μ is the call function.

Useful identities checked numerically after each cap:
  C_{T_c μ}(a) = (C_μ(a) - c)⁺
  D_{T_c μ}(a) = min(D_μ(a), m_μ - c)

The full one-step operator is:
  G_Δ(μ) = T_{c_Δ(μ)} S_Δ μ
with per-step budget c = α·Δq where Δq = q_{S_Δ μ} - q_μ.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from .laws import DiscreteLaw
from .semigroup import apply_semigroup


# ------------------------------------------------------------------ #
# Threshold (budget inversion)                                        #
# ------------------------------------------------------------------ #

def find_threshold(
    law: DiscreteLaw,
    c: float,
    xtol: float = 1e-10,
) -> float:
    """Solve C_μ(κ) = c for the unique threshold κ ≥ 0.

    C_μ is monotone non-increasing from C_μ(0) = mean (at a=0) to 0 (as a→∞).
    If c = 0, κ is at or above the supremum of the support (T_0 = identity).
    If c ≥ mean, raises ValueError.

    Parameters
    ----------
    law : DiscreteLaw
        The measure μ to invert.
    c : float
        Budget c ≥ 0.
    xtol : float
        Tolerance for root finder.

    Returns
    -------
    float
        Threshold κ = C_μ^{-1}(c).
    """
    if c < 0:
        raise ValueError(f"Budget c must be ≥ 0, got {c}")

    m = law.mean
    if m < 1e-15:
        raise ValueError("Law mean is (near) zero; cannot cap with positive budget.")

    if c <= 0.0:
        # T_0 μ = μ; threshold is effectively ∞ (no capping)
        # Return the supremum of positive support as a sentinal
        x_pos, _ = law.all_positive_points_and_weights()
        return float(x_pos[-1]) if len(x_pos) > 0 else 0.0

    if c >= m - 1e-12:
        raise ValueError(
            f"Budget c={c:.6g} ≥ mean={m:.6g}; T_c undefined "
            "(entire mean would be extracted)."
        )

    # Bracket: C_μ(0) = m > c, C_μ(x_max) ≈ 0 < c
    x_pos, w_pos = law.all_positive_points_and_weights()
    x_hi = float(x_pos[-1]) if len(x_pos) > 0 else 10.0

    # Extend upper bracket if needed
    for _ in range(20):
        if law.call(x_hi) < c:
            break
        x_hi *= 2.0

    # Lower bracket: the threshold is above the region where all mass sits
    x_lo = 0.0

    def f(kappa: float) -> float:
        return law.call(kappa) - c

    # Check bracket
    f_lo = f(x_lo)  # should be m - c > 0
    f_hi = f(x_hi)  # should be ≈ -c < 0

    if f_lo <= 0:
        return 0.0
    if f_hi >= 0:
        # Try a larger upper bracket
        x_hi_new = x_hi * 10
        if f(x_hi_new) < 0:
            x_hi = x_hi_new
        else:
            raise ValueError(f"Cannot bracket threshold: f(x_lo)={f_lo:.4g}, f(x_hi)={f(x_hi):.4g}")

    kappa = brentq(f, x_lo, x_hi, xtol=xtol, maxiter=200)
    return float(kappa)


# ------------------------------------------------------------------ #
# Capping operator T_c                                                #
# ------------------------------------------------------------------ #

def apply_cap(
    law: DiscreteLaw,
    c: float,
    xtol: float = 1e-10,
) -> tuple[DiscreteLaw, float]:
    """Apply the capping operator T_c to law, returning (T_c μ, κ).

    T_c μ = (x ↦ x ∧ κ)# μ:
      - Mass at points y > κ is moved to κ (creates/augments atom at κ).
      - Mass at y ≤ κ is unchanged.
      - Atom at 0 is unchanged.

    Parameters
    ----------
    law : DiscreteLaw
        Input measure μ with q_{μ} = law.q0.
    c : float
        Budget c ≥ 0. If c=0, returns (law, ∞) with law unchanged.

    Returns
    -------
    (DiscreteLaw, float)
        Capped law T_c μ and the threshold κ.
    """
    if c <= 0.0:
        return law, np.inf

    kappa = find_threshold(law, c, xtol=xtol)

    # ---- Grid density: zero out mass above κ, accumulate it ----
    new_density = law.density.copy()
    above_grid = law.grid > kappa
    mass_above_grid = float(np.sum(new_density[above_grid]))
    new_density[above_grid] = 0.0

    # ---- Extra atoms: zero out atoms above κ ----
    mass_above_atoms = 0.0
    new_atoms = []
    for x_a, w_a in law.atoms:
        if x_a > kappa:
            mass_above_atoms += w_a
        else:
            new_atoms.append((x_a, w_a))

    total_mass_above = mass_above_grid + mass_above_atoms

    # Add the accumulated mass as a new atom at kappa
    if total_mass_above > 1e-16:
        new_atoms.append((kappa, total_mass_above))

    capped_law = DiscreteLaw(
        q0=law.q0,
        grid=law.grid,
        density=new_density,
        atoms=new_atoms,
    )
    return capped_law, kappa


# ------------------------------------------------------------------ #
# One-step operator G_Δ                                              #
# ------------------------------------------------------------------ #

def cutoff_step(
    mu: DiscreteLaw,
    alpha: float,
    dt: float,
    output_grid: np.ndarray | None = None,
) -> tuple[DiscreteLaw, float, float, float]:
    """Apply one step of the discrete cutoff operator G_Δ:
      1. Propagate: μ̂ = S_Δ μ
      2. Compute new default: Δq = q_{μ̂} - q_μ
      3. Budget: c = α · Δq
      4. Cap: μ_new = T_c μ̂

    Parameters
    ----------
    mu : DiscreteLaw
        Current post-allocation law μ_n.
    alpha : float
        Loss allocation rate α > 0.
    dt : float
        Time step Δ > 0.
    output_grid : ndarray or None
        Grid for intermediate propagated law.

    Returns
    -------
    (mu_new, kappa, delta_q, c_budget)
        mu_new   : post-allocation law μ_{n+1}
        kappa    : threshold κ_{n+1}
        delta_q  : newly generated default mass Δq_{n+1}
        c_budget : per-step budget c_{n+1} = α·Δq_{n+1}
    """
    q_before = mu.default_mass

    # Step 1: propagate
    mu_hat = apply_semigroup(mu, dt, output_grid=output_grid)

    # Step 2: new default mass
    q_after = mu_hat.default_mass
    delta_q = max(q_after - q_before, 0.0)  # Δq ≥ 0

    # Step 3: budget
    c_budget = alpha * delta_q

    # Step 4: cap
    if c_budget < 1e-15:
        # No allocation needed (no new defaults)
        return mu_hat, np.inf, delta_q, c_budget

    mu_new, kappa = apply_cap(mu_hat, c_budget)
    return mu_new, kappa, delta_q, c_budget


# ------------------------------------------------------------------ #
# Full discrete cutoff law flow                                       #
# ------------------------------------------------------------------ #

def run_cutoff_flow(
    mu0: DiscreteLaw,
    alpha: float,
    T: float,
    Delta: float,
) -> dict:
    """Run the full discrete cutoff allocation scheme.

    Computes the sequence:
      μ_0^Δ = μ_0
      μ_{n+1}^Δ = G_Δ(μ_n^Δ)  for n = 0, 1, ..., M-1
    where M = T/Δ must be a positive integer.

    Parameters
    ----------
    mu0 : DiscreteLaw
        Initial law μ_0 (post-allocation at time 0).
    alpha : float
        Loss allocation rate α.
    T : float
        Terminal time.
    Delta : float
        Time step Δ (must divide T exactly).

    Returns
    -------
    dict with keys:
      'laws'      : list[DiscreteLaw] of length M+1 (post-allocation laws μ_n)
      'kappas'    : list[float] of length M (thresholds; inf if no allocation)
      'q_path'    : ndarray(M+1,) of default masses q_n
      'h_path'    : ndarray(M+1,) of cumulative allocated loss h_n = α·q_n
      'm_path'    : ndarray(M+1,) of mean reserves m_n
      'delta_q'   : ndarray(M,) of per-step new defaults Δq_n
      'budgets'   : ndarray(M,) of per-step budgets c_n
      't_grid'    : ndarray(M+1,) of allocation times t_n = n·Δ
      'M'         : int number of steps
      'Delta'     : float time step
      'alpha'     : float alpha
    """
    M_float = T / Delta
    M = int(round(M_float))
    if abs(M - M_float) > 1e-8:
        raise ValueError(f"T/Delta={T/Delta:.6g} must be an integer; got M={M}")

    t_grid = np.arange(M + 1) * Delta
    laws = [mu0]
    kappas = []
    q_path = np.zeros(M + 1)
    h_path = np.zeros(M + 1)
    m_path = np.zeros(M + 1)
    delta_qs = np.zeros(M)
    budgets = np.zeros(M)

    q_path[0] = mu0.default_mass
    h_path[0] = alpha * q_path[0]
    m_path[0] = mu0.mean

    mu_curr = mu0
    for n in range(M):
        mu_next, kappa, dq, c_bud = cutoff_step(
            mu_curr, alpha, Delta, output_grid=mu0.grid
        )
        laws.append(mu_next)
        kappas.append(kappa)
        delta_qs[n] = dq
        budgets[n] = c_bud
        q_path[n + 1] = mu_next.default_mass
        h_path[n + 1] = alpha * q_path[n + 1]
        m_path[n + 1] = mu_next.mean
        mu_curr = mu_next

    return {
        "laws": laws,
        "kappas": kappas,
        "q_path": q_path,
        "h_path": h_path,
        "m_path": m_path,
        "delta_q": delta_qs,
        "budgets": budgets,
        "t_grid": t_grid,
        "M": M,
        "Delta": Delta,
        "alpha": alpha,
    }


# ------------------------------------------------------------------ #
# Verification identities                                             #
# ------------------------------------------------------------------ #

def verify_cap_identities(
    law: DiscreteLaw,
    c: float,
    a_grid: np.ndarray,
    tol: float = 1e-5,
) -> dict:
    """Check C_{T_c μ}(a) = (C_μ(a) - c)⁺ and D_{T_c μ}(a) = min(D_μ(a), m-c).

    Returns a dict with 'max_call_residual' and 'max_d_residual'.
    """
    m = law.mean
    capped, kappa = apply_cap(law, c)

    C_orig = law.call_vec(a_grid)
    C_capped = capped.call_vec(a_grid)
    D_orig = law.truncated_moment_vec(a_grid)
    D_capped = capped.truncated_moment_vec(a_grid)

    C_expected = np.maximum(C_orig - c, 0.0)
    D_expected = np.minimum(D_orig, m - c)

    call_residual = float(np.max(np.abs(C_capped - C_expected)))
    d_residual = float(np.max(np.abs(D_capped - D_expected)))

    return {
        "max_call_residual": call_residual,
        "max_d_residual": d_residual,
        "max_trunc_residual": d_residual,  # alias for test compatibility
        "kappa": kappa,
        "passed": (call_residual < tol) and (d_residual < tol),
    }
