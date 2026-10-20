"""
semigroup.py
------------
Absorbed Brownian motion semigroup S_t applied to DiscreteLaw objects.

Transition density (Custom — Context7 found no library equivalent, paper Eq.):
  p_t^0(x, y) = (2πt)^{-1/2} [exp(-(x-y)²/(2t)) - exp(-(x+y)²/(2t))]
  for x, y > 0.

Default probability for a particle starting at x > 0:
  P_x(τ_0 ≤ t) = 2 Φ̄(x/√t) = 2(1 - Φ(x/√t)) = 2Φ(-x/√t).

The semigroup acts on a DiscreteLaw as follows:
  1. Existing atom at 0 stays at 0.
  2. Each positive mass element (grid or atom) at x generates:
     - New absorption mass: w · P_x(τ_0 ≤ t)  →  added to q0
     - New density contribution: w · p_t^0(x, ·) on (0, ∞)  →  evaluated at grid
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from .laws import DiscreteLaw


def _absorbed_bm_density_matrix(
    x_from: np.ndarray,
    x_to: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Compute p_dt^0(x_i, y_j) for all pairs.

    Parameters
    ----------
    x_from : ndarray, shape (M,)
        Source positions (positive).
    x_to : ndarray, shape (N,)
        Target positions (positive grid points).
    dt : float
        Time step > 0.

    Returns
    -------
    K : ndarray, shape (M, N)
        Transition density matrix. K[i, j] = p_dt^0(x_from[i], x_to[j]).
    """
    # Vectorised computation using broadcasting
    X = x_from[:, np.newaxis]  # (M, 1)
    Y = x_to[np.newaxis, :]    # (1, N)
    std = np.sqrt(dt)
    # p_t^0(x,y) = φ((y-x)/√t)/√t - φ((y+x)/√t)/√t
    term1 = norm.pdf(Y - X, scale=std)  # (M, N)
    term2 = norm.pdf(Y + X, scale=std)  # (M, N)
    K = term1 - term2
    np.maximum(K, 0.0, out=K)  # numerical safeguard: density ≥ 0
    return K  # (M, N)


def _absorption_prob(x: np.ndarray, dt: float) -> np.ndarray:
    """P_x(τ_0 ≤ dt) = 2 Φ(-x/√dt) for positive x values.

    Parameters
    ----------
    x : ndarray
        Positive starting positions.
    dt : float
        Time step.

    Returns
    -------
    ndarray of same shape as x, values in [0, 1].
    """
    return 2.0 * norm.cdf(-x / np.sqrt(dt))


def absorbed_bm_default_prob(x: float, t: float) -> float:
    """Public API: P_x(τ_0 ≤ t) = 2 Φ(-x/√t) for scalar x > 0, t > 0.

    Parameters
    ----------
    x : float
        Positive starting position.
    t : float
        Time horizon > 0.

    Returns
    -------
    float in [0, 1].
    """
    return float(2.0 * norm.cdf(-x / np.sqrt(t)))


def apply_semigroup(
    law: DiscreteLaw,
    dt: float,
    output_grid: np.ndarray | None = None,
) -> DiscreteLaw:
    """Apply the absorbed Brownian semigroup S_dt to a DiscreteLaw.

    The output law is represented on output_grid (defaults to law.grid).
    All atoms from the input law are propagated:
      - atom at 0 remains at 0
      - positive grid density and extra atoms contribute to:
          new q0 (absorbed fraction) + new density on output_grid

    Parameters
    ----------
    law : DiscreteLaw
        Input measure μ.
    dt : float
        Time step; must be > 0.
    output_grid : ndarray or None
        Grid for the output law. If None, uses law.grid.

    Returns
    -------
    DiscreteLaw
        S_dt μ represented on output_grid.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    if output_grid is None:
        output_grid = law.grid

    dx = output_grid[1] - output_grid[0]  # uniform grid spacing (approx)
    N = len(output_grid)
    new_density = np.zeros(N)
    new_q0 = float(law.q0)  # atom at 0 stays

    # ---------- Contribution from grid density ----------
    active_mask = law.density > 1e-18
    if np.any(active_mask):
        x_src = law.grid[active_mask]
        w_src = law.density[active_mask]

        # Absorption
        p_abs = _absorption_prob(x_src, dt)
        new_q0 += float(np.dot(w_src, p_abs))

        # Density on output_grid
        # K shape: (len(x_src), N)
        K = _absorbed_bm_density_matrix(x_src, output_grid, dt)
        new_density += w_src @ K * dx  # (N,) — integral over source weights

    # ---------- Contribution from extra positive atoms ----------
    for x_a, w_a in law.atoms:
        p_abs = float(_absorption_prob(np.array([x_a]), dt)[0])
        new_q0 += w_a * p_abs
        K_row = _absorbed_bm_density_matrix(
            np.array([x_a]), output_grid, dt
        )[0]  # (N,)
        new_density += w_a * K_row * dx

    # Normalise q0 to [0,1]; minor numerical drift can occur
    new_q0 = float(np.clip(new_q0, 0.0, 1.0))

    return DiscreteLaw(
        q0=new_q0,
        grid=output_grid,
        density=new_density,
        atoms=[],  # S_t destroys atoms; all mass spreads to continuous density
    )


def semigroup_call(
    law: DiscreteLaw,
    a: float,
    dt: float,
) -> float:
    """Compute C_{S_dt μ}(a) analytically without materialising S_dt μ.

    Uses the identity (paper implementation note):
      C_{S_dt μ}(a) = ∫ call_abs(x, a, dt) μ(dx)
    where
      call_abs(x, a, t) = ∫_0^∞ (y-a)⁺ p_t^0(x, y) dy
                        = (x-a)Φ((x-a)/√t) + √t φ((x-a)/√t)
                          - √t φ((x+a)/√t) + (x+a)Φ̄((x+a)/√t)
    (Custom — Context7 found no library equivalent)

    Parameters
    ----------
    law : DiscreteLaw
        Input measure μ.
    a : float
        Strike ≥ 0.
    dt : float
        Time step.
    """
    std = np.sqrt(dt)

    def _call_abs_scalar(x: float) -> float:
        """∫_0^∞ (y-a)⁺ p_dt^0(x, y) dy for x > 0, a ≥ 0."""
        if a < 0:
            raise ValueError("Strike a must be ≥ 0")
        d1 = (x - a) / std
        d2 = (x + a) / std
        val = (x - a) * norm.cdf(d1) + std * norm.pdf(d1)
        val -= (x + a) * norm.cdf(-d2) + std * norm.pdf(d2)
        # Equivalent: (x-a)*Φ(d1)+std*φ(d1) - √t*φ(d2) + (x+a)*Φ̄(d2)
        # via the reflection term
        return max(val, 0.0)

    total = 0.0
    # Grid contribution
    active_mask = law.density > 1e-18
    if np.any(active_mask):
        x_src = law.grid[active_mask]
        w_src = law.density[active_mask]
        d1 = (x_src - a) / std
        d2 = (x_src + a) / std
        vals = (x_src - a) * norm.cdf(d1) + std * norm.pdf(d1)
        vals -= (x_src + a) * norm.cdf(-d2) + std * norm.pdf(d2)
        np.maximum(vals, 0.0, out=vals)
        total += float(np.dot(w_src, vals))

    # Atom contributions
    for x_a, w_a in law.atoms:
        total += w_a * _call_abs_scalar(x_a)

    return total
