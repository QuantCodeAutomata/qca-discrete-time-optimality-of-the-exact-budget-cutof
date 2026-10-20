"""
policies.py
-----------
Benchmark admissible controls for comparison against the optimal cutoff policy.

Admissible set A(μ, c) = {ν : μ ≼₁ ν, q_ν = q_μ, m_ν = m_μ - c}.
The conditions are:
  - Same default mass as the pre-allocation law (q_ν = q_μ).
  - Budget balance: m_ν = m_μ - c (reduce mean by exactly c).
  - Stochastic order: μ ≼₁ ν (ν truncated-moment-dominates μ component-wise).

Implemented benchmark policies (Custom — Context7 found no library equivalent):
  1. Proportional haircut: Y = (1 - λ)X with λ = c/m_μ on the positive part.
  2. Uniform haircut: subtract a flat constant from all reserves above a floor.
  3. Bottom-stripping: reduce the lowest reserves first.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from .laws import DiscreteLaw


def proportional_haircut_law(
    law: DiscreteLaw,
    c: float,
) -> DiscreteLaw | None:
    """Apply proportional haircut: Y = (1 - λ)X with λ = c/m_pos.

    This preserves q_ν = q_μ = law.q0 (only the positive part is scaled),
    reduces the mean by exactly c, but may violate stochastic order.
    Included as a benchmark (NOT generally admissible in ≼₁ sense).

    Parameters
    ----------
    law : DiscreteLaw
        Pre-allocation law μ̂.
    c : float
        Budget c > 0.

    Returns
    -------
    DiscreteLaw or None
        Scaled law if admissible, else None.
    """
    m = law.mean
    if c >= m or c <= 0:
        return None

    lam = c / m  # haircut rate
    scale = 1.0 - lam

    # Scale all positive support points
    new_grid = law.grid * scale  # shrink grid
    new_atoms = [(x * scale, w) for x, w in law.atoms]

    # New grid is no longer the canonical grid; we need to project back
    # Use the canonical grid and redistribute mass
    canonical_grid = law.grid
    dx = canonical_grid[1] - canonical_grid[0]
    new_density = np.zeros(len(canonical_grid))

    for i, (xi_new, wi) in enumerate(zip(new_grid, law.density)):
        if law.density[i] < 1e-18:
            continue
        wi = law.density[i]
        # Find nearest grid point
        idx = np.searchsorted(canonical_grid, xi_new) - 1
        idx = int(np.clip(idx, 0, len(canonical_grid) - 2))
        # Linear interpolation
        alpha_w = (xi_new - canonical_grid[idx]) / dx
        alpha_w = np.clip(alpha_w, 0.0, 1.0)
        new_density[idx] += wi * (1.0 - alpha_w)
        new_density[idx + 1] += wi * alpha_w

    new_atom_list = []
    for x_a, w_a in law.atoms:
        x_scaled = x_a * scale
        idx = np.searchsorted(canonical_grid, x_scaled) - 1
        idx = int(np.clip(idx, 0, len(canonical_grid) - 2))
        alpha_w = (x_scaled - canonical_grid[idx]) / dx
        alpha_w = np.clip(alpha_w, 0.0, 1.0)
        new_density[idx] += w_a * (1.0 - alpha_w)
        new_density[idx + 1] += w_a * alpha_w

    return DiscreteLaw(
        q0=law.q0,
        grid=canonical_grid,
        density=new_density,
        atoms=[],
    )


def flat_haircut_law(
    law: DiscreteLaw,
    c: float,
) -> DiscreteLaw | None:
    """Uniform flat cut: subtract a constant δ from all positive reserves.

    Solves ∫ max(x - δ, 0) dμ = m - c, i.e. the cut reduces mean by c.
    Not generally admissible but included as a diagnostic benchmark.

    Parameters
    ----------
    law : DiscreteLaw
        Pre-allocation law μ̂.
    c : float
        Budget to extract.
    """
    m = law.mean
    if c >= m or c <= 0:
        return None

    # Find δ such that ∫ max(x - δ, 0) μ(dx) = m - c
    # i.e. m - C_μ(δ) = m - c  ⟹ C_μ(δ) = c
    # This is the same as the cutoff! So flat haircut on the whole measure is NOT cutoff;
    # here we shift x -> max(x - δ, 0) which does change q.
    # So we need a different parameterisation.
    #
    # Instead: subtract δ from x > 0 and clip at 0.
    # New mean: ∫_{(0,∞)} max(x - δ, 0) μ(dx) = C_μ(δ) (call function!)
    # So we need C_μ(δ) = m - c, i.e. C_μ(δ) = m - c.
    # New q0_new = q_μ + μ((0, δ)) (reserves below δ get defaulted).
    # This does NOT preserve q_ν = q_μ! So it's a different type of admissible control.

    def f(delta):
        return law.call(delta) - (m - c)

    f0 = f(0.0)  # = m - (m - c) = c > 0
    x_max = float(law.grid[-1])
    if f(x_max) > 0:
        return None  # cannot extract enough
    try:
        delta = brentq(f, 0.0, x_max, xtol=1e-10)
    except ValueError:
        return None

    canonical_grid = law.grid
    dx = canonical_grid[1] - canonical_grid[0]
    new_density = np.zeros(len(canonical_grid))
    new_q0 = float(law.q0)

    # Shift grid points: max(x_i - delta, 0)
    shifted = np.maximum(canonical_grid - delta, 0.0)
    new_q0 += float(np.sum(law.density[canonical_grid <= delta]))

    for i in range(len(canonical_grid)):
        if law.density[i] < 1e-18:
            continue
        xi_new = shifted[i]
        if xi_new <= 0:
            continue  # already absorbed in q0
        idx = np.searchsorted(canonical_grid, xi_new) - 1
        idx = int(np.clip(idx, 0, len(canonical_grid) - 2))
        a_w = (xi_new - canonical_grid[idx]) / dx
        a_w = float(np.clip(a_w, 0.0, 1.0))
        new_density[idx] += law.density[i] * (1.0 - a_w)
        new_density[idx + 1] += law.density[i] * a_w

    for x_a, w_a in law.atoms:
        xi_new = max(x_a - delta, 0.0)
        if xi_new <= 0:
            new_q0 += w_a
        else:
            idx = np.searchsorted(canonical_grid, xi_new) - 1
            idx = int(np.clip(idx, 0, len(canonical_grid) - 2))
            a_w = (xi_new - canonical_grid[idx]) / dx
            a_w = float(np.clip(a_w, 0.0, 1.0))
            new_density[idx] += w_a * (1.0 - a_w)
            new_density[idx + 1] += w_a * a_w

    return DiscreteLaw(
        q0=float(np.clip(new_q0, 0.0, 1.0)),
        grid=canonical_grid,
        density=new_density,
        atoms=[],
    )


def bottom_stripping_law(
    law: DiscreteLaw,
    c: float,
) -> DiscreteLaw | None:
    """Bottom-stripping: reduce the smallest positive reserves to zero first.

    Removes mass from the bottom of the positive distribution until budget c
    is consumed (in mean reduction). Does NOT preserve q_ν = q_μ.
    Included as an admissible alternative of a different structural type.

    Parameters
    ----------
    law : DiscreteLaw
        Pre-allocation law μ̂.
    c : float
        Budget.
    """
    m = law.mean
    if c >= m or c <= 0:
        return None

    # Find threshold θ s.t. ∫_0^θ x μ(dx) = c
    # That is, D_μ(θ) = ∫_0^θ x μ(dx) = c  (lower truncated moment)
    # Equivalently: truncated_moment(θ) = c, i.e. m - C_μ(θ) = c, C_μ(θ) = m - c
    # Same equation as flat haircut! But here x -> x * 1_{x > θ} instead.
    # Actually: ∫_0^θ x μ(dx) = D_μ(θ) = θ·μ([0,θ]) - ... no.
    # D_μ(θ) = ∫ (x∧θ) μ(dx) = ∫_0^θ x μ(dx) + θ·μ((θ,∞))
    # We want ∫_0^θ x μ(dx) = c, not D_μ(θ) = c.

    # Let L(θ) = ∫_0^θ x μ(dx). Then L(0) = 0, L(∞) = m.
    # We need L(θ) = c.
    canonical_grid = law.grid
    dx = canonical_grid[1] - canonical_grid[0]
    # Compute L(θ) using trapezoidal sum up to index
    lower_moment_cum = np.cumsum(canonical_grid * law.density)

    # Add atom contributions iteratively (atoms sorted by x)
    atoms_sorted = sorted(law.atoms, key=lambda p: p[0])

    def L(theta: float) -> float:
        mask = canonical_grid <= theta
        val = float(np.dot(canonical_grid[mask], law.density[mask]))
        for x_a, w_a in atoms_sorted:
            if x_a <= theta:
                val += x_a * w_a
        return val

    L_max = L(float(canonical_grid[-1]) + 1.0)
    if c > L_max:
        return None

    try:
        theta = brentq(lambda t: L(t) - c, 0.0, float(canonical_grid[-1]) * 2, xtol=1e-10)
    except ValueError:
        return None

    # Strip all cells strictly below theta; apply a partial strip to the
    # boundary cell so the total mean reduction is exactly c.
    new_density = law.density.copy()
    new_q0 = float(law.q0)

    below_mask = canonical_grid < theta
    new_q0 += float(np.sum(new_density[below_mask]))
    c_stripped = float(np.dot(canonical_grid[below_mask], new_density[below_mask]))
    new_density[below_mask] = 0.0

    # Partial-strip the boundary cell to consume the remaining budget exactly.
    c_remaining = c - c_stripped
    boundary_candidates = np.where(
        (canonical_grid >= theta) & (new_density > 0) & (canonical_grid * new_density >= c_remaining)
    )[0]
    if len(boundary_candidates) > 0:
        b_idx = int(boundary_candidates[0])
        if canonical_grid[b_idx] > 0:
            strip_frac = c_remaining / (canonical_grid[b_idx] * new_density[b_idx])
            strip_frac = float(np.clip(strip_frac, 0.0, 1.0))
            new_q0 += new_density[b_idx] * strip_frac
            new_density[b_idx] *= (1.0 - strip_frac)

    new_atoms = []
    for x_a, w_a in law.atoms:
        if x_a <= theta:
            new_q0 += w_a
        else:
            new_atoms.append((x_a, w_a))

    return DiscreteLaw(
        q0=float(np.clip(new_q0, 0.0, 1.0)),
        grid=canonical_grid,
        density=new_density,
        atoms=new_atoms,
    )


def benchmark_policies(
    law: DiscreteLaw,
    c: float,
) -> dict[str, DiscreteLaw | None]:
    """Return all benchmark admissible policies for a given (μ, c) pair.

    Parameters
    ----------
    law : DiscreteLaw
        Pre-allocation law μ̂.
    c : float
        Exact budget.

    Returns
    -------
    dict mapping policy name to resulting law (or None if inapplicable).
    """
    return {
        "proportional_haircut": proportional_haircut_law(law, c),
        "flat_haircut": flat_haircut_law(law, c),
        "bottom_stripping": bottom_stripping_law(law, c),
    }
