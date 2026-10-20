"""
laws.py
-------
Measure representations, call/truncated-moment functions, and initial-law
constructors for the discrete cutoff allocation experiments.

A law (probability measure on [0, ∞)) is represented as:
  - q0        : float  — mass at 0 (default / absorbed mass)
  - grid      : ndarray(N,) — fixed fine grid of positive x-values [x_1,...,x_N]
  - density   : ndarray(N,) — grid weights so that density[i] ≈ μ((x_i-dx/2, x_i+dx/2))
                              (i.e. density integrates to the continuous-part total mass)
  - atoms     : list[(x, w)] — point masses at positive x values NOT on the grid
                              (used for atoms created by the capping operator)

Total mass = q0 + sum(density) + sum(w for _, w in atoms)
Mean       = sum(x*density) + sum(x*w for x, w in atoms)   (q0 contributes 0)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Callable
from scipy.stats import norm


@dataclass
class DiscreteLaw:
    """Probability measure on [0, ∞) with a grid-based representation.

    Parameters
    ----------
    q0 : float
        Probability mass at the absorbing boundary 0.
    grid : ndarray
        Uniform grid of strictly positive x-values (shared across all laws
        in one experiment; do NOT mutate).
    density : ndarray
        Non-negative weights at each grid point. density[i] represents the
        measure of the interval (grid[i]-dx/2, grid[i]+dx/2).
    atoms : list of (float, float)
        Extra point masses (x, w) with x > 0 created by the capping operator.
    """

    q0: float
    grid: np.ndarray
    density: np.ndarray
    atoms: List[Tuple[float, float]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Basic properties                                                     #
    # ------------------------------------------------------------------ #

    @property
    def total_mass(self) -> float:
        """Sum of all probability mass; should equal 1.0."""
        return float(self.q0 + np.sum(self.density) + sum(w for _, w in self.atoms))

    @property
    def mean(self) -> float:
        """Mean of the law (contribution of atom at 0 is zero)."""
        m = float(np.dot(self.grid, self.density))
        m += sum(x * w for x, w in self.atoms)
        return m

    @property
    def default_mass(self) -> float:
        """q_μ = μ({0})."""
        return float(self.q0)

    # ------------------------------------------------------------------ #
    # Call and truncated-moment functions                                 #
    # ------------------------------------------------------------------ #

    def call(self, a: float) -> float:
        """C_μ(a) = ∫ (x - a)⁺ μ(dx).

        For a ≥ 0 the atom at 0 contributes nothing (0 - a)⁺ = 0.
        """
        val = float(np.dot(np.maximum(self.grid - a, 0.0), self.density))
        val += sum(max(x - a, 0.0) * w for x, w in self.atoms)
        return val

    def call_vec(self, a_arr: np.ndarray) -> np.ndarray:
        """Vectorised call function over an array of strikes."""
        # grid contrib: shape (len(a_arr), N)
        diff = np.maximum(self.grid[np.newaxis, :] - a_arr[:, np.newaxis], 0.0)
        val = diff @ self.density
        for x, w in self.atoms:
            val += np.maximum(x - a_arr, 0.0) * w
        return val

    def truncated_moment(self, a: float) -> float:
        """D_μ(a) = ∫ (x ∧ a) μ(dx) = mean - C_μ(a).

        Note: q0 contributes 0 ∧ a = 0 for a ≥ 0, so D = mean - C.
        """
        return self.mean - self.call(a)

    def truncated_moment_vec(self, a_arr: np.ndarray) -> np.ndarray:
        """Vectorised truncated-moment function."""
        return self.mean - self.call_vec(a_arr)

    def survival_probability(self) -> float:
        """P(X > 0) = 1 - q0."""
        return 1.0 - self.q0

    # ------------------------------------------------------------------ #
    # Positive-mass summary                                               #
    # ------------------------------------------------------------------ #

    def all_positive_points_and_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return all positive support points and their weights as arrays."""
        x_list = list(self.grid)
        w_list = list(self.density)
        for x, w in self.atoms:
            x_list.append(x)
            w_list.append(w)
        x_arr = np.array(x_list)
        w_arr = np.array(w_list)
        # Sort by x for consistency
        idx = np.argsort(x_arr)
        return x_arr[idx], w_arr[idx]

    # ------------------------------------------------------------------ #
    # Validation                                                          #
    # ------------------------------------------------------------------ #

    def validate(self, tol: float = 1e-6) -> None:
        """Assert basic sanity conditions."""
        assert self.q0 >= -tol, f"Negative q0: {self.q0}"
        assert np.all(self.density >= -tol), "Negative density weights"
        assert np.all(self.grid > 0), "Grid must be strictly positive"
        tm = self.total_mass
        assert abs(tm - 1.0) < 1e-4, f"Total mass not 1: {tm:.8f}"


# ------------------------------------------------------------------ #
# Factory functions for standard initial laws                        #
# ------------------------------------------------------------------ #

def _make_grid(x_max: float, N: int) -> Tuple[np.ndarray, float]:
    """Create a uniform cell-centred grid on (0, x_max] with N points."""
    dx = x_max / N
    grid = (np.arange(N) + 0.5) * dx  # midpoints: dx/2, 3dx/2, ...
    return grid, dx


def make_uniform_law(
    a: float,
    b: float,
    x_max: float = 4.0,
    N: int = 2000,
) -> DiscreteLaw:
    """μ₀ = Uniform[a, b].

    Parameters
    ----------
    a, b : float
        Support endpoints; 0 < a < b.
    x_max, N : float, int
        Grid parameters.

    Returns
    -------
    DiscreteLaw
    """
    assert 0 < a < b, f"Need 0 < a < b, got a={a}, b={b}"
    grid, dx = _make_grid(x_max, N)
    density_vals = np.where((grid >= a) & (grid <= b), dx / (b - a), 0.0)
    # Normalise to account for discretisation error
    total = float(np.sum(density_vals))
    density_vals = density_vals / total  # exact total mass = 1
    return DiscreteLaw(q0=0.0, grid=grid, density=density_vals, atoms=[])


def make_mixture_uniform_law(
    a1: float,
    b1: float,
    a2: float,
    b2: float,
    p1: float = 0.5,
    x_max: float = 4.0,
    N: int = 2000,
) -> DiscreteLaw:
    """μ₀ = p1·Uniform[a1, b1] + (1-p1)·Uniform[a2, b2].

    Parameters
    ----------
    a1, b1, a2, b2 : float
        Support endpoints of the two uniform components.
    p1 : float
        Mixture weight for the first component (0 < p1 < 1).
    """
    assert 0 < a1 < b1 and 0 < a2 < b2
    assert 0 < p1 < 1
    grid, dx = _make_grid(x_max, N)
    d1 = np.where((grid >= a1) & (grid <= b1), dx / (b1 - a1), 0.0)
    d2 = np.where((grid >= a2) & (grid <= b2), dx / (b2 - a2), 0.0)
    total1 = float(np.sum(d1))
    total2 = float(np.sum(d2))
    # Normalise each component then mix
    if total1 > 0:
        d1 = d1 / total1
    if total2 > 0:
        d2 = d2 / total2
    density_vals = p1 * d1 + (1 - p1) * d2
    return DiscreteLaw(q0=0.0, grid=grid, density=density_vals, atoms=[])


def make_atomic_law(
    locations: List[float],
    weights: List[float],
    x_max: float = 4.0,
    N: int = 2000,
) -> DiscreteLaw:
    """μ₀ = Σ_i w_i δ_{x_i} with all x_i > 0.

    The atoms are stored in the `atoms` list; the grid density is zero.
    """
    assert all(x > 0 for x in locations), "All atom locations must be positive"
    total = sum(weights)
    w_norm = [w / total for w in weights]
    grid, _ = _make_grid(x_max, N)
    atoms = list(zip(locations, w_norm))
    return DiscreteLaw(q0=0.0, grid=grid, density=np.zeros(N), atoms=atoms)


# ------------------------------------------------------------------ #
# Standard test suite laws from the paper                            #
# ------------------------------------------------------------------ #

def standard_test_cases(
    x_max: float = 4.0,
    N: int = 2000,
) -> List[dict]:
    """Return the three standard initial laws from the paper.

    Returns a list of dicts with keys: 'law', 'name', 'm0', 'a_inf', 'b_sup',
    'alpha_values', 'label'.
    """
    cases = [
        {
            "name": "mu1_uniform",
            "label": "μ₀¹ = Unif[0.5, 1.5]",
            "law": make_uniform_law(0.5, 1.5, x_max=x_max, N=N),
            "m0": 1.0,
            "a_inf": 0.5,  # inf supp(μ₀)
            "b_sup": 1.5,  # sup supp(μ₀)
            "alpha_values": [0.1, 0.3, 0.5],
        },
        {
            "name": "mu2_mixture_uniform",
            "label": "μ₀² = 0.5·Unif[0.2,0.4] + 0.5·Unif[1.2,1.4]",
            "law": make_mixture_uniform_law(0.2, 0.4, 1.2, 1.4, p1=0.5, x_max=x_max, N=N),
            "m0": 0.8,
            "a_inf": 0.2,
            "b_sup": 1.4,
            "alpha_values": [0.1, 0.3, 0.5],
        },
        {
            "name": "mu3_atomic",
            "label": "μ₀³ = 0.5·δ₀.₂₅ + 0.5·δ₁.₂₅",
            "law": make_atomic_law([0.25, 1.25], [0.5, 0.5], x_max=x_max, N=N),
            "m0": 0.75,
            "a_inf": 0.25,
            "b_sup": 1.25,
            "alpha_values": [0.1, 0.3, 0.5],
        },
    ]
    return cases


# ------------------------------------------------------------------ #
# Uncontrolled default curve q_μ^h(t)                               #
# ------------------------------------------------------------------ #

def uncontrolled_default_curve(
    law: DiscreteLaw,
    t_grid: np.ndarray,
) -> np.ndarray:
    """Compute q_μ^h(t) = ∫ 2 Φ̄(x/√t) μ(dx) + q0 for each t in t_grid.

    This is the total absorbed mass of S_t μ, i.e. q_{S_t μ}.
    Used for the stability metric d_T^#.

    Parameters
    ----------
    law : DiscreteLaw
        Initial law (must have q0 = 0 for paper's setting, but general).
    t_grid : ndarray
        Array of positive time values.

    Returns
    -------
    ndarray of shape (len(t_grid),)
    """
    q_curve = np.zeros(len(t_grid))

    # Contribution of atom at 0: it stays at 0 forever, contributes q0
    q_curve += law.q0

    # Contribution of grid density
    if np.any(law.density > 0):
        x_pos = law.grid  # (N,)
        w_pos = law.density  # (N,)
        for k, t in enumerate(t_grid):
            sqrt_t = np.sqrt(t)
            p_absorb = 2.0 * norm.cdf(-x_pos / sqrt_t)  # 2Φ̄(x/√t)
            q_curve[k] += float(np.dot(w_pos, p_absorb))

    # Contribution of point atoms
    for x_a, w_a in law.atoms:
        for k, t in enumerate(t_grid):
            p_absorb = 2.0 * norm.cdf(-x_a / np.sqrt(t))
            q_curve[k] += w_a * p_absorb

    return q_curve


# ------------------------------------------------------------------ #
# Stability metric d_T^#                                             #
# ------------------------------------------------------------------ #

def call_function_array(law: DiscreteLaw, a_arr: np.ndarray) -> np.ndarray:
    """C_μ(a) for an array of strikes."""
    return law.call_vec(a_arr)


def d_T_sharp(
    law1: DiscreteLaw,
    law2: DiscreteLaw,
    a_grid: np.ndarray,
    t_grid: np.ndarray,
) -> float:
    """Compute d_T^#(μ, ν) = ||C_μ - C_ν||_{L∞(R+)} + sup_{0<t≤T} |q_μ^h(t) - q_ν^h(t)|.

    Parameters
    ----------
    law1, law2 : DiscreteLaw
        Two initial laws to compare.
    a_grid : ndarray
        Strike grid for C∞ comparison.
    t_grid : ndarray
        Positive time grid for default curve comparison.
    """
    c1 = call_function_array(law1, a_grid)
    c2 = call_function_array(law2, a_grid)
    call_diff = float(np.max(np.abs(c1 - c2)))

    q1 = uncontrolled_default_curve(law1, t_grid)
    q2 = uncontrolled_default_curve(law2, t_grid)
    q_diff = float(np.max(np.abs(q1 - q2)))

    return call_diff + q_diff
