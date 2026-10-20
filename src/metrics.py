"""
metrics.py
----------
Payoff evaluation, D-curve computation, and conservation-law invariant checks.

Admissible payoffs tested (non-decreasing concave on [0, ∞)):
  f1(x) = 1_{(0,∞)}(x)   — survival probability
  f2(x) = x               — linear (mean reserve)
  f3(x) = √x              — square-root
  f4(x) = log(1 + x)      — logarithmic
  f5(x) = x ∧ 1           — min(x, 1) = capped linear

Invariant conditions:
  Exact balance:     E[L_{t_n}] = α · P(τ ≤ t_n)   ⟺   h_n = α · q_n
  Resource invariance: m_{t_n} + α · q_{t_n} = m_0   ⟺   m_n + h_n = m_0
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Dict
from .laws import DiscreteLaw


# ------------------------------------------------------------------ #
# Payoff functions                                                    #
# ------------------------------------------------------------------ #

def _f_survival(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def _f_linear(x: np.ndarray) -> np.ndarray:
    return x


def _f_sqrt(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(x, 0.0))


def _f_log(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0))


def _f_capped(x: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, 0.0), 1.0)


PAYOFFS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "survival": _f_survival,
    "linear": _f_linear,
    "sqrt": _f_sqrt,
    "log1p": _f_log,
    "capped_1": _f_capped,
}

PAYOFF_NAMES = list(PAYOFFS.keys())


def payoff_value(law: DiscreteLaw, payoff_name: str) -> float:
    """Compute E_μ[f(X)] = ∫ f(x) μ(dx) for a named payoff.

    The atom at 0 contributes f(0) = 0 for survival, sqrt, log, capped;
    f(0) = 0 for linear. So in all cases f(0) = 0 and q0 contributes 0.

    Parameters
    ----------
    law : DiscreteLaw
        The measure μ_T.
    payoff_name : str
        One of PAYOFF_NAMES.

    Returns
    -------
    float
        E[f(X_T)].
    """
    if payoff_name not in PAYOFFS:
        raise ValueError(f"Unknown payoff '{payoff_name}'. Choose from {PAYOFF_NAMES}.")
    f = PAYOFFS[payoff_name]

    # Grid contribution (atom at 0 contributes 0 for all payoffs)
    val = float(np.dot(f(law.grid), law.density))
    # Extra atoms
    for x_a, w_a in law.atoms:
        val += float(f(np.array([x_a]))[0]) * w_a
    return val


def all_payoffs(law: DiscreteLaw) -> Dict[str, float]:
    """Compute E[f(X)] for all standard payoffs.

    Returns
    -------
    dict mapping payoff_name -> E[f(X)]
    """
    return {name: payoff_value(law, name) for name in PAYOFF_NAMES}


# ------------------------------------------------------------------ #
# D-curve (truncated moment curve)                                   #
# ------------------------------------------------------------------ #

def d_curve(law: DiscreteLaw, a_grid: np.ndarray) -> np.ndarray:
    """Compute D_μ(a) = ∫ (x ∧ a) μ(dx) = mean - C_μ(a) for each a in a_grid.

    Parameters
    ----------
    law : DiscreteLaw
    a_grid : ndarray
        Array of strike values a ≥ 0.

    Returns
    -------
    ndarray of shape (len(a_grid),)
    """
    return law.truncated_moment_vec(a_grid)


def dominance_check(
    law_cutoff: DiscreteLaw,
    law_bench: DiscreteLaw,
    a_grid: np.ndarray,
) -> Dict[str, float | bool]:
    """Check D_{T_c μ}(a) ≥ D_ν(a) for all a in a_grid.

    Parameters
    ----------
    law_cutoff : DiscreteLaw
        Capped law (should dominate).
    law_bench : DiscreteLaw
        Benchmark law.
    a_grid : ndarray

    Returns
    -------
    dict with keys:
      'min_diff'   : min(D_cutoff - D_bench) over a_grid
      'max_diff'   : max(D_cutoff - D_bench)
      'dominates'  : bool, True if min_diff ≥ -tol
      'D_cutoff'   : ndarray
      'D_bench'    : ndarray
    """
    D_c = d_curve(law_cutoff, a_grid)
    D_b = d_curve(law_bench, a_grid)
    diff = D_c - D_b
    return {
        "min_diff": float(np.min(diff)),
        "max_diff": float(np.max(diff)),
        "dominates": bool(np.min(diff) >= -1e-6),
        "D_cutoff": D_c,
        "D_bench": D_b,
    }


# ------------------------------------------------------------------ #
# Conservation-law invariant checks                                  #
# ------------------------------------------------------------------ #

def compute_invariants(
    flow: dict,
    alpha: float,
    m0: float,
    tol: float = 1e-6,
) -> Dict[str, float | bool | np.ndarray]:
    """Check exact balance and resource invariance along a discrete law flow.

    Exact balance:     h_n = α · q_n
    Resource invariance: m_n + α · q_n = m_0

    Parameters
    ----------
    flow : dict
        Output of run_cutoff_flow: contains 'q_path', 'h_path', 'm_path'.
    alpha : float
    m0 : float
        Initial mean reserve.
    tol : float
        Tolerance for checks.

    Returns
    -------
    dict with invariant check results.
    """
    q = flow["q_path"]
    h = flow["h_path"]
    m = flow["m_path"]

    # Exact balance: h_n = α · q_n
    balance_residual = np.abs(h - alpha * q)

    # Resource invariance: m_n + α · q_n = m0
    invariance_residual = np.abs(m + alpha * q - m0)

    return {
        "balance_residual": balance_residual,
        "max_balance_residual": float(np.max(balance_residual)),
        "invariance_residual": invariance_residual,
        "max_invariance_residual": float(np.max(invariance_residual)),
        "exact_balance_ok": bool(np.max(balance_residual) < tol),
        "resource_invariance_ok": bool(np.max(invariance_residual) < tol),
    }


# ------------------------------------------------------------------ #
# Terminal objective comparison                                       #
# ------------------------------------------------------------------ #

def compare_terminal_objectives(
    law_cutoff: DiscreteLaw,
    bench_laws: Dict[str, DiscreteLaw | None],
) -> Dict[str, Dict[str, float]]:
    """Compare terminal payoff values across cutoff and benchmark policies.

    Parameters
    ----------
    law_cutoff : DiscreteLaw
        Terminal law under the cutoff policy.
    bench_laws : dict
        Mapping name -> terminal law for benchmark policies.

    Returns
    -------
    dict: {policy_name: {payoff_name: value}} for all policies and payoffs.
    """
    results = {"cutoff": all_payoffs(law_cutoff)}
    for name, law in bench_laws.items():
        if law is not None:
            results[name] = all_payoffs(law)
        else:
            results[name] = {p: np.nan for p in PAYOFF_NAMES}
    return results


# ------------------------------------------------------------------ #
# Flow path summaries                                                 #
# ------------------------------------------------------------------ #

def flow_path_summary(flow: dict) -> Dict[str, np.ndarray]:
    """Extract q, h, m paths and compute terminal survival probability."""
    return {
        "t_grid": flow["t_grid"],
        "q_path": flow["q_path"],
        "h_path": flow["h_path"],
        "m_path": flow["m_path"],
        "survival_prob": 1.0 - flow["q_path"],
        "kappas": np.array(
            [k if not np.isinf(k) else np.nan for k in flow["kappas"]]
        ),
    }
