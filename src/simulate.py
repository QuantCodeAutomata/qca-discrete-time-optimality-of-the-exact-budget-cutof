"""
simulate.py
-----------
Independent pathwise discrete reflection construction for validation.

Given a sequence of thresholds {κ_j^Δ} from the law-level recursion,
simulate:
  K_t^Δ = max_{j: jΔ ≤ t} (X_0 + B_{jΔ} - κ_j^Δ)^+
  X̂_t^Δ = X_0 + B_t - K_t^Δ
  τ^Δ = inf{t: X̂_t^Δ ≤ 0}
  L_t^Δ = K_{t ∧ τ^Δ}^Δ
  X_t^Δ = X̂_t^Δ · 1_{t < τ^Δ}

Then compare the empirical distribution at each t_n = nΔ to the
law-level DiscreteLaw objects from the cutoff flow.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional
from .laws import DiscreteLaw


def sample_from_law(
    law: DiscreteLaw,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n_samples values from a DiscreteLaw.

    Parameters
    ----------
    law : DiscreteLaw
    n_samples : int
    rng : numpy Generator

    Returns
    -------
    ndarray of shape (n_samples,) with values in [0, ∞).
    """
    # Collect all mass: atom at 0, grid, extra atoms
    all_x = np.concatenate([[0.0], law.grid, [x for x, _ in law.atoms]])
    all_w = np.concatenate(
        [[law.q0], law.density, [w for _, w in law.atoms]]
    )
    total = float(np.sum(all_w))
    all_w = all_w / total  # normalise

    # Sample indices
    idx = rng.choice(len(all_x), size=n_samples, p=all_w)
    samples = all_x[idx]

    # Add small jitter within grid cells for smooth laws (not atoms)
    # Grid contribution: add uniform jitter ±dx/2
    grid_start = 1  # index 0 is the atom at 0
    grid_end = 1 + len(law.grid)
    dx = law.grid[1] - law.grid[0] if len(law.grid) > 1 else 0.0
    is_grid = (idx >= grid_start) & (idx < grid_end)
    jitter = rng.uniform(-dx / 2, dx / 2, size=n_samples)
    samples[is_grid] += jitter[is_grid]
    samples = np.maximum(samples, 0.0)

    return samples


def pathwise_discrete_reflection(
    kappas: List[float],
    Delta: float,
    mu0: DiscreteLaw,
    n_paths: int = 10000,
    seed: int = 42,
) -> dict:
    """Simulate the discrete reflected process using the law-level thresholds.

    Parameters
    ----------
    kappas : list of float
        Thresholds κ_1^Δ, ..., κ_M^Δ from run_cutoff_flow.
    Delta : float
        Time step.
    mu0 : DiscreteLaw
        Initial law (used to sample X_0).
    n_paths : int
        Number of Monte Carlo paths.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
      'X_paths'         : ndarray (n_paths, M+1) — X_t^Δ at each step
      'L_paths'         : ndarray (n_paths, M+1) — cumulative loss L_t^Δ
      'tau_idx'         : ndarray (n_paths,) int — step index of absorption (M+1 if never)
      'q_empirical'     : ndarray (M+1,) — empirical P(τ ≤ t_n)
      'h_empirical'     : ndarray (M+1,) — empirical E[L_{t_n}]
      'm_empirical'     : ndarray (M+1,) — empirical E[X_{t_n}]
      'balance_empirical': ndarray (M+1,) — |h_n - α·q_n| (not computable here; computed outside)
      't_grid'          : ndarray (M+1,)
    """
    M = len(kappas)
    rng = np.random.default_rng(seed)

    # Sample initial states X_0
    X0 = sample_from_law(mu0, n_paths, rng)

    t_grid = np.arange(M + 1) * Delta

    # Brownian increments: B_{jΔ} - B_{(j-1)Δ} ~ N(0, Δ)
    dB = rng.normal(0.0, np.sqrt(Delta), size=(n_paths, M))  # (n_paths, M)
    B = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dB, axis=1)], axis=1)  # (n_paths, M+1)

    # Reflected process
    X = np.zeros((n_paths, M + 1))
    L = np.zeros((n_paths, M + 1))  # cumulative loss K_{t∧τ}
    tau_idx = np.full(n_paths, M + 1, dtype=int)  # absorption step (M+1 = never absorbed)

    X[:, 0] = X0
    # K_t^Δ = max_{j≥1, jΔ≤t} (X_0 + B_{jΔ} - κ_j)^+, initialise at 0
    K_running = np.zeros(n_paths)  # running maximum of cumulative loss

    alive = np.ones(n_paths, dtype=bool)

    for j in range(1, M + 1):
        kappa_j = kappas[j - 1]  # κ_j (may be inf if no allocation)

        # Increment to running maximum: (X_0 + B_{jΔ} - κ_j)^+
        overshoot = X0 + B[:, j] - kappa_j
        new_K = np.maximum(overshoot, 0.0)
        K_running = np.maximum(K_running, new_K)

        # Candidate X̂_{t_j} = X_0 + B_{jΔ} - K_running
        X_hat = X0 + B[:, j] - K_running

        # Absorption: X̂ ≤ 0 for alive paths
        newly_absorbed = alive & (X_hat <= 0)
        tau_idx[newly_absorbed] = j
        alive[newly_absorbed] = False

        # L at this step
        # For absorbed paths, L = K at absorption (constant thereafter)
        L[:, j] = np.where(alive, K_running, L[:, j - 1])
        # Correction: newly absorbed paths get their K at absorption
        L[newly_absorbed, j] = K_running[newly_absorbed]
        # Previously absorbed paths: L stays constant
        prev_absorbed = ~alive & (tau_idx < j)
        L[prev_absorbed, j] = L[prev_absorbed, j - 1]

        # X_t = X̂_t · 1_{t < τ}
        X[:, j] = np.where(alive, np.maximum(X_hat, 0.0), 0.0)

    # Empirical statistics at each step
    q_empirical = np.zeros(M + 1)
    h_empirical = np.zeros(M + 1)
    m_empirical = np.zeros(M + 1)

    for j in range(M + 1):
        q_empirical[j] = float(np.mean(tau_idx <= j))
        h_empirical[j] = float(np.mean(L[:, j]))
        m_empirical[j] = float(np.mean(X[:, j]))

    return {
        "X_paths": X,
        "L_paths": L,
        "tau_idx": tau_idx,
        "q_empirical": q_empirical,
        "h_empirical": h_empirical,
        "m_empirical": m_empirical,
        "t_grid": t_grid,
    }


def compare_law_vs_pathwise(
    flow: dict,
    pathwise: dict,
    alpha: float,
) -> dict:
    """Compare the law-level flow and pathwise simulation statistics.

    Parameters
    ----------
    flow : dict
        Output of run_cutoff_flow.
    pathwise : dict
        Output of pathwise_discrete_reflection.
    alpha : float

    Returns
    -------
    dict with discrepancy metrics.
    """
    q_law = flow["q_path"]
    q_mc = pathwise["q_empirical"]
    m_law = flow["m_path"]
    m_mc = pathwise["m_empirical"]
    h_law = flow["h_path"]
    h_mc = pathwise["h_empirical"]

    n = min(len(q_law), len(q_mc))

    q_diff = np.abs(q_law[:n] - q_mc[:n])
    m_diff = np.abs(m_law[:n] - m_mc[:n])
    h_diff = np.abs(h_law[:n] - h_mc[:n])

    # Pathwise balance check: h_mc ≈ alpha * q_mc
    balance_mc = np.abs(h_mc[:n] - alpha * q_mc[:n])

    return {
        "max_q_diff": float(np.max(q_diff)),
        "max_m_diff": float(np.max(m_diff)),
        "max_h_diff": float(np.max(h_diff)),
        "max_balance_mc": float(np.max(balance_mc)),
        "q_diff": q_diff,
        "m_diff": m_diff,
        "h_diff": h_diff,
    }
