"""
exp2_continuous_limit.py
------------------------
Experiment 2: Continuous-Time Limit and Reflected Free-Boundary Characterization.

Takes the vanishing-mesh limit of the discrete cutoff scheme:
  Δ ↓ 0  →  discrete optimal values V_f^Δ converge.
  Threshold paths κ_n^Δ converge on (0, T] to a continuous boundary Λ_t.
  Limiting control is a Skorokhod reflection stopped at 0.
  Validates: h_t = α·P(τ≤t), supp(μ_t) = [0, Λ_t], boundary trace Λ_t → a+b.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d

from src.laws import standard_test_cases, DiscreteLaw
from src.cutoff import run_cutoff_flow
from src.metrics import all_payoffs, PAYOFF_NAMES, compute_invariants

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")


# ================================================================== #
# Mesh-refinement convergence study                                   #
# ================================================================== #

def mesh_refinement_study(
    case: dict,
    alpha: float,
    T: float = 1.0,
    Delta_list: list | None = None,
) -> dict:
    """Run the discrete cutoff scheme for a sequence of mesh sizes.

    Parameters
    ----------
    case : dict
        Standard test case dict.
    alpha : float
    T : float
    Delta_list : list of float
        Mesh sizes Δ (must divide T).

    Returns
    -------
    dict with flows, boundary proxies, and payoff convergence data.
    """
    if Delta_list is None:
        # Use 2^{-4} to 2^{-8} (finer meshes are slow)
        Delta_list = [2**(-k) for k in range(4, 9)]

    mu0: DiscreteLaw = case["law"]
    flows = {}
    boundaries = {}  # Λ^Δ as piecewise-constant function on [0, T]
    terminal_payoffs = {}  # {delta_val: {payoff_name: value}}

    for Delta in Delta_list:
        M = int(round(T / Delta))
        if abs(M - T / Delta) > 1e-8:
            print(f"  Skip Δ={Delta:.4f} (T/Δ not integer)")
            continue

        print(f"  Running Δ={Delta:.5f} (M={M})", end=" ... ", flush=True)
        try:
            flow = run_cutoff_flow(mu0, alpha, T, Delta)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        flows[Delta] = flow
        kappas = flow["kappas"]
        t_alloc = flow["t_grid"][1:]  # t_1, ..., t_M

        # Piecewise-constant boundary proxy (left-continuous interpolation)
        # Λ^Δ(t) = κ_n for t ∈ [t_{n-1}, t_n)
        kappas_finite = [k if not np.isinf(k) else np.nan for k in kappas]
        boundaries[Delta] = {"t": t_alloc, "kappa": np.array(kappas_finite)}

        terminal_payoffs[Delta] = all_payoffs(flow["laws"][-1])
        print(f"done. Terminal V_linear={terminal_payoffs[Delta]['linear']:.5f}")

    return {
        "flows": flows,
        "boundaries": boundaries,
        "terminal_payoffs": terminal_payoffs,
        "Delta_list": [d for d in Delta_list if d in flows],
    }


def payoff_convergence_table(terminal_payoffs: dict, Delta_list: list) -> pd.DataFrame:
    """Build a payoff convergence table across mesh sizes."""
    rows = []
    for Delta in Delta_list:
        if Delta not in terminal_payoffs:
            continue
        row = {"Delta": Delta}
        row.update(terminal_payoffs[Delta])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Delta")


def boundary_convergence_on_interval(
    boundaries: dict,
    Delta_list: list,
    t_eval: np.ndarray,
) -> dict:
    """Evaluate all Λ^Δ boundaries on a common t_eval grid.

    Returns a dict Delta -> ndarray of boundary values at t_eval.
    """
    result = {}
    for Delta in Delta_list:
        if Delta not in boundaries:
            continue
        t_bnd = boundaries[Delta]["t"]
        kap = boundaries[Delta]["kappa"]
        # Left-continuous step function: Λ^Δ(t) = kappa[n] for t ∈ (t_{n-1}, t_n]
        # Use numpy searchsorted
        boundary_interp = np.full(len(t_eval), np.nan)
        for k, tv in enumerate(t_eval):
            idx = np.searchsorted(t_bnd, tv, side="right") - 1
            if idx >= 0 and idx < len(kap):
                boundary_interp[k] = kap[idx]
        result[Delta] = boundary_interp
    return result


# ================================================================== #
# Augmented-call boundary recovery                                    #
# ================================================================== #

def augmented_call_boundary(
    law: DiscreteLaw,
    h_t: float,
    x_grid: np.ndarray,
) -> float:
    """Recover Λ_t = inf{x ≥ 0: H(t,x) = h_t} where H(t,x) = C(t,x) + h_t.

    C(t,x) = ∫ (y-x)^+ μ_t(dy) is the call function of μ_t at strike x.
    H(t,x) = h_t means C(t,x) = 0, which happens at x = sup supp(μ_t).

    Actually: H(t,x) = C_{μ_t}(x) + h_t.
    The condition H(t,x) = h_t  ⟺  C_{μ_t}(x) = 0  ⟺  x ≥ sup supp(μ_t).

    So Λ_t = inf{x: C_{μ_t}(x) = 0} = sup supp^+(μ_t)  (supremum of the positive support).

    This recovers the boundary as the supremum of the post-allocation density support.

    Parameters
    ----------
    law : DiscreteLaw
        Post-allocation law μ_t^Δ at some step.
    h_t : float
        Cumulative allocated loss h_t = α·q_t.
    x_grid : ndarray
        Grid of strike values to search.

    Returns
    -------
    float
        Recovered boundary Λ_t.
    """
    # C_{μ_t}(x) decreasing; find last x where C > 0 (≈ machine eps)
    C_vals = law.call_vec(x_grid)
    # Λ_t is the smallest x such that C_{μ_t}(x) ≈ 0
    tol = 1e-6
    positive_idx = np.where(C_vals > tol)[0]
    if len(positive_idx) == 0:
        return 0.0
    return float(x_grid[positive_idx[-1]])


def compute_augmented_call_boundaries(flow: dict) -> np.ndarray:
    """Compute the augmented-call boundary at each step of the discrete flow."""
    laws = flow["laws"]
    h_path = flow["h_path"]
    grid = laws[0].grid
    # Use a coarse x_grid for speed
    x_search = grid[::5]  # every 5th grid point
    kappas_aug = np.zeros(len(laws))
    for n, (law_n, h_n) in enumerate(zip(laws, h_path)):
        kappas_aug[n] = augmented_call_boundary(law_n, h_n, x_search)
    return kappas_aug


# ================================================================== #
# PDE residual checks (on surviving density)                        #
# ================================================================== #

def estimate_pde_residuals(
    flow: dict,
    tau_cutoff: float = 0.1,
) -> dict:
    """Estimate the heat equation residual ∂_t u ≈ 0.5 ∂_{xx} u.

    Uses finite differences on the grid density across consecutive steps.
    Only valid for t > tau_cutoff (away from the singular initial time).

    Parameters
    ----------
    flow : dict
    tau_cutoff : float
        Minimum time to start evaluating residuals.

    Returns
    -------
    dict with 'max_pde_residual', 'mean_pde_residual', 't_checked'.
    """
    laws = flow["laws"]
    t_grid = flow["t_grid"]
    Delta = float(flow["Delta"])
    grid = laws[0].grid
    dx = grid[1] - grid[0]

    residuals = []
    t_checked = []

    for n in range(1, len(laws) - 1):
        if t_grid[n] < tau_cutoff:
            continue

        # u(t_n, x) ≈ density at step n (already the post-allocation density)
        u_prev = laws[n - 1].density  # t_{n-1}
        u_curr = laws[n].density       # t_n
        u_next = laws[n + 1].density   # t_{n+1}

        # Time derivative (backward finite difference)
        dt_u = (u_curr - u_prev) / Delta

        # Spatial second derivative (centred, interior points only)
        d2x_u = np.zeros_like(u_curr)
        d2x_u[1:-1] = (u_curr[2:] - 2.0 * u_curr[1:-1] + u_curr[:-2]) / (dx ** 2)

        pde_res = dt_u - 0.5 * d2x_u
        # Only check on positive-density interior points
        active = u_curr[1:-1] > 1e-8
        if np.any(active):
            res = float(np.mean(np.abs(pde_res[1:-1][active])))
            residuals.append(res)
            t_checked.append(float(t_grid[n]))

    if not residuals:
        return {"max_pde_residual": np.nan, "mean_pde_residual": np.nan, "t_checked": []}

    return {
        "max_pde_residual": float(max(residuals)),
        "mean_pde_residual": float(np.mean(residuals)),
        "t_checked": t_checked,
    }


# ================================================================== #
# Plotting                                                            #
# ================================================================== #

def plot_boundary_convergence(
    bnd_on_grid: dict,
    t_eval: np.ndarray,
    case: dict,
    alpha: float,
    a_inf: float,
    b_sup: float,
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(bnd_on_grid)))
    for (Delta, kap_vals), color in zip(sorted(bnd_on_grid.items()), colors):
        ax.plot(t_eval, kap_vals, "-", color=color, lw=1.2, label=f"Δ={Delta:.4f}")
    ax.axhline(a_inf + b_sup, color="red", ls="--", lw=1.5, label=f"a+b = {a_inf+b_sup:.2f}")
    ax.set_xlabel("t"); ax.set_ylabel("Λ^Δ(t)")
    ax.set_title(f"Boundary convergence: {case['label']}, α={alpha}")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_payoff_convergence(conv_df: pd.DataFrame, case: dict, alpha: float, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for pname in PAYOFF_NAMES:
        if pname in conv_df.columns:
            ax.plot(1.0 / conv_df["Delta"], conv_df[pname], "o-", lw=1.5, label=pname)
    ax.set_xlabel("M = T/Δ (steps)"); ax.set_ylabel("V_f^Δ")
    ax.set_title(f"Payoff convergence: {case['label']}, α={alpha}")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_support_evolution(flows: dict, case: dict, alpha: float, save_path: str) -> None:
    """Plot the support supremum (augmented-call boundary) over time."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors_list = plt.cm.Greens(np.linspace(0.4, 1.0, len(flows)))
    for (Delta, flow), color in zip(sorted(flows.items()), colors_list):
        kap_aug = compute_augmented_call_boundaries(flow)
        t_plot = flow["t_grid"]
        ax.plot(t_plot, kap_aug, "-", color=color, lw=1, label=f"Δ={Delta:.4f}")
    ax.axhline(case["a_inf"] + case["b_sup"], color="r", ls="--",
               lw=1.5, label=f"a+b={case['a_inf']+case['b_sup']:.2f}")
    ax.set_xlabel("t"); ax.set_ylabel("Λ_t (aug-call recovery)")
    ax.set_title(f"Support boundary: {case['label']}, α={alpha}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ================================================================== #
# Main orchestration                                                  #
# ================================================================== #

def run_experiment_2(
    T: float = 1.0,
    Delta_list: list | None = None,
    tau_cutoff: float = 0.1,
) -> dict:
    """Run Experiment 2: continuous-time limit and free-boundary checks."""
    if Delta_list is None:
        Delta_list = [2**(-k) for k in range(4, 9)]

    cases = standard_test_cases(x_max=4.0, N=1000)
    summary_records = []

    for case in cases:
        m0 = case["m0"]
        cname = case["name"]

        for alpha in case["alpha_values"]:
            if m0 <= alpha:
                continue

            print(f"\n=== Exp2: {cname}, α={alpha} ===")

            # 1. Mesh refinement study
            study = mesh_refinement_study(case, alpha, T=T, Delta_list=Delta_list)
            if not study["Delta_list"]:
                print("  No valid Δ values; skip.")
                continue

            valid_deltas = study["Delta_list"]

            # 2. Payoff convergence table
            conv_df = payoff_convergence_table(study["terminal_payoffs"], valid_deltas)
            csv_path = RESULTS_DIR / f"exp2_payoff_conv_{cname}_a{alpha}.csv"
            conv_df.to_csv(csv_path, index=False)

            # 3. Boundary convergence on (tau_cutoff, T]
            t_eval = np.linspace(tau_cutoff, T, 80)
            bnd_on_grid = boundary_convergence_on_interval(study["boundaries"], valid_deltas, t_eval)

            # Boundary convergence: check max pairwise diff at finest vs next mesh
            delta_sorted = sorted(valid_deltas)
            bnd_diff = {}
            if len(delta_sorted) >= 2:
                finest = delta_sorted[0]
                next_finest = delta_sorted[1]
                diff_arr = np.abs(
                    np.nan_to_num(bnd_on_grid[finest]) - np.nan_to_num(bnd_on_grid[next_finest])
                )
                bnd_diff["max_bnd_diff_finest_vs_next"] = float(np.nanmax(diff_arr))
            else:
                bnd_diff["max_bnd_diff_finest_vs_next"] = np.nan

            # 4. Invariant checks for finest mesh
            finest_flow = study["flows"][delta_sorted[0]]
            inv = compute_invariants(finest_flow, alpha, m0)
            print(f"  Invariants (finest Δ={delta_sorted[0]}): "
                  f"balance_ok={inv['exact_balance_ok']}, "
                  f"resource_ok={inv['resource_invariance_ok']}")

            # 5. PDE residuals for finest mesh
            pde = estimate_pde_residuals(finest_flow, tau_cutoff=tau_cutoff)
            print(f"  PDE residual: max={pde['max_pde_residual']:.4g}, "
                  f"mean={pde['mean_pde_residual']:.4g}")

            # 6. Initial boundary trace: Λ_{t→0} → a+b
            kappas_finest = study["boundaries"][delta_sorted[0]]["kappa"]
            kappa_first = float(kappas_finest[0]) if len(kappas_finest) > 0 and not np.isnan(kappas_finest[0]) else np.nan
            a_b_sum = case["a_inf"] + case["b_sup"]
            print(f"  First threshold κ₁={kappa_first:.4f} vs a+b={a_b_sum:.4f}")

            # Payoff convergence summary: std across Δ values
            payoff_std = {}
            for pname in PAYOFF_NAMES:
                vals = [study["terminal_payoffs"][d][pname] for d in valid_deltas if pname in study["terminal_payoffs"][d]]
                payoff_std[pname] = float(np.std(vals)) if len(vals) > 1 else 0.0

            # 7. Plots
            plot_boundary_convergence(
                bnd_on_grid, t_eval, case, alpha,
                case["a_inf"], case["b_sup"],
                str(RESULTS_DIR / f"exp2_boundary_{cname}_a{alpha}.png"),
            )
            plot_payoff_convergence(
                conv_df, case, alpha,
                str(RESULTS_DIR / f"exp2_payoff_conv_{cname}_a{alpha}.png"),
            )
            plot_support_evolution(
                study["flows"], case, alpha,
                str(RESULTS_DIR / f"exp2_support_{cname}_a{alpha}.png"),
            )

            record = {
                "case": cname,
                "alpha": alpha,
                "m0": m0,
                "finest_delta": delta_sorted[0],
                "n_meshes": len(valid_deltas),
                "exact_balance_ok": str(inv["exact_balance_ok"]),
                "resource_inv_ok": str(inv["resource_invariance_ok"]),
                "max_bal_res": f"{inv['max_balance_residual']:.2e}",
                "max_inv_res": f"{inv['max_invariance_residual']:.2e}",
                "max_pde_residual": f"{pde['max_pde_residual']:.4g}",
                "bnd_convergence_diff": f"{bnd_diff['max_bnd_diff_finest_vs_next']:.4g}",
                "kappa_first": f"{kappa_first:.4f}",
                "a_plus_b": f"{a_b_sum:.4f}",
                **{f"payoff_std_{p}": f"{payoff_std[p]:.2e}" for p in PAYOFF_NAMES},
            }
            summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(RESULTS_DIR / "exp2_summary.csv", index=False)
    print(f"\nExp2 summary saved to {RESULTS_DIR}/exp2_summary.csv")
    return {"summary_df": summary_df}


if __name__ == "__main__":
    run_experiment_2()
