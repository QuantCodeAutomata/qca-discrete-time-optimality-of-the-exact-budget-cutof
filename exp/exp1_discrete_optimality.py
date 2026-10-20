"""
exp1_discrete_optimality.py
----------------------------
Experiment 1: Discrete-Time Optimality of the Exact-Budget Cutoff Allocation Rule.

Validates:
  1. One-step cutoff dominance: D_{T_c μ}(a) ≥ D_ν(a) for admissible ν.
  2. Dynamic optimality: repeated cutoff outperforms benchmark policies.
  3. Payoff invariance: same cutoff flow is optimal for all admissible payoffs.
  4. Exact balance and resource invariance along the discrete flow.
  5. Agreement between law-level recursion and pathwise discrete reflection.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Custom imports — Context7 found no library equivalent for these objects
from src.laws import standard_test_cases, DiscreteLaw
from src.cutoff import run_cutoff_flow, verify_cap_identities, apply_cap, find_threshold
from src.semigroup import apply_semigroup
from src.policies import benchmark_policies
from src.metrics import (
    d_curve,
    dominance_check,
    compute_invariants,
    all_payoffs,
    flow_path_summary,
    PAYOFF_NAMES,
)
from src.simulate import pathwise_discrete_reflection, compare_law_vs_pathwise

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
PALETTE = sns.color_palette("tab10")


# ================================================================== #
# 1. One-step cutoff dominance                                        #
# ================================================================== #

def run_one_step_dominance(
    case: dict,
    alpha: float,
    T: float = 1.0,
    Delta: float = 0.125,
    a_grid: np.ndarray | None = None,
) -> dict:
    """Run one-step comparison: T_c μ̂ vs benchmark admissible policies.

    Uses the first step of the discrete flow for a representative pre-allocation law.

    Returns a dict with dominance results and objective comparisons.
    """
    mu0: DiscreteLaw = case["law"]
    if a_grid is None:
        a_grid = np.linspace(0.0, float(mu0.grid[-1]), 300)

    # Propagate one step
    mu_hat = apply_semigroup(mu0, Delta)
    q_before = mu0.default_mass
    q_after = mu_hat.default_mass
    delta_q = max(q_after - q_before, 0.0)
    c_budget = alpha * delta_q

    if c_budget < 1e-12:
        return {"skipped": True, "reason": "c_budget too small"}

    # Cutoff law
    mu_capped, kappa = apply_cap(mu_hat, c_budget)

    # Verify identities C_{T_c μ}(a) = (C_μ(a) - c)⁺
    ident_check = verify_cap_identities(mu_hat, c_budget, a_grid)

    # Benchmark laws
    bench = benchmark_policies(mu_hat, c_budget)

    # Dominance checks
    dom_results = {}
    payoff_results = {"cutoff": all_payoffs(mu_capped)}
    for name, bench_law in bench.items():
        if bench_law is None:
            dom_results[name] = {"dominates": None}
            payoff_results[name] = {p: np.nan for p in PAYOFF_NAMES}
            continue
        dom = dominance_check(mu_capped, bench_law, a_grid)
        dom_results[name] = {
            "min_diff": dom["min_diff"],
            "dominates": dom["dominates"],
        }
        payoff_results[name] = all_payoffs(bench_law)

    return {
        "skipped": False,
        "c_budget": float(c_budget),
        "kappa": float(kappa),
        "delta_q": float(delta_q),
        "identity_check": ident_check,
        "dominance": dom_results,
        "payoffs": payoff_results,
        "a_grid": a_grid,
        "D_cutoff": d_curve(mu_capped, a_grid),
        "D_bench": {
            name: d_curve(bl, a_grid) if bl is not None else None
            for name, bl in bench.items()
        },
    }


# ================================================================== #
# 2. Dynamic optimality                                               #
# ================================================================== #

def run_dynamic_comparison(
    case: dict,
    alpha: float,
    T: float = 1.0,
    Delta: float = 0.0625,
) -> dict:
    """Run full dynamic cutoff flow and compare to benchmark policies.

    Benchmark: run each benchmark policy at each step with the same budget c_n.
    """
    mu0: DiscreteLaw = case["law"]

    # --- Cutoff flow ---
    flow_cutoff = run_cutoff_flow(mu0, alpha, T, Delta)

    # --- Benchmark flows: apply benchmark policy at each step ---
    M = flow_cutoff["M"]
    bench_names = ["proportional_haircut", "flat_haircut", "bottom_stripping"]
    bench_flows = {name: {"laws": [mu0], "q": [mu0.default_mass],
                          "m": [mu0.mean], "h": [0.0]} for name in bench_names}

    for n in range(M):
        c_n = float(flow_cutoff["budgets"][n])
        # Each benchmark runs its policy using the pre-allocation law from *its own* flow
        for name in bench_names:
            mu_prev = bench_flows[name]["laws"][-1]
            mu_hat_n = apply_semigroup(mu_prev, Delta, output_grid=mu0.grid)
            if c_n < 1e-12:
                mu_next_n = mu_hat_n
            else:
                from src.policies import (
                    proportional_haircut_law,
                    flat_haircut_law,
                    bottom_stripping_law,
                )
                policy_fn = {
                    "proportional_haircut": proportional_haircut_law,
                    "flat_haircut": flat_haircut_law,
                    "bottom_stripping": bottom_stripping_law,
                }[name]
                mu_next_n = policy_fn(mu_hat_n, c_n)
                if mu_next_n is None:
                    mu_next_n = mu_hat_n  # fallback: no allocation

            bench_flows[name]["laws"].append(mu_next_n)
            bench_flows[name]["q"].append(mu_next_n.default_mass)
            bench_flows[name]["m"].append(mu_next_n.mean)
            bench_flows[name]["h"].append(alpha * mu_next_n.default_mass)

    # --- Terminal payoff comparison ---
    terminal_cutoff = all_payoffs(flow_cutoff["laws"][-1])
    terminal_bench = {}
    for name in bench_names:
        terminal_bench[name] = all_payoffs(bench_flows[name]["laws"][-1])

    # --- Invariant checks for cutoff ---
    inv = compute_invariants(flow_cutoff, alpha, case["m0"])

    return {
        "flow_cutoff": flow_cutoff,
        "bench_flows": bench_flows,
        "terminal_cutoff": terminal_cutoff,
        "terminal_bench": terminal_bench,
        "invariants": inv,
    }


# ================================================================== #
# 3. Pathwise validation                                              #
# ================================================================== #

def run_pathwise_validation(
    case: dict,
    alpha: float,
    T: float = 1.0,
    Delta: float = 0.0625,
    n_paths: int = 5000,
    seed: int = 42,
) -> dict:
    """Compare law-level recursion with independent pathwise construction."""
    mu0: DiscreteLaw = case["law"]
    flow = run_cutoff_flow(mu0, alpha, T, Delta)
    kappas = flow["kappas"]

    pw = pathwise_discrete_reflection(kappas, Delta, mu0, n_paths=n_paths, seed=seed)
    comparison = compare_law_vs_pathwise(flow, pw, alpha)
    return {"flow": flow, "pathwise": pw, "comparison": comparison}


# ================================================================== #
# Plotting helpers                                                    #
# ================================================================== #

def plot_d_curves_one_step(result: dict, case: dict, alpha: float, save_path: str) -> None:
    if result.get("skipped"):
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    a_grid = result["a_grid"]
    ax.plot(a_grid, result["D_cutoff"], "k-", lw=2, label="Cutoff T_c μ̂")
    colors = PALETTE[1:]
    for i, (name, D_b) in enumerate(result["D_bench"].items()):
        if D_b is not None:
            ax.plot(a_grid, D_b, "--", color=colors[i], lw=1.5, label=name.replace("_", " "))
    ax.set_xlabel("Strike a")
    ax.set_ylabel("D_μ(a)")
    ax.set_title(
        f"One-step D-curves: {case['label']}, α={alpha}, κ={result['kappa']:.4f}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_flow_paths(flow: dict, case: dict, alpha: float, save_path: str) -> None:
    t = flow["t_grid"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(t, flow["q_path"], "b-o", ms=3, label="q_n (cutoff)")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("Default mass q"); axes[0].set_title("Default path")
    axes[1].plot(t, flow["m_path"], "g-o", ms=3, label="m_n (cutoff)")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("Mean reserve m"); axes[1].set_title("Mean path")
    kap_t = t[1:]
    kappas_plot = [k if not np.isinf(k) else np.nan for k in flow["kappas"]]
    axes[2].plot(kap_t, kappas_plot, "r-o", ms=3)
    axes[2].set_xlabel("t"); axes[2].set_ylabel("κ"); axes[2].set_title("Threshold κ_n")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(f"{case['label']}, α={alpha}, Δ={flow['Delta']:.4f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_pathwise_comparison(pw: dict, flow: dict, case: dict, alpha: float, save_path: str) -> None:
    t = flow["t_grid"]
    n = min(len(t), len(pw["q_empirical"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(t[:n], flow["q_path"][:n], "b-", lw=2, label="Law-level q")
    axes[0].plot(t[:n], pw["q_empirical"][:n], "r--", lw=1.5, label="Pathwise q (MC)")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("P(τ≤t)"); axes[0].set_title("Default probability")
    axes[0].legend()

    axes[1].plot(t[:n], flow["m_path"][:n], "b-", lw=2, label="Law-level m")
    axes[1].plot(t[:n], pw["m_empirical"][:n], "r--", lw=1.5, label="Pathwise m (MC)")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("Mean reserve"); axes[1].set_title("Mean reserve")
    axes[1].legend()

    fig.suptitle(f"Law vs Pathwise: {case['label']}, α={alpha}", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ================================================================== #
# Main orchestration                                                  #
# ================================================================== #

def run_experiment_1(
    T: float = 1.0,
    Delta_dynamic: float = 2**-5,
    Delta_pathwise: float = 2**-4,
    n_paths_mc: int = 5000,
    seed: int = 42,
) -> dict:
    """Run all sub-experiments for Experiment 1.

    Returns
    -------
    dict with summary metrics for RESULTS.md.
    """
    cases = standard_test_cases(x_max=4.0, N=1000)
    summary_records = []
    all_results = {}

    for case in cases:
        m0 = case["m0"]
        cname = case["name"]
        all_results[cname] = {}

        for alpha in case["alpha_values"]:
            if m0 <= alpha:
                print(f"  Skip {cname}, α={alpha} (m0={m0:.2f} ≤ α)")
                continue

            print(f"\n=== Exp1: {cname}, α={alpha} ===")
            key = f"{cname}_a{alpha}"
            all_results[cname][f"alpha_{alpha}"] = {}

            # ---- 1. One-step dominance ----
            ods = run_one_step_dominance(case, alpha, T=T, Delta=Delta_pathwise)
            all_results[cname][f"alpha_{alpha}"]["one_step"] = ods
            if not ods.get("skipped"):
                # Identity check
                id_ok = ods["identity_check"]["passed"]
                print(f"  Identity check passed: {id_ok}  (max_call_res={ods['identity_check']['max_call_residual']:.2e})")
                for pname, dom in ods["dominance"].items():
                    if dom["dominates"] is not None:
                        print(f"  {pname}: dominates={dom['dominates']}, min_diff={dom['min_diff']:.4f}")

                plot_d_curves_one_step(
                    ods, case, alpha,
                    str(RESULTS_DIR / f"exp1_dcurves_{cname}_a{alpha}.png"),
                )

            # ---- 2. Dynamic comparison ----
            dyn = run_dynamic_comparison(case, alpha, T=T, Delta=Delta_dynamic)
            all_results[cname][f"alpha_{alpha}"]["dynamic"] = dyn
            inv = dyn["invariants"]
            print(f"  Dynamic invariants: balance_ok={inv['exact_balance_ok']}, "
                  f"resource_ok={inv['resource_invariance_ok']}, "
                  f"max_bal_res={inv['max_balance_residual']:.2e}, "
                  f"max_inv_res={inv['max_invariance_residual']:.2e}")

            # Payoff comparison
            tc = dyn["terminal_cutoff"]
            print(f"  Terminal payoffs (cutoff): {tc}")
            for bname, bval in dyn["terminal_bench"].items():
                cutoff_wins = {
                    p: tc[p] >= bval[p] - 1e-8
                    for p in PAYOFF_NAMES if not np.isnan(bval.get(p, np.nan))
                }
                print(f"  vs {bname}: cutoff wins = {cutoff_wins}")

            plot_flow_paths(
                dyn["flow_cutoff"], case, alpha,
                str(RESULTS_DIR / f"exp1_flow_{cname}_a{alpha}.png"),
            )

            # ---- 3. Pathwise validation ----
            pv = run_pathwise_validation(
                case, alpha, T=T, Delta=Delta_pathwise,
                n_paths=n_paths_mc, seed=seed,
            )
            all_results[cname][f"alpha_{alpha}"]["pathwise"] = pv
            comp = pv["comparison"]
            print(f"  Pathwise comparison: max_q_diff={comp['max_q_diff']:.4f}, "
                  f"max_m_diff={comp['max_m_diff']:.4f}, max_h_diff={comp['max_h_diff']:.4f}")

            plot_pathwise_comparison(
                pv["pathwise"], pv["flow"], case, alpha,
                str(RESULTS_DIR / f"exp1_pathwise_{cname}_a{alpha}.png"),
            )

            # ---- Summary record ----
            dom_all = "N/A" if ods.get("skipped") else str(
                all(d.get("dominates", False) for d in ods["dominance"].values()
                    if d.get("dominates") is not None)
            )
            record = {
                "case": cname,
                "alpha": alpha,
                "m0": m0,
                "identity_ok": str(ods.get("identity_check", {}).get("passed", "skipped")),
                "one_step_dominance": dom_all,
                "exact_balance_ok": str(inv["exact_balance_ok"]),
                "resource_inv_ok": str(inv["resource_invariance_ok"]),
                "max_bal_res": float(f"{inv['max_balance_residual']:.2e}"),
                "max_inv_res": float(f"{inv['max_invariance_residual']:.2e}"),
                "max_q_diff_mc": float(f"{comp['max_q_diff']:.4f}"),
                **{f"cutoff_{p}": f"{tc[p]:.6f}" for p in PAYOFF_NAMES},
            }
            summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(RESULTS_DIR / "exp1_summary.csv", index=False)
    print(f"\nExp1 summary saved to {RESULTS_DIR}/exp1_summary.csv")

    return {"summary_df": summary_df, "all_results": all_results}


if __name__ == "__main__":
    run_experiment_1()
