"""
exp3_stability.py
-----------------
Experiment 3: Uniqueness, Stability, and Robustness Across Initial Laws and Payoffs.

Three blocks:
  Block A: Payoff invariance — cutoff flow is optimal for all admissible concave payoffs.
  Block B: Robustness across baseline families (smooth/disconnected/atomic).
  Block C: Perturbation stability — controlled perturbations of initial law give
           controlled changes in terminal values, default curves, and boundaries.
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

from src.laws import (
    DiscreteLaw,
    standard_test_cases,
    make_uniform_law,
    make_mixture_uniform_law,
    make_atomic_law,
    d_T_sharp,
    uncontrolled_default_curve,
)
from src.cutoff import run_cutoff_flow
from src.metrics import all_payoffs, PAYOFF_NAMES, compute_invariants, flow_path_summary
from src.semigroup import apply_semigroup
from src.policies import benchmark_policies

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
sns.set_style("whitegrid")


# ================================================================== #
# Block A: Payoff invariance                                          #
# ================================================================== #

def block_a_payoff_invariance(
    cases: list,
    T: float = 1.0,
    Delta: float = 2**-5,
) -> dict:
    """Verify that the cutoff flow is optimal for all admissible payoffs.

    For each (case, alpha): run cutoff flow once, evaluate all payoffs,
    compare terminal payoffs against benchmark policies (which use the
    same per-step budgets from the cutoff flow).
    """
    records = []
    for case in cases:
        m0 = case["m0"]
        for alpha in case["alpha_values"]:
            if m0 <= alpha:
                continue
            print(f"  Block A: {case['name']}, α={alpha}")
            mu0 = case["law"]

            flow = run_cutoff_flow(mu0, alpha, T, Delta)
            terminal_cutoff = all_payoffs(flow["laws"][-1])

            # Run each benchmark at the same per-step budgets
            bench_names = ["proportional_haircut", "flat_haircut", "bottom_stripping"]
            from src.policies import (
                proportional_haircut_law,
                flat_haircut_law,
                bottom_stripping_law,
            )
            policy_fns = {
                "proportional_haircut": proportional_haircut_law,
                "flat_haircut": flat_haircut_law,
                "bottom_stripping": bottom_stripping_law,
            }
            terminal_bench = {}
            for bname in bench_names:
                mu_bench = mu0
                fn = policy_fns[bname]
                for n in range(flow["M"]):
                    c_n = float(flow["budgets"][n])
                    mu_hat = apply_semigroup(mu_bench, Delta, output_grid=mu0.grid)
                    if c_n > 1e-12:
                        mu_new = fn(mu_hat, c_n)
                        if mu_new is None:
                            mu_new = mu_hat
                    else:
                        mu_new = mu_hat
                    mu_bench = mu_new
                terminal_bench[bname] = all_payoffs(mu_bench)

            # For each payoff: is cutoff ≥ all benchmarks?
            dominates_all = {}
            for pname in PAYOFF_NAMES:
                v_cutoff = terminal_cutoff[pname]
                bench_vals = [terminal_bench[b][pname] for b in bench_names
                              if not np.isnan(terminal_bench[b][pname])]
                dominates_all[pname] = all(v_cutoff >= bv - 1e-8 for bv in bench_vals)

            record = {
                "case": case["name"],
                "alpha": alpha,
                "cutoff_always_optimal": all(dominates_all.values()),
                **{f"cutoff_{p}": terminal_cutoff[p] for p in PAYOFF_NAMES},
                **{f"dom_{p}": dominates_all[p] for p in PAYOFF_NAMES},
            }
            records.append(record)
            print(f"    Cutoff always optimal: {record['cutoff_always_optimal']}")
            print(f"    Payoff dominance: {dominates_all}")

    return {"records": records}


# ================================================================== #
# Block B: Robustness across baseline families                        #
# ================================================================== #

def block_b_robustness(
    cases: list,
    T: float = 1.0,
    Delta: float = 2**-5,
) -> dict:
    """Run the cutoff scheme for all three baseline families and compare."""
    rows = []
    for case in cases:
        m0 = case["m0"]
        for alpha in case["alpha_values"]:
            if m0 <= alpha:
                continue
            print(f"  Block B: {case['name']}, α={alpha}")
            flow = run_cutoff_flow(case["law"], alpha, T, Delta)
            inv = compute_invariants(flow, alpha, m0)
            terminal = all_payoffs(flow["laws"][-1])
            row = {
                "case": case["name"],
                "alpha": alpha,
                "m0": m0,
                "terminal_q": float(flow["q_path"][-1]),
                "terminal_m": float(flow["m_path"][-1]),
                "exact_balance_ok": inv["exact_balance_ok"],
                "resource_inv_ok": inv["resource_invariance_ok"],
                "max_bal_res": inv["max_balance_residual"],
                "max_inv_res": inv["max_invariance_residual"],
                **{f"V_{p}": terminal[p] for p in PAYOFF_NAMES},
            }
            rows.append(row)

    return {"records": rows}


# ================================================================== #
# Block C: Perturbation stability                                     #
# ================================================================== #

def build_perturbed_laws(
    case: dict,
    eps_list: list,
    x_max: float = 4.0,
    N: int = 1000,
) -> dict[float, DiscreteLaw]:
    """Build perturbed versions of the baseline law.

    Perturbation types:
      - uniform:    Unif[0.5+ε, 1.5+ε]
      - mixture:    0.5·Unif[0.2, 0.4+ε] + 0.5·Unif[1.2, 1.4-ε]
      - atomic:     0.5·δ_{0.25+ε} + 0.5·δ_{1.25-ε}
    Each perturbed law must satisfy support ⊂ (0, ∞) and mean > alpha.
    """
    perturbed = {}
    cname = case["name"]
    for eps in eps_list:
        if cname == "mu1_uniform":
            # Shift both endpoints right by eps
            a_p, b_p = 0.5 + eps, 1.5 + eps
            if a_p <= 0 or b_p <= a_p:
                continue
            law_p = make_uniform_law(a_p, b_p, x_max=x_max, N=N)
        elif cname == "mu2_mixture_uniform":
            # Expand second component
            a2_p, b2_p = 1.2, 1.4 - eps
            if b2_p <= a2_p + 0.05 or b2_p <= 0:
                continue
            law_p = make_mixture_uniform_law(0.2, 0.4, a2_p, b2_p, p1=0.5, x_max=x_max, N=N)
        elif cname == "mu3_atomic":
            # Shift atom locations
            x1_p = 0.25 + eps
            x2_p = 1.25 - eps
            if x1_p >= x2_p or x1_p <= 0:
                continue
            law_p = make_atomic_law([x1_p, x2_p], [0.5, 0.5], x_max=x_max, N=N)
        else:
            continue
        perturbed[eps] = law_p
    return perturbed


def block_c_perturbation_stability(
    cases: list,
    T: float = 1.0,
    Delta: float = 2**-5,
    eps_list: list | None = None,
    tau_positive: float = 0.1,
) -> dict:
    """Measure how small perturbations of μ_0 affect the optimal flow.

    For each (case, alpha, eps): compute d_T^# and induced changes in
    terminal values, default curves, and boundary paths.
    """
    if eps_list is None:
        eps_list = [0.10, 0.05, 0.02, 0.01, 0.005]

    a_grid = np.linspace(0.0, 4.0, 200)
    t_grid_def = np.linspace(0.01, T, 50)

    all_records = []
    plot_data = {}  # For stability plots

    for case in cases:
        m0 = case["m0"]
        cname = case["name"]

        for alpha in case["alpha_values"]:
            if m0 <= alpha:
                continue
            print(f"  Block C: {cname}, α={alpha}")

            # Baseline flow
            mu0_base = case["law"]
            flow_base = run_cutoff_flow(mu0_base, alpha, T, Delta)
            term_base = all_payoffs(flow_base["laws"][-1])
            kap_base = np.array([k if not np.isinf(k) else np.nan for k in flow_base["kappas"]])

            # Perturbed laws
            perturbed = build_perturbed_laws(case, eps_list)

            eps_records = []
            d_T_list, term_diffs, kap_diffs = [], [], {}

            for eps, law_p in perturbed.items():
                # Check admissibility
                if law_p.mean <= alpha:
                    print(f"    eps={eps}: perturbed law not admissible (mean={law_p.mean:.3f}≤α)")
                    continue

                # d_T^# metric
                d_metric = d_T_sharp(mu0_base, law_p, a_grid, t_grid_def)

                # Flow for perturbed law
                try:
                    flow_p = run_cutoff_flow(law_p, alpha, T, Delta)
                except Exception as e:
                    print(f"    eps={eps}: flow failed: {e}")
                    continue

                term_p = all_payoffs(flow_p["laws"][-1])

                # Discrepancies
                q_base_arr = flow_base["q_path"]
                q_p_arr = flow_p["q_path"]
                n_min = min(len(q_base_arr), len(q_p_arr))
                sup_q_diff = float(np.max(np.abs(q_base_arr[:n_min] - q_p_arr[:n_min])))
                sup_m_diff = float(np.max(np.abs(flow_base["m_path"][:n_min] - flow_p["m_path"][:n_min])))
                sup_h_diff = float(np.max(np.abs(flow_base["h_path"][:n_min] - flow_p["h_path"][:n_min])))

                # Boundary discrepancy on [tau, T]
                kap_p = np.array([k if not np.isinf(k) else np.nan for k in flow_p["kappas"]])
                t_alloc = flow_base["t_grid"][1:]
                mask = t_alloc >= tau_positive
                if np.any(mask) and len(kap_base) == len(kap_p):
                    sup_kap_diff = float(np.nanmax(np.abs(kap_base[mask] - kap_p[mask])))
                else:
                    sup_kap_diff = np.nan

                # Terminal payoff differences
                v_diffs = {p: abs(term_base[p] - term_p[p]) for p in PAYOFF_NAMES}

                record = {
                    "case": cname,
                    "alpha": alpha,
                    "eps": eps,
                    "d_T_sharp": d_metric,
                    "sup_q_diff": sup_q_diff,
                    "sup_m_diff": sup_m_diff,
                    "sup_h_diff": sup_h_diff,
                    "sup_kap_diff": sup_kap_diff,
                    **{f"v_diff_{p}": v_diffs[p] for p in PAYOFF_NAMES},
                }
                eps_records.append(record)
                all_records.append(record)
                d_T_list.append(d_metric)
                for p in PAYOFF_NAMES:
                    kap_diffs.setdefault(p, []).append(v_diffs[p])

                print(f"    eps={eps:.3f}: d_T#={d_metric:.4f}, "
                      f"sup_q_diff={sup_q_diff:.4f}, sup_kap_diff={sup_kap_diff:.4f}")

            plot_data[f"{cname}_a{alpha}"] = {
                "eps_records": eps_records,
                "d_T_list": d_T_list,
                "kap_diffs": kap_diffs,
            }

    return {"records": all_records, "plot_data": plot_data}


# ================================================================== #
# Plotting                                                            #
# ================================================================== #

def plot_payoff_invariance(block_a_records: list, save_path: str) -> None:
    """Bar chart: cutoff payoff values across cases and payoffs."""
    df = pd.DataFrame(block_a_records)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    payoff_cols = [f"cutoff_{p}" for p in PAYOFF_NAMES if f"cutoff_{p}" in df.columns]
    df_plot = df[["case", "alpha"] + payoff_cols].copy()
    df_melt = df_plot.melt(id_vars=["case", "alpha"], value_vars=payoff_cols,
                           var_name="payoff", value_name="value")
    df_melt["payoff"] = df_melt["payoff"].str.replace("cutoff_", "")
    df_melt["label"] = df_melt["case"].str[-4:] + " α=" + df_melt["alpha"].astype(str)
    sns.barplot(data=df_melt, x="payoff", y="value", hue="label", ax=ax)
    ax.set_title("Block A: Cutoff payoffs across cases and α values")
    ax.set_xlabel("Payoff f"); ax.set_ylabel("E[f(X_T)]")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_stability_curves(plot_data: dict, save_path: str) -> None:
    """Plot d_T^# vs terminal payoff differences."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, data in plot_data.items():
        if not data["eps_records"]:
            continue
        eps_rec = pd.DataFrame(data["eps_records"])
        for p in ["linear", "survival"]:
            col = f"v_diff_{p}"
            if col in eps_rec.columns:
                ax.scatter(eps_rec["d_T_sharp"], eps_rec[col], label=f"{label}/{p}", alpha=0.7)
    ax.set_xlabel("d_T^#(μ, μ_ε)")
    ax.set_ylabel("|V_f(μ) - V_f(μ_ε)|")
    ax.set_title("Block C: Perturbation stability")
    ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_robustness_paths(block_b_records: list, save_path: str) -> None:
    """Compare q, m paths across the three baseline law families."""
    if not block_b_records:
        return
    df = pd.DataFrame(block_b_records)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, (col, ylabel) in enumerate([("terminal_q", "P(τ≤T)"), ("terminal_m", "E[X_T]")]):
        ax = axes[i]
        for (case_name,), grp in df.groupby(["case"]):
            ax.plot(grp["alpha"], grp[col], "o-", lw=1.5, label=case_name, ms=6)
        ax.set_xlabel("α"); ax.set_ylabel(ylabel)
        ax.set_title(f"Block B: {ylabel} across baseline families")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ================================================================== #
# Main orchestration                                                  #
# ================================================================== #

def run_experiment_3(
    T: float = 1.0,
    Delta: float = 2**-5,
    eps_list: list | None = None,
) -> dict:
    """Run all blocks of Experiment 3."""
    if eps_list is None:
        eps_list = [0.10, 0.05, 0.02, 0.01, 0.005]

    cases = standard_test_cases(x_max=4.0, N=1000)

    print("\n--- Block A: Payoff Invariance ---")
    result_a = block_a_payoff_invariance(cases, T=T, Delta=Delta)

    print("\n--- Block B: Baseline Robustness ---")
    result_b = block_b_robustness(cases, T=T, Delta=Delta)

    print("\n--- Block C: Perturbation Stability ---")
    result_c = block_c_perturbation_stability(cases, T=T, Delta=Delta, eps_list=eps_list)

    # Save results
    df_a = pd.DataFrame(result_a["records"])
    df_b = pd.DataFrame(result_b["records"])
    df_c = pd.DataFrame(result_c["records"])

    df_a.to_csv(RESULTS_DIR / "exp3_block_a.csv", index=False)
    df_b.to_csv(RESULTS_DIR / "exp3_block_b.csv", index=False)
    df_c.to_csv(RESULTS_DIR / "exp3_block_c.csv", index=False)

    # Plots
    plot_payoff_invariance(
        result_a["records"],
        str(RESULTS_DIR / "exp3_payoff_invariance.png"),
    )
    plot_robustness_paths(
        result_b["records"],
        str(RESULTS_DIR / "exp3_robustness_paths.png"),
    )
    plot_stability_curves(
        result_c["plot_data"],
        str(RESULTS_DIR / "exp3_stability_curves.png"),
    )

    print(f"\nExp3 results saved to {RESULTS_DIR}/")

    # Summary
    print("\n=== Exp3 Block A: Payoff Invariance Summary ===")
    print(df_a[["case", "alpha", "cutoff_always_optimal"]].to_string(index=False))

    print("\n=== Exp3 Block B: Baseline Robustness Summary ===")
    print(df_b[["case", "alpha", "exact_balance_ok", "resource_inv_ok"]].to_string(index=False))

    if not df_c.empty:
        print("\n=== Exp3 Block C: Perturbation Stability Summary ===")
        print(df_c[["case", "alpha", "eps", "d_T_sharp", "sup_q_diff"]].to_string(index=False))

    return {
        "block_a": result_a,
        "block_b": result_b,
        "block_c": result_c,
        "df_a": df_a,
        "df_b": df_b,
        "df_c": df_c,
    }


if __name__ == "__main__":
    run_experiment_3()
