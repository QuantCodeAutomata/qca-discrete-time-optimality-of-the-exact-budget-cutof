# src package: core mathematical modules for cutoff allocation experiments
from .laws import DiscreteLaw, make_uniform_law, make_mixture_uniform_law, make_atomic_law
from .semigroup import apply_semigroup
from .cutoff import find_threshold, apply_cap, cutoff_step
from .policies import proportional_haircut_law, benchmark_policies
from .metrics import (
    payoff_value,
    d_curve,
    compute_invariants,
    all_payoffs,
    PAYOFF_NAMES,
)
from .simulate import pathwise_discrete_reflection

__all__ = [
    "DiscreteLaw",
    "make_uniform_law",
    "make_mixture_uniform_law",
    "make_atomic_law",
    "apply_semigroup",
    "find_threshold",
    "apply_cap",
    "cutoff_step",
    "proportional_haircut_law",
    "benchmark_policies",
    "payoff_value",
    "d_curve",
    "compute_invariants",
    "all_payoffs",
    "PAYOFF_NAMES",
    "pathwise_discrete_reflection",
]
