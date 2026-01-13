"""
Causal Fairness Term for the FAMAIL Objective Function.

This module implements the Causal Fairness term ($F_{causal}$), which measures
the degree to which taxi service supply is explained by passenger demand alone,
rather than by other contextual factors that may represent unfair biases.

Core Principle:
    In a causally fair system, the service supply-to-demand ratio should be
    consistent across all locations when controlling for demand. If two areas
    have the same demand, they should receive the same supply.

Mathematical Formulation:
    F_causal = (1/|P|) * Σ_p F_causal^p
    
    where F_causal^p = R² = Var(g(D)) / Var(Y) = 1 - Var(R) / Var(Y)
    
    - Y = Service ratio (Supply / Demand)
    - D = Demand (pickup counts)
    - g(d) = Expected service ratio given demand d
    - R = Y - g(D) = Residual (unexplained variation)

Value Range: [0, 1]
    - F_causal = 1: Service perfectly explained by demand (no contextual bias)
    - F_causal = 0: Service independent of demand (maximum unfairness)

Reference:
    FAMAIL project internal documentation based on counterfactual fairness literature
"""

from .config import CausalFairnessConfig
from .term import CausalFairnessTerm

__all__ = [
    'CausalFairnessConfig',
    'CausalFairnessTerm',
]
