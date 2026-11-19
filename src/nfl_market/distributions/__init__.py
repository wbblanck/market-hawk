"""Outcome distributions for NFL games (margin, total, team points).

See ``docs/methodology/distributions.md``. Margin = home - away, total = home + away
(``nfl_market.models.interface``).
"""
from .pmf import IntPMF, JointPMF, mixture
from .margin import (DEFAULT_MARGIN_KEYS, SPREAD_BUCKETS, KeyNumberWeights, ResidualPool, TParams,
                     adjust_to_key_targets, apply_key_weights, fit_key_weights, fit_residual_pool,
                     fit_student_t, margin_pmf, spread_bucket, stack_pmfs, t_pmf)
from .totals import total_pmf_discretized, total_pmf_empirical
from .team_points import NBParams, fit_nb_team_points, independence_diagnostics, nb_joint_pmf
from .markets import (ThreeWay, away_spread_probs, conditional_cover, fair_american, home_spread_probs,
                      key_number_probs, moneyline_probs, p_home_win, p_tie, total_probs)

__all__ = [
    "IntPMF", "JointPMF", "mixture", "DEFAULT_MARGIN_KEYS", "SPREAD_BUCKETS", "KeyNumberWeights",
    "ResidualPool", "TParams", "adjust_to_key_targets", "apply_key_weights", "fit_key_weights",
    "fit_residual_pool", "fit_student_t", "margin_pmf", "spread_bucket", "stack_pmfs", "t_pmf",
    "total_pmf_discretized", "total_pmf_empirical", "NBParams", "fit_nb_team_points",
    "independence_diagnostics", "nb_joint_pmf", "ThreeWay", "away_spread_probs", "conditional_cover",
    "fair_american", "home_spread_probs", "key_number_probs", "moneyline_probs", "p_home_win", "p_tie",
    "total_probs",
]
