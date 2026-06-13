"""
Piezo.AI ML Core — optimization subpackage.

Crystal structure analysis + NSGA-II multi-objective property optimization.
"""

from piezo_ml.optimization.structural_analyzer import (
    StructuralAnalyzer,
    StructuralDescriptor,
)
from piezo_ml.optimization.nsga2_optimizer import (
    NSGA2Optimizer,
    OptimizationConfig,
    OptimizationResult,
    ParetoSolution,
    USE_CASE_PRESETS,
    SEARCH_ELEMENTS,
)
from piezo_ml.optimization.pareto_utils import (
    tag_use_case,
    compute_hypervolume_indicator,
    filter_dominated,
    rank_solutions,
)

__all__ = [
    "StructuralAnalyzer",
    "StructuralDescriptor",
    "NSGA2Optimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "ParetoSolution",
    "USE_CASE_PRESETS",
    "SEARCH_ELEMENTS",
    "tag_use_case",
    "compute_hypervolume_indicator",
    "filter_dominated",
    "rank_solutions",
]
