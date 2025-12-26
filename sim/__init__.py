"""
Arborhedron Simulation Module

A differentiable tree growth simulator that models resource allocation
economics under environmental stress.

Modules:
    config: Constants and configuration
    stress: Environmental stressor functions (light, moisture, wind)
    surrogates: Biological surrogate functions (photosynthesis, uptake, damage)
    dynamics: State update logic
    policies: Allocation policies (hand-coded baseline, learned)
    rollout: Full season simulation
    carbon: Carbon sequestration metrics
    resilience: Tipping point and resilience analysis
    stained_glass: L-system tree with stained glass leaf panels
"""

from sim.carbon import (
    carbon_objective,
    carbon_seed_tradeoff,
    compute_carbon_content,
    compute_carbon_efficiency,
    compute_carbon_score,
    compute_carbon_summary,
)
from sim.config import Allocation, ClimateConfig, SimConfig, StressParams, TreeState
from sim.policies import (
    NeuralPolicy,
    apply_neural_policy,
    make_neural_policy_fn,
    make_policy_features,
)
from sim.resilience import (
    compute_resilience_boundary,
    compute_sensitivity,
    find_tipping_points,
    make_climate_with_offsets,
    parameter_sweep_2d,
    plot_fitness_landscape,
    plot_resilience_boundary,
    plot_sensitivity_curve,
    resilience_report,
    sensitivity_sweep,
)
from sim.rollout import Trajectory, run_season
from sim.stained_glass import (
    Branch,
    LeafGeom,
    # Stress integration
    StressVisuals,
    TreeParams,
    TreeSkeleton,
    TreeStyle,
    compute_stress_visuals,
    generate_tree_skeleton,
    make_leaf,
    render_stained_glass,
    render_stressed_tree,
    render_tree,
    save_stained_glass,
    save_stressed_tree,
    save_tree,
    stress_to_params,
    stress_to_style,
)

__all__ = [
    # Config
    "Allocation",
    "ClimateConfig",
    "SimConfig",
    "StressParams",
    "TreeState",
    # Policies
    "NeuralPolicy",
    "apply_neural_policy",
    "make_neural_policy_fn",
    "make_policy_features",
    # Simulation
    "Trajectory",
    "run_season",
    # Carbon sequestration
    "carbon_objective",
    "carbon_seed_tradeoff",
    "compute_carbon_content",
    "compute_carbon_efficiency",
    "compute_carbon_score",
    "compute_carbon_summary",
    # Resilience analysis
    "compute_resilience_boundary",
    "compute_sensitivity",
    "find_tipping_points",
    "make_climate_with_offsets",
    "parameter_sweep_2d",
    "plot_fitness_landscape",
    "plot_resilience_boundary",
    "plot_sensitivity_curve",
    "resilience_report",
    "sensitivity_sweep",
    # Tree generation
    "Branch",
    "LeafGeom",
    "TreeParams",
    "TreeSkeleton",
    "TreeStyle",
    "generate_tree_skeleton",
    "make_leaf",
    # Rendering
    "render_stained_glass",
    "render_tree",
    "save_stained_glass",
    "save_tree",
    # Stress-morphology integration
    "StressVisuals",
    "compute_stress_visuals",
    "stress_to_params",
    "stress_to_style",
    "render_stressed_tree",
    "save_stressed_tree",
]
