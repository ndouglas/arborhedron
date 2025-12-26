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
    stained_glass: L-system tree with stained glass leaf panels
"""

from sim.config import Allocation, ClimateConfig, SimConfig, StressParams, TreeState
from sim.policies import (
    NeuralPolicy,
    apply_neural_policy,
    make_neural_policy_fn,
    make_policy_features,
)
from sim.rollout import Trajectory, run_season
from sim.stained_glass import (
    LeafGeom,
    TreeParams,
    TreeStyle,
    TreeSkeleton,
    Branch,
    make_leaf,
    generate_tree_skeleton,
    render_tree,
    save_tree,
    render_stained_glass,
    save_stained_glass,
    # Stress integration
    StressVisuals,
    compute_stress_visuals,
    stress_to_params,
    stress_to_style,
    render_stressed_tree,
    save_stressed_tree,
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
