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
    stained_glass: Stained glass visualization renderer
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
    BlossomGeom,
    BlossomStyle,
    LeafGeom,
    LeafStyle,
    ShootGeom,
    ShootStyle,
    StressProfile,
    make_blossom,
    make_leaf,
    make_shoot,
    render_blossom,
    render_leaf,
    render_shoot_scene,
    render_stained_glass,
    save_shoot_scene,
    save_stained_glass,
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
    # Stained glass geometry
    "BlossomGeom",
    "LeafGeom",
    "ShootGeom",
    # Stained glass styles
    "BlossomStyle",
    "LeafStyle",
    "ShootStyle",
    # Stress system
    "StressProfile",
    # Factory functions
    "make_blossom",
    "make_leaf",
    "make_shoot",
    # Rendering
    "render_blossom",
    "render_leaf",
    "render_shoot_scene",
    "render_stained_glass",
    # Save utilities
    "save_shoot_scene",
    "save_stained_glass",
]
