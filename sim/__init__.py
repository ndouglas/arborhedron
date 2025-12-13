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
"""

from sim.config import Allocation, ClimateConfig, SimConfig, StressParams, TreeState
from sim.rollout import Trajectory, run_season

__all__ = [
    "Allocation",
    "ClimateConfig",
    "SimConfig",
    "StressParams",
    "TreeState",
    "Trajectory",
    "run_season",
]
