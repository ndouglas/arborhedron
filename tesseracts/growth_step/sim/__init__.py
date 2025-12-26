"""
Minimal sim module for growth_step tesseract.

Only includes config, dynamics, and surrogates needed for the step function.
"""

from sim.config import Allocation, SimConfig, TreeState
from sim.dynamics import step

__all__ = [
    "Allocation",
    "SimConfig",
    "TreeState",
    "step",
]
