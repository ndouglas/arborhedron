"""
Allocation policies for tree growth.

This module provides various policies that determine how a tree
allocates its energy to different compartments. Policies can be:

1. Hand-coded baselines (for testing and comparison)
2. Learned policies (MLP, etc.) - to be added in Week 2

All policies output Allocation tuples that sum to 1.
"""

import jax.numpy as jnp
from jax import Array
from jax.nn import softmax

from sim.config import Allocation, TreeState


def softmax_allocation(logits: Array) -> Allocation:
    """
    Convert 5 logits to an allocation using softmax.

    This is the core building block for differentiable policies.

    Args:
        logits: Array of shape (5,) - [roots, trunk, shoots, leaves, flowers]

    Returns:
        Valid Allocation with fractions summing to 1
    """
    probs = softmax(logits)
    return Allocation(
        roots=probs[0],
        trunk=probs[1],
        shoots=probs[2],
        leaves=probs[3],
        flowers=probs[4],
    )


def baseline_policy(
    state: TreeState,  # noqa: ARG001
    day: int,
    num_days: int,
    wind: float = 0.0,
) -> Allocation:
    """
    Hand-coded baseline policy following PvZ economics.

    Strategy phases:
    - Early (0-20%): Build roots and leaves for resource gathering
    - Mid (20-60%): Expand with shoots, start building trunk
    - Late (60-100%): Consolidate trunk, invest in flowers

    Args:
        state: Current tree state (unused in baseline, but API-compatible)
        day: Current day (0-indexed)
        num_days: Total season length
        wind: Current wind level (for adaptive response)

    Returns:
        Allocation for this timestep
    """
    progress = day / num_days

    # Base allocations that shift over time
    if progress < 0.2:
        # Early: foundation building
        base_logits = jnp.array([2.0, 0.5, 1.0, 2.5, -2.0])
    elif progress < 0.6:
        # Mid: expansion
        base_logits = jnp.array([1.5, 1.5, 1.5, 1.5, 0.0])
    else:
        # Late: consolidation and reproduction
        base_logits = jnp.array([0.5, 2.0, 0.5, 1.0, 2.0])

    # Wind adaptation: increase trunk allocation when windy
    wind_bonus = jnp.array([0.0, wind * 2.0, -wind * 0.5, -wind * 0.5, 0.0])
    logits = base_logits + wind_bonus

    return softmax_allocation(logits)


def growth_focused_policy(
    state: TreeState,
    day: int,
    num_days: int,
    wind: float = 0.0,
) -> Allocation:
    """
    Policy focused on maximizing growth (roots + leaves).

    Good for benign environments with low stress.

    Args:
        state: Current tree state
        day: Current day
        num_days: Total season length
        wind: Current wind level (unused)

    Returns:
        Growth-focused allocation
    """
    # Heavy investment in roots and leaves
    logits = jnp.array([2.0, 0.0, 1.0, 2.5, 0.0])
    return softmax_allocation(logits)


def defensive_policy(
    state: TreeState,
    day: int,
    num_days: int,
    wind: float = 0.0,
) -> Allocation:
    """
    Policy focused on structural resilience.

    Good for harsh environments with wind/stress.

    Args:
        state: Current tree state
        day: Current day
        num_days: Total season length
        wind: Current wind level (unused)

    Returns:
        Defensive allocation
    """
    # Heavy investment in trunk for structural support
    logits = jnp.array([1.5, 3.0, 0.5, 1.0, 0.5])
    return softmax_allocation(logits)


def reproduction_policy(
    state: TreeState,
    day: int,
    num_days: int,
    wind: float = 0.0,
) -> Allocation:
    """
    Policy focused on reproduction (flowers).

    Good for late-season all-in on seeds.

    Args:
        state: Current tree state
        day: Current day
        num_days: Total season length
        wind: Current wind level (unused)

    Returns:
        Reproduction-focused allocation
    """
    # Heavy investment in flowers
    logits = jnp.array([0.5, 1.0, 0.5, 1.0, 3.0])
    return softmax_allocation(logits)


def phased_policy(
    state: TreeState,
    day: int,
    num_days: int,
    wind: float = 0.0,
    early_logits: Array | None = None,
    mid_logits: Array | None = None,
    late_logits: Array | None = None,
) -> Allocation:
    """
    Parameterized phased policy for optimization.

    This policy can be optimized by learning the logits for each phase.

    Args:
        state: Current tree state
        day: Current day
        num_days: Total season length
        wind: Current wind level
        early_logits: Logits for early phase (days 0-20%)
        mid_logits: Logits for mid phase (days 20-60%)
        late_logits: Logits for late phase (days 60-100%)

    Returns:
        Allocation based on current phase
    """
    # Defaults if not provided
    if early_logits is None:
        early_logits = jnp.array([2.0, 0.5, 1.0, 2.5, -2.0])
    if mid_logits is None:
        mid_logits = jnp.array([1.5, 1.5, 1.5, 1.5, 0.0])
    if late_logits is None:
        late_logits = jnp.array([0.5, 2.0, 0.5, 1.0, 2.0])

    progress = day / num_days

    # Smooth interpolation between phases using sigmoids
    early_weight = 1.0 - jnp.tanh(10.0 * (progress - 0.2))
    late_weight = jnp.tanh(10.0 * (progress - 0.6))
    mid_weight = 1.0 - jnp.abs(early_weight) - jnp.abs(late_weight)

    # Ensure weights are positive and normalized
    early_weight = jnp.maximum(early_weight, 0.0)
    mid_weight = jnp.maximum(mid_weight, 0.0)
    late_weight = jnp.maximum(late_weight, 0.0)
    total_weight = early_weight + mid_weight + late_weight + 1e-8

    # Interpolate logits
    logits = (
        early_weight * early_logits
        + mid_weight * mid_logits
        + late_weight * late_logits
    ) / total_weight

    return softmax_allocation(logits)


# Type alias for policy functions
PolicyFn = type(baseline_policy)
