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
from jax.nn import sigmoid, softmax

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

    Strategy phases with SMOOTH transitions:
    - Early (0-20%): Build roots and leaves for resource gathering
    - Mid (20-60%): Expand with shoots, start building trunk
    - Late (60-100%): Consolidate trunk, invest in flowers

    Uses sigmoid interpolation between phases to avoid gradient discontinuities.

    Args:
        state: Current tree state (unused in baseline, but API-compatible)
        day: Current day (0-indexed)
        num_days: Total season length
        wind: Current wind level (for adaptive response)

    Returns:
        Allocation for this timestep
    """
    progress = day / num_days

    # Phase logits
    early_logits = jnp.array([2.0, 0.5, 1.0, 2.5, -2.0])  # Foundation
    mid_logits = jnp.array([1.5, 1.5, 1.5, 1.5, 0.0])  # Expansion
    late_logits = jnp.array([0.5, 2.0, 0.5, 1.0, 2.0])  # Consolidation

    # Smooth transitions using sigmoid blending
    # k controls transition sharpness (15 gives smooth ~10% transition band)
    k = 15.0
    # blend1: 0->1 around 0.2 (early -> mid transition)
    # blend2: 0->1 around 0.6 (mid -> late transition)
    blend1 = sigmoid(k * (progress - 0.2))
    blend2 = sigmoid(k * (progress - 0.6))

    # Interpolate: early * (1-blend1) + mid * (blend1 - blend2) + late * blend2
    # At progress=0:   blend1≈0, blend2≈0 → early
    # At progress=0.4: blend1≈1, blend2≈0 → mid
    # At progress=0.8: blend1≈1, blend2≈1 → late
    base_logits = (
        early_logits * (1.0 - blend1)
        + mid_logits * (blend1 - blend2)
        + late_logits * blend2
    )

    # Wind adaptation: increase trunk allocation when windy
    wind_bonus = jnp.array([0.0, wind * 2.0, -wind * 0.5, -wind * 0.5, 0.0])
    logits = base_logits + wind_bonus

    return softmax_allocation(logits)


def growth_focused_policy(
    state: TreeState,  # noqa: ARG001
    day: int,  # noqa: ARG001
    num_days: int,  # noqa: ARG001
    wind: float = 0.0,  # noqa: ARG001
) -> Allocation:
    """
    Policy focused on maximizing growth (roots + leaves).

    Good for benign environments with low stress.

    Args:
        state: Current tree state (API-compatible, unused)
        day: Current day (API-compatible, unused)
        num_days: Total season length (API-compatible, unused)
        wind: Current wind level (API-compatible, unused)

    Returns:
        Growth-focused allocation
    """
    # Heavy investment in roots and leaves
    logits = jnp.array([2.0, 0.0, 1.0, 2.5, 0.0])
    return softmax_allocation(logits)


def defensive_policy(
    state: TreeState,  # noqa: ARG001
    day: int,  # noqa: ARG001
    num_days: int,  # noqa: ARG001
    wind: float = 0.0,  # noqa: ARG001
) -> Allocation:
    """
    Policy focused on structural resilience.

    Good for harsh environments with wind/stress.

    Args:
        state: Current tree state (API-compatible, unused)
        day: Current day (API-compatible, unused)
        num_days: Total season length (API-compatible, unused)
        wind: Current wind level (API-compatible, unused)

    Returns:
        Defensive allocation
    """
    # Heavy investment in trunk for structural support
    logits = jnp.array([1.5, 3.0, 0.5, 1.0, 0.5])
    return softmax_allocation(logits)


def reproduction_policy(
    state: TreeState,  # noqa: ARG001
    day: int,  # noqa: ARG001
    num_days: int,  # noqa: ARG001
    wind: float = 0.0,  # noqa: ARG001
) -> Allocation:
    """
    Policy focused on reproduction (flowers).

    Good for late-season all-in on seeds.

    Args:
        state: Current tree state (API-compatible, unused)
        day: Current day (API-compatible, unused)
        num_days: Total season length (API-compatible, unused)
        wind: Current wind level (API-compatible, unused)

    Returns:
        Reproduction-focused allocation
    """
    # Heavy investment in flowers
    logits = jnp.array([0.5, 1.0, 0.5, 1.0, 3.0])
    return softmax_allocation(logits)


def phased_policy(
    state: TreeState,  # noqa: ARG001
    day: int,
    num_days: int,
    wind: float = 0.0,  # noqa: ARG001
    early_logits: Array | None = None,
    mid_logits: Array | None = None,
    late_logits: Array | None = None,
    transition_steepness: float = 15.0,
) -> Allocation:
    """
    Parameterized phased policy for optimization.

    This policy can be optimized by learning the logits for each phase.
    Uses smooth sigmoid blending between phases.

    Args:
        state: Current tree state (API-compatible, unused)
        day: Current day
        num_days: Total season length
        wind: Current wind level (API-compatible, unused)
        early_logits: Logits for early phase (days 0-20%)
        mid_logits: Logits for mid phase (days 20-60%)
        late_logits: Logits for late phase (days 60-100%)
        transition_steepness: Controls smoothness of phase transitions

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

    # Smooth transitions using sigmoid blending (same as baseline_policy)
    k = transition_steepness
    blend1 = sigmoid(k * (progress - 0.2))
    blend2 = sigmoid(k * (progress - 0.6))

    # Interpolate: early * (1-blend1) + mid * (blend1 - blend2) + late * blend2
    logits = (
        early_logits * (1.0 - blend1)
        + mid_logits * (blend1 - blend2)
        + late_logits * blend2
    )

    return softmax_allocation(logits)


# =============================================================================
# LEARNED POLICY PRIMITIVES (Week 2)
# =============================================================================

# Domain-informed priors from 03C analysis:
# - Early: leaves-focused for photosynthesis
# - Late: flowers-focused for reproduction
PRIOR_EARLY_LOGITS = jnp.array([0.3, -0.5, 0.0, 1.0, -1.5])  # R, T, S, L, F
PRIOR_LATE_LOGITS = jnp.array([-1.0, -1.0, -1.0, -0.5, 2.0])  # Strong flower dominance


def linear_interpolation_policy(
    early_logits: Array,
    late_logits: Array,
    day: int,
    num_days: int,
) -> Allocation:
    """
    Simple linear interpolation between early and late logits.

    This is the minimal learnable policy architecture:
    logits(t) = early_logits * (1 - t/T) + late_logits * (t/T)

    Args:
        early_logits: Allocation logits at day 0
        late_logits: Allocation logits at final day
        day: Current day
        num_days: Total season length

    Returns:
        Allocation for this timestep
    """
    progress = day / num_days
    logits = early_logits * (1.0 - progress) + late_logits * progress
    return softmax_allocation(logits)


def prior_delta_policy(
    delta_early: Array,
    delta_late: Array,
    day: int,
    num_days: int,
) -> Allocation:
    """
    Prior + delta parameterization for stable optimization.

    Uses domain-informed prior (from 03C analysis) with learnable deviations.
    This avoids the "L2 fights commitment" pathology where regularization
    prevents the sharp late-season flower transition needed for high fitness.

    logits(t) = prior(t) + delta(t)

    Regularize ||delta||^2, not ||logits||^2.

    Args:
        delta_early: Deviation from PRIOR_EARLY_LOGITS
        delta_late: Deviation from PRIOR_LATE_LOGITS
        day: Current day
        num_days: Total season length

    Returns:
        Allocation for this timestep
    """
    progress = day / num_days
    early_logits = PRIOR_EARLY_LOGITS + delta_early
    late_logits = PRIOR_LATE_LOGITS + delta_late
    logits = early_logits * (1.0 - progress) + late_logits * progress
    return softmax_allocation(logits)


# Type alias for policy functions
PolicyFn = type(baseline_policy)
