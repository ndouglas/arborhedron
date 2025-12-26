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


# =============================================================================
# NEURAL POLICY (Week 2/3)
# =============================================================================

import equinox as eqx
from jax import random as jr


class NeuralPolicy(eqx.Module):
    """
    Neural network policy for tree growth allocation.

    Takes tree state + environment as input, outputs allocation fractions.

    Architecture:
    - Input: [state (8), progress (1), environment (3)] = 12 features
    - Hidden: 2 layers of 32 units with tanh activation
    - Output: 5 logits → softmax → allocation

    The architecture is deliberately simple - we're testing whether
    autodiff can learn temporal strategies, not building a massive model.
    """

    layers: list

    def __init__(self, key: Array, hidden_size: int = 32, num_hidden: int = 2):
        """
        Initialize neural policy with random weights.

        Args:
            key: JAX random key for initialization
            hidden_size: Number of units per hidden layer
            num_hidden: Number of hidden layers
        """
        input_size = 12  # 8 state + 1 progress + 3 environment
        output_size = 5  # allocation logits

        keys = jr.split(key, num_hidden + 1)

        layers = []
        in_size = input_size
        for i in range(num_hidden):
            layers.append(eqx.nn.Linear(in_size, hidden_size, key=keys[i]))
            in_size = hidden_size
        layers.append(eqx.nn.Linear(in_size, output_size, key=keys[-1]))

        self.layers = layers

    def __call__(self, features: Array) -> Array:
        """
        Forward pass: features → logits.

        Args:
            features: Input features [batch, 12] or [12]

        Returns:
            Allocation logits [batch, 5] or [5]
        """
        x = features
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))  # tanh for smooth gradients
        logits = self.layers[-1](x)
        return logits


def make_policy_features(
    state: TreeState,
    day: int,
    num_days: int,
    light: float,
    moisture: float,
    wind: float,
) -> Array:
    """
    Extract input features for neural policy.

    Features are lightly normalized to roughly [-1, 2] range.

    Args:
        state: Current tree state
        day: Current day
        num_days: Total season length
        light: Current light level [0, 1]
        moisture: Current moisture level [0, 1]
        wind: Current wind level [0, 1]

    Returns:
        Feature vector of shape [12]
    """
    # State features (divide by typical max values for rough normalization)
    # Energy/water/nutrients typically in [0, 1]
    # Biomass typically in [0, 3]
    state_features = jnp.array(
        [
            state.energy,  # [0, 1] typically
            state.water,  # [0, 1] typically
            state.nutrients,  # [0, 1] typically
            state.roots / 2.0,  # normalize to ~[0, 1.5]
            state.trunk / 2.0,
            state.shoots / 2.0,
            state.leaves / 2.0,
            state.flowers / 2.0,
        ]
    )

    # Progress through season [0, 1]
    progress = jnp.array([day / num_days])

    # Environment features (already in [0, 1])
    env_features = jnp.array([light, moisture, wind])

    return jnp.concatenate([state_features, progress, env_features])


def apply_neural_policy(
    policy: NeuralPolicy,
    state: TreeState,
    day: int,
    num_days: int,
    light: float,
    moisture: float,
    wind: float,
) -> Allocation:
    """
    Apply neural policy to get allocation.

    This is the main interface for using the neural policy.

    Args:
        policy: Trained NeuralPolicy module
        state: Current tree state
        day: Current day
        num_days: Total season length
        light: Current light level
        moisture: Current moisture level
        wind: Current wind level

    Returns:
        Allocation for this timestep
    """
    features = make_policy_features(state, day, num_days, light, moisture, wind)
    logits = policy(features)
    return softmax_allocation(logits)


def make_neural_policy_fn(
    policy: NeuralPolicy,
    num_days: int,
    light_fn,
    moisture_fn,
    wind_fn,
):
    """
    Create a standard policy function from a NeuralPolicy.

    This wraps the neural policy to match the signature expected by run_season.

    Args:
        policy: NeuralPolicy module
        num_days: Total season length
        light_fn: Function day → light value
        moisture_fn: Function day → moisture value
        wind_fn: Function day → wind value

    Returns:
        Policy function with signature (state, day, num_days, wind) → Allocation
    """

    def policy_fn(
        state: TreeState, day: int, num_days_arg: int, wind: float = 0.0
    ) -> Allocation:
        # Get environment values for this day
        light = light_fn(day)
        moisture = moisture_fn(day)
        wind_val = wind_fn(day)
        return apply_neural_policy(
            policy, state, day, num_days, light, moisture, wind_val
        )

    return policy_fn


# =============================================================================
# SMART BASELINE (for fair comparison with neural policy)
# =============================================================================


def smart_baseline_policy(
    state: TreeState,
    day: int,
    num_days: int,
    wind: float = 0.0,
    # Maturity thresholds (should match config)
    trunk_threshold: float = 0.15,
    leaves_threshold: float = 0.25,
    roots_threshold: float = 0.15,
    # Energy threshold for safety
    energy_threshold: float = 0.3,
) -> Allocation:
    """
    Smart baseline policy with maturity awareness and resource maintenance.

    This is a FAIR comparison baseline that:
    1. Uses the same maturity gate logic as the fruit dynamics
    2. Maintains leaves/roots above thresholds during reproduction
    3. Adapts to wind by reducing flower allocation when windy
    4. Doesn't go "all-in" on flowers (knows about diminishing returns)

    Strategy:
    - Build infrastructure until mature (trunk, leaves, roots above thresholds)
    - Once mature AND energy is healthy, allocate to flowers
    - BUT keep leaves and roots alive (50% infrastructure / 50% flowers)
    - Reduce flower allocation when wind is high

    This baseline should produce more seeds than the naive baseline because:
    - It knows about maturity requirements
    - It maintains infrastructure during reproduction
    - It doesn't waste flowers when windy

    If neural policy can't beat THIS baseline, it's not learning anything useful.

    Args:
        state: Current tree state
        day: Current day
        num_days: Total season length
        wind: Current wind level [0, 1]
        trunk_threshold: Minimum trunk for fruit maturity
        leaves_threshold: Minimum leaves for fruit maturity
        roots_threshold: Minimum roots for fruit maturity
        energy_threshold: Minimum energy before flowering

    Returns:
        Allocation for this timestep
    """
    progress = day / num_days

    # Check maturity requirements
    trunk_ready = float(state.trunk) >= trunk_threshold
    leaves_ready = float(state.leaves) >= leaves_threshold
    roots_ready = float(state.roots) >= roots_threshold
    energy_healthy = float(state.energy) >= energy_threshold

    # All conditions met for flowering
    is_mature = trunk_ready and leaves_ready and roots_ready

    if not is_mature:
        # BUILD PHASE: Focus on infrastructure
        # Priority: leaves (photosynthesis) > roots (water) > trunk (maturity) > shoots
        if not leaves_ready:
            # Leaves first for energy production
            logits = jnp.array([1.0, 0.5, 1.5, 2.5, -2.0])  # Heavy leaves, some shoots
        elif not roots_ready:
            # Roots for water
            logits = jnp.array([2.5, 0.5, 1.0, 1.5, -2.0])  # Heavy roots
        else:
            # Need trunk for maturity
            logits = jnp.array([1.0, 2.5, 1.0, 1.5, -2.0])  # Heavy trunk

    elif not energy_healthy:
        # LOW ENERGY: Rebuild resources before flowering
        logits = jnp.array([2.0, 0.5, 1.0, 2.0, -1.0])  # Roots + leaves for recovery

    else:
        # REPRODUCTIVE PHASE: Flowers + infrastructure maintenance
        # Key insight: with saturating fruit conversion, going 100% flowers
        # is barely better than 50% flowers, so maintain infrastructure

        # Base: balanced flowering with maintenance
        # ~40% flowers, ~30% leaves, ~20% roots, ~10% trunk
        base_logits = jnp.array([1.0, 0.5, 0.5, 1.5, 2.0])

        # Wind adaptation: reduce flower allocation when windy
        # Flowers are vulnerable to wind, and we need trunk protection
        wind_penalty = wind * 1.5  # Reduce flowers
        wind_bonus = wind * 1.0  # Boost trunk
        wind_adjustment = jnp.array([0.0, wind_bonus, 0.0, 0.0, -wind_penalty])

        # Late season push: increase flower allocation as season ends
        # (less time left = less to lose from wind damage)
        late_push = jnp.maximum(0.0, (progress - 0.7) * 5.0)  # Ramp after day 70%
        late_adjustment = jnp.array([0.0, 0.0, 0.0, -late_push * 0.5, late_push])

        logits = base_logits + wind_adjustment + late_adjustment

    return softmax_allocation(logits)
