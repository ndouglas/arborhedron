"""
Biological surrogate functions for the tree growth simulation.

These are smooth, differentiable approximations of biological processes.
All functions are designed to be compatible with JAX autodiff.

Key design principles:
- All outputs bounded appropriately (usually [0, 1] for efficiencies)
- Smooth transitions (no hard discontinuities)
- Michaelis-Menten kinetics for saturation effects
- Sigmoid gates for threshold behaviors
"""

from typing import TYPE_CHECKING, Union

import jax.numpy as jnp
from jax import Array
from jax.nn import sigmoid, softplus

# Type alias for values that can be either JAX arrays or Python floats
Scalar = Union[Array, float]

if TYPE_CHECKING:
    from sim.config import TreeState


def saturation(x: Array, k: float, eps: float = 1e-6) -> Array:
    """
    Michaelis-Menten saturation function with numerical stability.

    Models diminishing returns: f(x) = x / (x + K + ε)

    The epsilon prevents gradient blow-up near x→0 when K is small.

    Properties:
    - f(0) ≈ 0
    - f(K) ≈ 0.5 (half-saturation)
    - f(∞) → 1

    Args:
        x: Input value (nonnegative)
        k: Half-saturation constant (positive, recommend k ≥ 0.2)
        eps: Small constant for numerical stability

    Returns:
        Value in [0, 1]
    """
    return x / (x + k + eps)


def temperature_window(
    temp: Array,
    t_min: float,
    t_max: float,
    steepness: float = 1.0,
) -> Array:
    """
    Soft temperature efficiency window using double sigmoid.

    Efficiency is high between t_min and t_max, low outside.

    f(T) = σ(k(T - T_min)) · σ(k(T_max - T))

    Note: For cleaner gradients, consider using temperature_gaussian instead.

    Args:
        temp: Temperature value
        t_min: Lower bound of optimal range
        t_max: Upper bound of optimal range
        steepness: How sharply efficiency drops outside window

    Returns:
        Efficiency in [0, 1]
    """
    lower_gate = sigmoid(steepness * (temp - t_min))
    upper_gate = sigmoid(steepness * (t_max - temp))
    return lower_gate * upper_gate


def temperature_gaussian(
    temp: Array,
    t_opt: float,
    sigma: float = 5.0,
) -> Array:
    """
    Gaussian temperature efficiency function.

    f(T) = exp(-(T - T_opt)² / (2σ²))

    This provides cleaner gradients than the double-sigmoid window,
    avoiding the "product of small numbers" problem when multiplied
    with other efficiency factors.

    Args:
        temp: Temperature value
        t_opt: Optimal temperature
        sigma: Standard deviation (width of the efficiency band)

    Returns:
        Efficiency in (0, 1]
    """
    return jnp.exp(-((temp - t_opt) ** 2) / (2 * sigma**2))


def wind_damage(
    wind: Array,
    threshold: float,
    steepness: float = 8.0,
) -> Array:
    """
    Sigmoid damage function for wind stress.

    Damage ramps up smoothly once wind exceeds threshold.

    d(v) = σ(k(v - v₀))

    Note: steepness of 8-15 recommended. Higher values cause vanishing
    gradients outside a narrow band. Lower values make causality mushy.

    Args:
        wind: Wind speed (normalized, typically [0, 1])
        threshold: Wind level where damage reaches 50%
        steepness: How sharply damage ramps up (recommend 8-15)

    Returns:
        Damage factor in [0, 1]
    """
    return sigmoid(steepness * (wind - threshold))


def structural_penalty(load: Scalar, capacity: Scalar) -> Array:
    """
    Soft penalty for unsupported structural load.

    Uses baseline-subtracted softplus for smooth transition:
    penalty = softplus(load - capacity) - softplus(-capacity)

    This ensures penalty = 0 when load = 0, avoiding an always-on
    energy tax that would distort early-game dynamics.

    When load = 0: penalty = 0 (exactly)
    When load < capacity: penalty ≈ 0
    When load > capacity: penalty ≈ (load - capacity)

    Args:
        load: Total load from canopy (leaves + shoots + flowers)
        capacity: Structural capacity from trunk

    Returns:
        Penalty value (nonnegative, zero at load=0)
    """
    # Baseline subtraction ensures penalty(0, c) = 0 for any capacity
    raw_penalty = softplus(jnp.array(load) - jnp.array(capacity))
    baseline = softplus(-jnp.array(capacity))
    return raw_penalty - baseline


def photosynthesis(
    leaves: Scalar,
    light: Scalar,
    water: Scalar,
    nutrients: Scalar,
    p_max: float,
    k_light: float,
    k_water: float,
    k_nutrient: float,
) -> Array:
    """
    Combined photosynthesis rate with multiple limiting factors.

    P = L · P_max · f_I(I) · f_W(W) · f_N(N)

    Uses multiplicative Michaelis-Menten model where each resource
    can independently limit the rate.

    Args:
        leaves: Leaf biomass
        light: Light availability [0, 1]
        water: Water store
        nutrients: Nutrient store
        p_max: Maximum photosynthesis rate per unit leaf
        k_light: Light half-saturation constant
        k_water: Water half-saturation constant
        k_nutrient: Nutrient half-saturation constant

    Returns:
        Energy produced this timestep
    """
    light_eff = saturation(jnp.array(light), k_light)
    water_eff = saturation(jnp.array(water), k_water)
    nutrient_eff = saturation(jnp.array(nutrients), k_nutrient)

    return jnp.array(leaves) * p_max * light_eff * water_eff * nutrient_eff


def root_uptake(
    roots: Scalar,
    moisture: Scalar,
    u_water_max: float,
    u_nutrient_max: float,
    k_root: float,
) -> tuple[Array, Array]:
    """
    Root uptake of water and nutrients.

    U_W = U_W_max · f_R(R) · f_M(M)
    U_N = U_N_max · f_R(R) · f_M(M)

    Both water and nutrient uptake depend on root biomass and soil moisture.

    Args:
        roots: Root biomass
        moisture: Soil moisture [0, 1]
        u_water_max: Maximum water uptake rate
        u_nutrient_max: Maximum nutrient uptake rate
        k_root: Root half-saturation constant

    Returns:
        Tuple of (water_uptake, nutrient_uptake)
    """
    root_eff = saturation(jnp.array(roots), k_root)
    moisture_eff = saturation(jnp.array(moisture), k_root)  # Reuse k_root for moisture

    water_uptake = u_water_max * root_eff * moisture_eff
    nutrient_uptake = u_nutrient_max * root_eff * moisture_eff

    return water_uptake, nutrient_uptake


def maintenance_cost(
    state: "TreeState",
    m_root: float,
    m_trunk: float,
    m_shoot: float,
    m_leaf: float,
    m_flower: float,
) -> Array:
    """
    Total maintenance cost for all biomass compartments.

    Cost = m_R·R + m_T·T + m_S·S + m_L·L + m_F·F

    Each compartment has a per-unit maintenance cost that represents
    the energy needed to keep that tissue alive.

    Args:
        state: Current tree state
        m_root: Maintenance cost per unit root
        m_trunk: Maintenance cost per unit trunk
        m_shoot: Maintenance cost per unit shoot
        m_leaf: Maintenance cost per unit leaf
        m_flower: Maintenance cost per unit flower

    Returns:
        Total energy cost for maintenance
    """
    return (
        m_root * state.roots
        + m_trunk * state.trunk
        + m_shoot * state.shoots
        + m_leaf * state.leaves
        + m_flower * state.flowers
    )


def seed_production(
    flowers: Scalar,
    energy: Scalar,
    energy_threshold: float,
    conversion: float,
) -> Array:
    """
    Calculate seed production from flowers.

    Seeds = conversion · F · σ(E - E_threshold)

    Seeds are only produced if there's sufficient energy.
    The sigmoid gates production based on energy availability.

    Args:
        flowers: Flower biomass
        energy: Current energy store
        energy_threshold: Minimum energy to produce seeds
        conversion: Seeds per unit flower biomass

    Returns:
        Number of seeds produced
    """
    energy_gate = sigmoid(10.0 * (jnp.array(energy) - energy_threshold))
    return conversion * jnp.array(flowers) * energy_gate


def compute_load(
    leaves: Scalar,
    shoots: Scalar,
    flowers: Scalar,
    c_leaf: float,
    c_shoot: float,
    c_flower: float,
) -> Array:
    """
    Compute total structural load from canopy.

    Load = c_L·L + c_S·S + c_F·F

    Args:
        leaves: Leaf biomass
        shoots: Shoot biomass
        flowers: Flower biomass
        c_leaf: Load coefficient for leaves
        c_shoot: Load coefficient for shoots
        c_flower: Load coefficient for flowers

    Returns:
        Total load
    """
    return c_leaf * jnp.array(leaves) + c_shoot * jnp.array(shoots) + c_flower * jnp.array(flowers)


def compute_capacity(trunk: Scalar, c_trunk: float, gamma: float) -> Array:
    """
    Compute structural capacity from trunk.

    Capacity = c_T · T^γ

    Sublinear scaling (γ < 1) means you need disproportionately
    more trunk to support larger canopies.

    Args:
        trunk: Trunk biomass
        c_trunk: Capacity coefficient
        gamma: Scaling exponent (typically < 1)

    Returns:
        Structural capacity
    """
    # Use jnp.power for differentiability and handle trunk=0 case
    return c_trunk * jnp.power(jnp.maximum(jnp.array(trunk), 1e-8), gamma)


def growth_efficiency(
    water: Scalar,
    nutrients: Scalar,
    base_efficiency: float,
    k_water: float,
    k_nutrient: float,
) -> Array:
    """
    Compute growth efficiency based on resource availability.

    η = η₀ · f_W(W) · f_N(N)

    Growth is limited by both water and nutrients.

    Args:
        water: Water store
        nutrients: Nutrient store
        base_efficiency: Base efficiency (e.g., eta_root, eta_leaf)
        k_water: Water half-saturation
        k_nutrient: Nutrient half-saturation

    Returns:
        Effective growth efficiency
    """
    water_eff = saturation(jnp.array(water), k_water)
    nutrient_eff = saturation(jnp.array(nutrients), k_nutrient)
    return base_efficiency * water_eff * nutrient_eff
