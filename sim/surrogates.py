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

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from jax.nn import sigmoid, softplus

# Type alias for values that can be either JAX arrays or Python floats
Scalar = Array | float

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
    max_damage: float = 0.5,
) -> Array:
    """
    Sigmoid damage function for wind stress with damage cap.

    Damage ramps up smoothly once wind exceeds threshold, but is capped
    to prevent instant wipeout and preserve gradient signal.

    d(v) = max_damage · σ(k(v - v₀))

    The cap ensures that even in extreme wind, some biomass survives
    each timestep. This:
    1. Preserves gradient signal for learning
    2. Makes wind a continuous hazard, not a binary death sentence
    3. Allows wood investment to be a meaningful defense

    Note: steepness of 8-15 recommended. Higher values cause vanishing
    gradients outside a narrow band.

    Args:
        wind: Wind speed (normalized, typically [0, 1])
        threshold: Wind level where damage reaches 50% of max
        steepness: How sharply damage ramps up (recommend 8-15)
        max_damage: Maximum damage fraction per timestep (default 0.5)

    Returns:
        Damage factor in [0, max_damage]
    """
    raw_damage = sigmoid(steepness * (wind - threshold))
    return max_damage * raw_damage


def wood_protection(
    trunk: Scalar,
    k_protection: float = 1.0,
    max_protection: float = 0.7,
) -> Array:
    """
    Compute protection factor from trunk/wood against wind damage.

    Wood provides structural support that reduces wind damage to
    tender growth (shoots and leaves).

    protection = max_protection · (1 - exp(-k · T))

    Properties:
    - protection(0) = 0 (no trunk, no protection)
    - protection → max_protection as T → ∞
    - More trunk = less effective wind damage

    Args:
        trunk: Trunk biomass
        k_protection: How quickly protection saturates with trunk
        max_protection: Maximum protection fraction (default 0.7 = 70% reduction)

    Returns:
        Protection factor in [0, max_protection]
    """
    t = jnp.array(trunk)
    return max_protection * (1.0 - jnp.exp(-k_protection * t))


def effective_wind_damage(
    wind: Scalar,
    trunk: Scalar,
    threshold: float = 0.5,
    steepness: float = 8.0,
    max_damage: float = 0.5,
    k_protection: float = 1.0,
    max_protection: float = 0.7,
) -> Array:
    """
    Compute effective wind damage after wood protection.

    effective_damage = base_damage · (1 - protection)

    This makes trunk investment a meaningful defense against wind:
    - More trunk = more protection
    - Even in high wind, damage is capped and can be mitigated

    Args:
        wind: Wind speed [0, 1]
        trunk: Trunk biomass (provides protection)
        threshold: Wind threshold for damage
        steepness: Sigmoid steepness
        max_damage: Maximum damage per timestep
        k_protection: Trunk protection rate
        max_protection: Maximum protection from trunk

    Returns:
        Effective damage factor after protection
    """
    base_damage = wind_damage(jnp.array(wind), threshold, steepness, max_damage)
    protection = wood_protection(trunk, k_protection, max_protection)
    return base_damage * (1.0 - protection)


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


def leaf_area_efficiency(leaves: Scalar, k_leaf: float) -> Array:
    """
    Self-shading efficiency using Beer-Lambert law.

    Models diminishing returns from leaf area due to self-shading:
    f(L) = 1 - exp(-k_L * L)

    Properties:
    - f(0) = 0 (no leaves, no photosynthesis)
    - f(L) → 1 as L → ∞ (saturates at high leaf area)
    - df/dL = k_L * exp(-k_L * L) (gradient decreases with more leaves)

    This prevents runaway leaf growth by making additional leaves
    less valuable than the first few.

    Args:
        leaves: Leaf biomass
        k_leaf: Light extinction coefficient (higher = faster saturation)

    Returns:
        Effective leaf area fraction in [0, 1)
    """
    return 1.0 - jnp.exp(-k_leaf * jnp.array(leaves))


def photosynthesis(
    leaves: Scalar,
    light: Scalar,
    water: Scalar,
    nutrients: Scalar,
    p_max: float,
    k_light: float,
    k_water: float,
    k_nutrient: float,
    k_leaf: float = 1.5,
    gradient_floor: float = 0.03,
) -> Array:
    """
    Combined photosynthesis rate with multiple limiting factors.

    P = P_max · f_L(L) · combined_efficiency

    where f_L uses Beer-Lambert self-shading:
        f_L(L) = 1 - exp(-k_leaf * L)

    and combined_efficiency uses a gradient-preserving floor:
        eff = eps + (1 - eps) · f_I(I) · f_W(W) · f_N(N)

    IMPORTANT: Pure multiplicative limiting (f_I * f_W * f_N) kills gradients
    when any factor is near 0 (common at seed stage). The floor ensures
    gradient signal flows even in resource-poor states, preventing the
    optimizer from getting stuck in "can't escape seed stage" local optima.

    The Beer-Lambert self-shading prevents runaway leaf growth by making
    additional leaves progressively less valuable (they shade lower leaves).

    Args:
        leaves: Leaf biomass
        light: Light availability [0, 1]
        water: Water store
        nutrients: Nutrient store
        p_max: Maximum photosynthesis rate per unit leaf
        k_light: Light half-saturation constant
        k_water: Water half-saturation constant
        k_nutrient: Nutrient half-saturation constant
        k_leaf: Leaf area extinction coefficient (higher = faster saturation)
        gradient_floor: Minimum efficiency floor (0.02-0.05 recommended)

    Returns:
        Energy produced this timestep
    """
    # Self-shading: more leaves have diminishing returns
    leaf_eff = leaf_area_efficiency(leaves, k_leaf)

    # Resource limiting factors
    light_eff = saturation(jnp.array(light), k_light)
    water_eff = saturation(jnp.array(water), k_water)
    nutrient_eff = saturation(jnp.array(nutrients), k_nutrient)

    # Gradient-preserving floor: prevents gradient death at low resources
    raw_product = light_eff * water_eff * nutrient_eff
    combined_eff = gradient_floor + (1.0 - gradient_floor) * raw_product

    return p_max * leaf_eff * combined_eff


def moisture_efficiency(
    moisture: Scalar,
    m_opt: float = 0.6,
    sigma: float = 0.25,
) -> Array:
    """
    Moisture efficiency with optimal range (inverted-U shape).

    Uses a Gaussian to create an optimum at m_opt:
        f(M) = exp(-(M - m_opt)² / (2σ²))

    This prevents "more moisture is always better" by penalizing:
    - Low moisture (drought stress)
    - High moisture (flooding/root rot)

    Args:
        moisture: Soil moisture [0, 1]
        m_opt: Optimal moisture level (default 0.6)
        sigma: Width of optimal band (default 0.25)

    Returns:
        Efficiency in (0, 1]
    """
    m = jnp.array(moisture)
    return jnp.exp(-((m - m_opt) ** 2) / (2 * sigma**2))


def root_uptake(
    roots: Scalar,
    moisture: Scalar,
    u_water_max: float,
    u_nutrient_max: float,
    k_root: float,
    m_opt: float = 0.6,
    m_sigma: float = 0.25,
) -> tuple[Array, Array]:
    """
    Root uptake of water and nutrients with optimal moisture.

    U_W = U_W_max · f_R(R) · f_M(M)
    U_N = U_N_max · f_R(R) · f_M(M)

    where f_M uses a Gaussian centered at m_opt to create an inverted-U:
    - Too dry: reduced uptake (drought)
    - Optimal: maximum uptake
    - Too wet: reduced uptake (root rot / anoxia)

    Args:
        roots: Root biomass
        moisture: Soil moisture [0, 1]
        u_water_max: Maximum water uptake rate
        u_nutrient_max: Maximum nutrient uptake rate
        k_root: Root half-saturation constant
        m_opt: Optimal moisture level
        m_sigma: Width of optimal moisture band

    Returns:
        Tuple of (water_uptake, nutrient_uptake)
    """
    root_eff = saturation(jnp.array(roots), k_root)
    # Inverted-U moisture response instead of monotonic
    moisture_eff = moisture_efficiency(moisture, m_opt, m_sigma)

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
    return (
        c_leaf * jnp.array(leaves)
        + c_shoot * jnp.array(shoots)
        + c_flower * jnp.array(flowers)
    )


def transport_capacity(trunk: Scalar, kappa: float, beta: float) -> Array:
    """
    Compute water transport capacity based on trunk (xylem/phloem).

    W_max = kappa * T^beta

    The trunk limits how much water can be delivered to leaves.
    This creates a natural constraint: you need trunk before you can
    support a large canopy.

    Args:
        trunk: Trunk biomass
        kappa: Transport capacity coefficient
        beta: Transport capacity exponent (typically < 1 for sublinear)

    Returns:
        Maximum water that can be delivered per timestep
    """
    return kappa * jnp.power(jnp.maximum(jnp.array(trunk), 1e-8), beta)


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


def water_stress(
    water: Scalar,
    water_threshold: float = 0.2,
    steepness: float = 10.0,
) -> Array:
    """
    Compute water stress level based on internal water store.

    stress = σ(k · (threshold - water))

    When water > threshold: stress ≈ 0
    When water < threshold: stress → 1

    This is the inverse of a "water sufficiency" gate.

    Args:
        water: Internal water store
        water_threshold: Water level where stress reaches 50%
        steepness: How sharply stress ramps up (recommend 8-15)

    Returns:
        Stress level in [0, 1]
    """
    return sigmoid(steepness * (water_threshold - jnp.array(water)))


def stomatal_conductance(
    water: Scalar,
    water_threshold: float = 0.2,
    steepness: float = 10.0,
    min_conductance: float = 0.1,
) -> Array:
    """
    Compute stomatal conductance based on internal water.

    When water is low, stomata close to conserve water, reducing
    photosynthesis but preventing desiccation.

    conductance = min + (1 - min) · σ(k · (water - threshold))

    When water > threshold: conductance → 1 (stomata open)
    When water < threshold: conductance → min (stomata closed)

    The minimum conductance prevents complete shutdown and preserves
    gradient signal.

    Args:
        water: Internal water store
        water_threshold: Water level where conductance reaches 50%
        steepness: How sharply conductance changes
        min_conductance: Minimum conductance when stomata closed (0.05-0.2)

    Returns:
        Conductance factor in [min_conductance, 1]
    """
    raw_conductance = sigmoid(steepness * (jnp.array(water) - water_threshold))
    return min_conductance + (1.0 - min_conductance) * raw_conductance


def drought_damage(
    water: Scalar,
    water_critical: float = 0.1,
    steepness: float = 15.0,
    max_damage: float = 0.3,
) -> Array:
    """
    Compute leaf damage from severe water stress (senescence).

    When water drops below critical threshold, leaves die back.
    This is the "nuclear option" - the tree sacrifices leaves to survive.

    damage = max_damage · σ(k · (threshold - water))

    When water > threshold: damage ≈ 0
    When water < threshold: damage → max_damage

    Args:
        water: Internal water store
        water_critical: Critical water level for damage onset
        steepness: How sharply damage ramps up (recommend 10-20)
        max_damage: Maximum daily leaf loss fraction (0.2-0.4)

    Returns:
        Damage factor in [0, max_damage]
    """
    raw_stress = sigmoid(steepness * (water_critical - jnp.array(water)))
    return max_damage * raw_stress


def growth_efficiency(
    water: Scalar,
    nutrients: Scalar,
    base_efficiency: float,
    k_water: float,
    k_nutrient: float,
    gradient_floor: float = 0.03,
) -> Array:
    """
    Compute growth efficiency based on resource availability.

    Uses gradient-preserving floor:
        η = η₀ · (eps + (1 - eps) · f_W(W) · f_N(N))

    Growth is limited by both water and nutrients, but the floor
    ensures gradient signal flows even at low resource levels.

    Args:
        water: Water store
        nutrients: Nutrient store
        base_efficiency: Base efficiency (e.g., eta_root, eta_leaf)
        k_water: Water half-saturation
        k_nutrient: Nutrient half-saturation
        gradient_floor: Minimum efficiency floor (0.02-0.05 recommended)

    Returns:
        Effective growth efficiency
    """
    water_eff = saturation(jnp.array(water), k_water)
    nutrient_eff = saturation(jnp.array(nutrients), k_nutrient)

    # Gradient-preserving floor
    raw_product = water_eff * nutrient_eff
    combined_eff = gradient_floor + (1.0 - gradient_floor) * raw_product

    return base_efficiency * combined_eff
