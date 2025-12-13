"""
Configuration and type definitions for the tree growth simulation.

This module defines all constants, state representations, and configuration
for the differentiable tree growth simulator.

State Vector:
    E: Energy store
    W: Water store
    N: Nutrient store
    R: Root biomass
    T: Trunk/wood biomass
    S: Shoot biomass
    L: Leaf biomass
    F: Flower biomass

All quantities are nonnegative and continuous.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class TreeState(NamedTuple):
    """
    Complete state of a tree at a given timestep.

    All values are JAX arrays (scalars) and must be nonnegative.

    Note: `water` is internal plant water (xylem/cells), while `soil_water`
    is the available water in the soil that roots can access.

    Reproductive cycle: flowers → fruit → seeds
    - Flowers are the reproductive investment (policy allocates here)
    - Fruit accumulates from mature flowers over time
    - Fruit can be damaged by stress (wind, drought)
    - Seeds = integral of fruit over season
    """

    energy: Array  # E: Energy store
    water: Array  # W: Internal water store (plant tissue)
    nutrients: Array  # N: Nutrient store
    roots: Array  # R: Root biomass
    trunk: Array  # T: Trunk/wood biomass
    shoots: Array  # S: Shoot biomass
    leaves: Array  # L: Leaf biomass
    flowers: Array  # F: Flower biomass
    fruit: Array  # Q: Fruit/developing seeds (accumulated from mature flowers)
    soil_water: Array  # SW: Soil water reservoir

    @classmethod
    def initial(cls, energy: float = 1.0, soil_water: float = 0.5) -> "TreeState":
        """Create initial state for a seed with given energy.

        Start with enough leaves/roots to bootstrap photosynthesis.
        A seed germinates with cotyledon leaves, initial root structure,
        and a small hypocotyl (stem) for water transport.

        Args:
            energy: Starting energy reserve
            soil_water: Initial soil water level (depends on climate)
        """
        eps = 1e-4  # Small but nonzero for numerical stability
        return cls(
            energy=jnp.array(energy),
            water=jnp.array(0.3),
            nutrients=jnp.array(0.3),
            roots=jnp.array(0.1),  # Initial root system
            trunk=jnp.array(0.05),  # Hypocotyl for water transport
            shoots=jnp.array(0.05),
            leaves=jnp.array(0.2),  # Cotyledon leaves for initial photosynthesis
            flowers=jnp.array(eps),
            fruit=jnp.array(0.0),  # No fruit initially
            soil_water=jnp.array(soil_water),  # Soil water reservoir
        )

    def is_valid(self) -> Array:
        """Check that all state values are nonnegative and finite."""
        all_nonneg = jnp.all(
            jnp.array(
                [
                    self.energy >= 0,
                    self.water >= 0,
                    self.nutrients >= 0,
                    self.roots >= 0,
                    self.trunk >= 0,
                    self.shoots >= 0,
                    self.leaves >= 0,
                    self.flowers >= 0,
                    self.fruit >= 0,
                    self.soil_water >= 0,
                ]
            )
        )
        all_finite = jnp.all(
            jnp.isfinite(
                jnp.array(
                    [
                        self.energy,
                        self.water,
                        self.nutrients,
                        self.roots,
                        self.trunk,
                        self.shoots,
                        self.leaves,
                        self.flowers,
                        self.fruit,
                        self.soil_water,
                    ]
                )
            )
        )
        return jnp.logical_and(all_nonneg, all_finite)

    def total_biomass(self) -> Array:
        """Total biomass across all compartments (excluding fruit which is reproductive)."""
        return self.roots + self.trunk + self.shoots + self.leaves + self.flowers


class Allocation(NamedTuple):
    """
    Resource allocation fractions (must sum to 1).

    Each value represents the fraction of available energy
    to allocate to that compartment.
    """

    roots: Array
    trunk: Array
    shoots: Array
    leaves: Array
    flowers: Array

    def is_valid(self) -> Array:
        """Check allocations are valid (nonnegative, sum to 1)."""
        total = self.roots + self.trunk + self.shoots + self.leaves + self.flowers
        all_nonneg = jnp.all(
            jnp.array(
                [
                    self.roots >= 0,
                    self.trunk >= 0,
                    self.shoots >= 0,
                    self.leaves >= 0,
                    self.flowers >= 0,
                ]
            )
        )
        sums_to_one = jnp.isclose(total, 1.0, atol=1e-5)
        return jnp.logical_and(all_nonneg, sums_to_one)


@dataclass(frozen=True)
class StressParams:
    """
    Parameters for a sinusoidal stress signal.

    signal(t) = offset + amplitude * sin(frequency * t + phase)

    The signal is then clipped to [0, 1] range.
    """

    offset: float  # D: baseline value
    amplitude: float  # A: swing magnitude
    frequency: float  # ω: radians per day
    phase: float  # φ: phase shift

    def __post_init__(self) -> None:
        if self.amplitude < 0:
            raise ValueError("Amplitude must be nonnegative")


@dataclass(frozen=True)
class ClimateConfig:
    """Configuration for environmental stress signals."""

    light: StressParams
    moisture: StressParams
    wind: StressParams

    @classmethod
    def mild(cls) -> "ClimateConfig":
        """A mild climate with moderate, slow-varying stressors."""
        return cls(
            light=StressParams(offset=0.7, amplitude=0.2, frequency=0.1, phase=0.0),
            moisture=StressParams(
                offset=0.6, amplitude=0.15, frequency=0.08, phase=1.0
            ),
            wind=StressParams(offset=0.2, amplitude=0.1, frequency=0.15, phase=0.5),
        )

    @classmethod
    def droughty(cls) -> "ClimateConfig":
        """A drought-prone climate with low, variable moisture."""
        return cls(
            light=StressParams(offset=0.8, amplitude=0.15, frequency=0.1, phase=0.0),
            moisture=StressParams(offset=0.3, amplitude=0.2, frequency=0.12, phase=0.0),
            wind=StressParams(offset=0.3, amplitude=0.15, frequency=0.1, phase=0.0),
        )

    @classmethod
    def windy(cls) -> "ClimateConfig":
        """A windy climate with frequent strong gusts."""
        return cls(
            light=StressParams(offset=0.6, amplitude=0.2, frequency=0.1, phase=0.0),
            moisture=StressParams(
                offset=0.5, amplitude=0.15, frequency=0.08, phase=0.0
            ),
            wind=StressParams(offset=0.5, amplitude=0.3, frequency=0.2, phase=0.0),
        )


@dataclass(frozen=True)
class SimConfig:
    """
    Complete simulation configuration.

    Contains all constants for the differentiable tree growth model.
    Values chosen to produce reasonable dynamics over ~100 day seasons.
    """

    # Simulation parameters
    num_days: int = 100
    seed_energy: float = 1.0
    investment_rate: float = (
        0.3  # Maximum fraction of energy to invest in growth each day
    )
    investment_energy_threshold: float = (
        0.3  # Energy level where investment rate is 50%
    )
    investment_steepness: float = 5.0  # How sharply investment gates at low energy

    # Photosynthesis parameters
    p_max: float = 0.5  # Maximum photosynthesis rate per unit leaf
    # K values chosen so typical state values live around K (gradient health)
    # For saturation f(x)=x/(x+K), gradient = K/(x+K)^2, max at x=0 is 1/K
    # K=0.2-0.5 keeps max gradients around 2-5, which is manageable
    k_light: float = 0.3  # Light half-saturation (light in [0,1])
    k_water: float = 0.2  # Water half-saturation
    k_nutrient: float = 0.2  # Nutrient half-saturation
    k_leaf: float = 1.5  # Leaf area extinction coefficient (Beer-Lambert self-shading)

    # Resource consumption parameters
    # Water and nutrients are consumed during photosynthesis and growth
    c_water_photo: float = 0.1  # Water consumed per unit photosynthesis
    c_nutrient_photo: float = 0.05  # Nutrient consumed per unit photosynthesis
    c_water_growth: float = 0.2  # Water consumed per unit biomass growth
    c_nutrient_growth: float = 0.15  # Nutrient consumed per unit biomass growth

    # Transport bottleneck parameters
    # Water delivery is limited by trunk capacity (xylem/phloem)
    # W_delivered = min(W, kappa * T^beta)
    kappa_transport: float = 2.0  # Transport capacity coefficient
    beta_transport: float = 0.7  # Transport capacity exponent (< 1 for sublinear)

    # Root uptake parameters
    u_water_max: float = 0.3  # Maximum water uptake rate
    u_nutrient_max: float = 0.2  # Maximum nutrient uptake rate
    k_root: float = 0.5  # Root half-saturation constant

    # Moisture optimum (inverted-U response)
    # Too dry: reduced uptake (drought stress)
    # Too wet: reduced uptake (root rot / anoxia)
    # Sigma widened so sub-optimal moisture is survivable (was 0.25, then 0.35)
    moisture_optimum: float = 0.6  # Optimal moisture level
    moisture_sigma: float = 0.45  # Width of optimal moisture band

    # Maintenance costs (per unit biomass per day)
    m_root: float = 0.01
    m_trunk: float = 0.005  # Wood is cheap to maintain
    m_shoot: float = 0.02
    m_leaf: float = 0.03  # Leaves are expensive
    m_flower: float = 0.08  # Flowers are VERY expensive (doubled from 0.04)
    # Higher flower maintenance creates real cost to camping in flowers.
    # If flowers don't hurt, optimizer will stay there forever.

    # Growth efficiency (energy to biomass conversion)
    eta_root: float = 0.8
    eta_trunk: float = 0.6  # Wood is expensive to build
    eta_shoot: float = 0.9
    eta_leaf: float = 0.85
    eta_flower: float = 0.7

    # Structural constraint parameters
    c_leaf: float = 1.0  # Load contribution of leaves
    c_shoot: float = 0.5  # Load contribution of shoots
    c_flower: float = 0.8  # Load contribution of flowers
    c_trunk: float = 2.0  # Capacity contribution of trunk

    # Trunk capacity exponent: Capacity = c_trunk * trunk^gamma
    # Design choice - this controls the "bonsai vs exuberant canopy" tradeoff:
    #   gamma < 1: Diminishing returns to wood (wood procrastination)
    #   gamma = 1: Linear scaling (balanced)
    #   gamma > 1: Superlinear returns (investing in wood unlocks big canopy)
    # For learned policies, gamma >= 1 helps wood investment emerge naturally.
    gamma: float = 1.0

    # Structural penalty: Energy drain = penalty * softplus(load - capacity)
    # Keep this small! Too high makes policy "coward" (avoid all canopy growth).
    # Rule of thumb: in a healthy rollout, penalty should be < 20% of photosynthesis.
    structural_penalty: float = 0.05

    # Shoot-leaf capacity constraint
    # Shoots are the scaffolding for leaves - you can't grow leaves without branches.
    # leaf_capacity = k_shoot_leaf * shoots
    # When leaves > capacity, leaf growth efficiency drops via sigmoid penalty.
    # This creates the dynamic: need shoots to hold leaves, need leaves to photosynthesize.
    # Lower k = more shoots needed. At k=2, half your canopy biomass must be shoots.
    k_shoot_leaf: float = 2.0  # Each unit of shoot supports 2 units of leaves
    leaf_crowding_steepness: float = 5.0  # How sharply growth penalty ramps up
    leaf_crowding_floor: float = 0.1  # Minimum growth efficiency even when crowded

    # Shoot-flower capacity constraint
    # Flowers also grow on branches - you can't have flowers without shoots to hold them.
    # flower_capacity = k_shoot_flower * shoots
    # This makes shoots doubly valuable: they hold both leaves AND flowers.
    k_shoot_flower: float = 3.0  # Each unit of shoot supports 3 units of flowers
    flower_crowding_steepness: float = 5.0  # How sharply growth penalty ramps up
    flower_crowding_floor: float = 0.1  # Minimum growth efficiency even when crowded

    # Wind damage parameters
    wind_threshold: float = 0.5  # Wind level where damage starts
    # Steepness: 8-15 recommended. Higher = vanishing gradients outside narrow band.
    wind_steepness: float = 8.0
    # Damage coefficients tuned so adverse climates stress but don't destroy trees
    # Rule: (1 - alpha * max_damage * 0.5)^100 should leave ~10-20% survival
    alpha_shoot: float = 0.12  # Shoot damage coefficient (was 0.3)
    alpha_leaf: float = 0.08  # Leaf damage coefficient (was 0.2)
    alpha_flower: float = 0.25  # Flower damage coefficient (was 0.7, still vulnerable)
    max_wind_damage: float = 0.5  # Maximum base damage per day (before protection)

    # Wood protection against wind
    # Trunk provides structural support that reduces wind damage
    # protection = max_protection * (1 - exp(-k_protection * trunk))
    k_wind_protection: float = 1.0  # How quickly trunk provides protection
    max_wind_protection: float = 0.7  # Maximum protection (70% damage reduction)

    # Flower-specific wind protection (trunk shields irreversible investment)
    # Flowers get MUCH more protection from trunk than leaves/shoots
    # This creates the "oak vs reeds" bifurcation: invest in trunk → protect flowers
    k_flower_protection: float = 2.0  # Faster saturation (trunk helps flowers more)
    max_flower_protection: float = 0.9  # Up to 90% damage reduction for flowers

    # Transpiration parameters (water loss from leaves)
    # Transpiration = transp_rate * leaves * light
    # This makes drought actually force root investment
    transpiration_rate: float = 0.15  # Water lost per unit leaf per unit light

    # Stomatal closure parameters (A)
    # When internal water is low, stomata close to conserve water
    # This reduces photosynthesis but prevents desiccation
    stomatal_threshold: float = 0.25  # Water level where stomata start closing
    stomatal_steepness: float = 10.0  # How sharply stomata respond
    stomatal_min: float = 0.1  # Minimum conductance (keeps some gradient)

    # Drought leaf damage parameters (B)
    # When water drops critically low, leaves die back (senescence)
    # This is the "nuclear option" for water conservation
    # Tuned so drought stresses but doesn't destroy: (1 - 0.10 * 0.5)^100 ≈ 0.6%
    drought_critical: float = 0.15  # Water level where damage starts
    drought_steepness: float = 15.0  # How sharply damage ramps up
    drought_max_damage: float = 0.10  # Maximum daily leaf loss (was 0.25)

    # Flowering and seed production
    flowering_maturity: float = 0.4  # Season progress (0-1) before flowers can grow
    flowering_trunk_threshold: float = 0.2  # Minimum trunk to support flowers
    flowering_leaves_threshold: float = 0.3  # Minimum leaves to support flowers
    flowering_gate_steepness: float = 10.0  # How sharply phenology gates kick in
    seed_energy_threshold: float = 0.5  # Minimum energy to produce seeds
    seed_conversion: float = 10.0  # Seeds per unit fruit (was flower) biomass

    # Fruit dynamics (flowers → fruit → seeds)
    # This creates the "bonsai economics" risk/reward dynamic:
    # - Flowers convert to fruit only when tree is mature enough
    # - Fruit accumulates over time but can be damaged by stress
    # - Seeds = integral of fruit (rewards sustained reproduction, not dumps)
    #
    # Maturity gate: m(t) = σ(k_T(T-T_0)) · σ(k_L(L-L_0)) · σ(k_R(R-R_0))
    # Fruit rate: dQ/dt = α · m(t) · (F/(F+K)) · f_L · f_W · f_I - ρ · Q - damage
    fruit_maturity_trunk: float = 0.15  # T_0: trunk threshold for mature fruit
    fruit_maturity_leaves: float = 0.25  # L_0: leaves threshold for mature fruit
    fruit_maturity_roots: float = 0.15  # R_0: roots threshold for mature fruit
    fruit_maturity_steepness: float = 10.0  # k: steepness of maturity gates
    fruit_conversion_rate: float = 0.5  # α: base flowers → fruit rate (increased to offset saturation)
    fruit_decay_rate: float = 0.02  # ρ: natural fruit decay (ripening/falling)

    # Saturating flower→fruit conversion (prevents "all-in flowers" being optimal)
    # fruit_contribution = F / (F + K_F)
    # At K_F=0.3: 0.3 flowers → 50% max, 0.9 flowers → 75% max
    # This makes 80% flower allocation only marginally better than 40%
    fruit_saturation_k: float = 0.3  # K_F: half-saturation for flower→fruit

    # Resource gates for fruit formation (must maintain infrastructure)
    # fruit_gain *= f_L(leaves) · f_W(water) · f_I(light)
    # This forces policy to keep leaves/roots online during reproduction
    fruit_leaf_k: float = 0.3  # Leaf half-saturation for fruit support
    fruit_water_k: float = 0.2  # Water half-saturation for fruit support
    fruit_light_k: float = 0.3  # Light half-saturation for fruit support

    # Fruit stress damage (the "gambling on reproduction" mechanic)
    # Wind and drought can destroy developing fruit
    # This creates real risk: early flowering in unstable weather = lost investment
    fruit_wind_vulnerability: float = 0.4  # How much wind damages fruit (0-1)
    fruit_drought_vulnerability: float = 0.3  # How much drought damages fruit
    fruit_drought_threshold: float = 0.2  # Water level below which fruit suffers

    # Store decay (resources decay slightly each day)
    water_decay: float = 0.05
    nutrient_decay: float = 0.03

    # Soil water reservoir parameters
    # The soil is a finite reservoir that moisture replenishes and roots deplete.
    # This makes drought *actually* create water scarcity.
    soil_water_capacity: float = 2.0  # Maximum soil water the reservoir can hold
    soil_recharge_rate: float = 0.4  # How fast moisture replenishes soil (per day)
    soil_drain_rate: float = 0.02  # Natural drainage/evaporation from soil
    # Root uptake now pulls from soil_water instead of being directly gated by moisture
    # This means: drought → low soil recharge → soil depletes → roots can't find water
