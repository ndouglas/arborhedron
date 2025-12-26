"""
Tree growth dynamics - the core simulation step.

This module implements the state update for one day of tree growth.
The update follows this sequence:

1. Root uptake of water and nutrients
2. Transport bottleneck (trunk limits water delivery)
3. Photosynthesis (energy production with self-shading)
   3b. Stomatal closure (internal water gates photosynthesis)
4. Water/nutrient consumption for photosynthesis
   4b. Transpiration (leaves lose water)
5. Maintenance costs
6. Resource allocation and growth
7. Water/nutrient consumption for growth
8. Wind damage
   8b. Drought leaf senescence (low water damages leaves)
9. Structural penalty
10. Resource decay
11. Clamping to nonnegative values

Key stabilization mechanisms:
- Self-shading (Beer-Lambert): prevents runaway leaf growth
- Resource consumption: water/nutrients are used during photosynthesis and growth
- Transport bottleneck: trunk limits water delivery to canopy
- Stomatal closure: low water reduces photosynthesis (prevents desiccation)
- Drought senescence: critically low water damages leaves (forces root investment)
- Transpiration: leaves consume water proportional to light (drought pressure)

All operations are differentiable for gradient-based optimization.
"""

import jax.numpy as jnp
from jax import Array
from jax.nn import sigmoid as jax_sigmoid

from sim import surrogates
from sim.config import Allocation, SimConfig, TreeState


def step(
    state: TreeState,
    allocation: Allocation,
    light: float,
    moisture: float,
    wind: float,
    config: SimConfig,
    day: int = 50,
) -> TreeState:
    """
    Perform one day of tree growth simulation.

    Args:
        state: Current tree state
        allocation: Resource allocation fractions (from policy)
        light: Light availability [0, 1]
        moisture: Soil moisture [0, 1]
        wind: Wind speed [0, 1]
        config: Simulation configuration
        day: Current day (for maturity gating)

    Returns:
        New tree state after one day
    """
    # Unpack state for cleaner code
    energy = state.energy
    water = state.water
    nutrients = state.nutrients
    roots = state.roots
    trunk = state.trunk
    shoots = state.shoots
    leaves = state.leaves
    flowers = state.flowers
    fruit = state.fruit
    soil_water = state.soil_water

    # 0. Soil water reservoir dynamics
    # Moisture replenishes soil, roots deplete it - this creates REAL water scarcity
    recharge = surrogates.soil_water_recharge(
        moisture=moisture,
        soil_water=soil_water,
        soil_capacity=config.soil_water_capacity,
        recharge_rate=config.soil_recharge_rate,
    )
    soil_water = soil_water + recharge

    # 1. Root uptake from soil reservoir (creates actual water scarcity)
    # Includes inverted-U moisture response (both drought AND waterlogging hurt roots)
    water_uptake, nutrient_uptake, water_extracted = surrogates.root_uptake_from_soil(
        roots=roots,
        soil_water=soil_water,
        moisture=moisture,
        u_water_max=config.u_water_max,
        u_nutrient_max=config.u_nutrient_max,
        k_root=config.k_root,
        m_opt=config.moisture_optimum,
        m_sigma=config.moisture_sigma,
    )
    # Deplete soil water and add to internal water
    soil_water = soil_water - water_extracted
    water = water + water_uptake
    nutrients = nutrients + nutrient_uptake

    # 2. Transport bottleneck: trunk limits water delivery to canopy
    # This creates a natural constraint requiring trunk investment
    transport_cap = surrogates.transport_capacity(
        trunk=trunk,
        kappa=config.kappa_transport,
        beta=config.beta_transport,
    )
    # Water available for photosynthesis is limited by transport
    water_available = jnp.minimum(water, transport_cap)

    # 3. Photosynthesis (uses transport-limited water)
    # First compute raw photosynthetic potential
    photo_potential = surrogates.photosynthesis(
        leaves=leaves,
        light=light,
        water=water_available,
        nutrients=nutrients,
        p_max=config.p_max,
        k_light=config.k_light,
        k_water=config.k_water,
        k_nutrient=config.k_nutrient,
        k_leaf=config.k_leaf,
    )

    # 3b. Stomatal closure: when internal water is low, stomata close
    # This reduces photosynthesis but conserves water (prevents desiccation)
    stomatal_gate = surrogates.stomatal_conductance(
        water=water,
        water_threshold=config.stomatal_threshold,
        steepness=config.stomatal_steepness,
        min_conductance=config.stomatal_min,
    )
    photo_energy = photo_potential * stomatal_gate
    energy = energy + photo_energy

    # 4. Consume water and nutrients for photosynthesis
    # Photosynthesis requires water (transpiration) and nutrients
    water = water - config.c_water_photo * photo_energy
    nutrients = nutrients - config.c_nutrient_photo * photo_energy

    # 4b. Transpiration: leaves lose water proportional to leaf area and light
    # This creates the "drought forces root investment" dynamic
    transpiration = config.transpiration_rate * leaves * light
    water = water - transpiration

    # 5. Maintenance costs
    maintenance = surrogates.maintenance_cost(
        state,
        m_root=config.m_root,
        m_trunk=config.m_trunk,
        m_shoot=config.m_shoot,
        m_leaf=config.m_leaf,
        m_flower=config.m_flower,
    )
    energy = energy - maintenance

    # 6. Resource allocation and growth
    # Investment is gated by energy level to prevent "suicide investing"
    # At low energy, reduce investment to conserve reserves
    available_energy = jnp.maximum(energy, 0.0)
    investment_gate = jax_sigmoid(
        config.investment_steepness
        * (available_energy - config.investment_energy_threshold)
    )
    effective_investment_rate = config.investment_rate * investment_gate
    energy_to_invest = effective_investment_rate * available_energy

    # Growth efficiency depends on water and nutrient availability
    # Each compartment has a base efficiency modified by resource availability
    def compute_growth(alloc_frac: Array, base_eta: float) -> tuple[Array, Array]:
        """Compute growth and energy cost for a compartment."""
        efficiency = surrogates.growth_efficiency(
            water=water,
            nutrients=nutrients,
            base_efficiency=base_eta,
            k_water=config.k_water,
            k_nutrient=config.k_nutrient,
        )
        # Energy invested = fraction of investment budget (not total energy)
        invested = alloc_frac * energy_to_invest
        # Biomass gained = invested * efficiency
        growth = invested * efficiency
        return growth, invested

    # Flower gating: prevent flowering before maturity, without trunk, or without leaves
    # This prevents "suicide flowering" exploits where policy dumps all energy into flowers early
    # Uses three gates that must ALL be satisfied:
    # 1. Maturity (season progress): can't flower too early
    # 2. Trunk threshold: need structural support
    # 3. Leaves threshold: need photosynthetic capacity (can't flower without leaves)
    progress = day / config.num_days
    maturity_gate = jax_sigmoid(
        config.flowering_gate_steepness * (progress - config.flowering_maturity)
    )
    biomass_gate = surrogates.maturity_gate(
        trunk=trunk,
        leaves=leaves,
        trunk_threshold=config.flowering_trunk_threshold,
        leaves_threshold=config.flowering_leaves_threshold,
        steepness=config.flowering_gate_steepness,
    )
    flower_gate = maturity_gate * biomass_gate

    # Gated flower allocation - any blocked flower energy goes nowhere (wasted)
    # This creates a strong incentive to NOT allocate to flowers until ready
    gated_flower_alloc = allocation.flowers * flower_gate

    # Leaf crowding penalty: shoots are scaffolding for leaves
    # When leaves exceed shoot capacity, leaf growth efficiency drops
    # This creates the dynamic: need shoots → can grow leaves → can photosynthesize
    leaf_crowding = surrogates.leaf_crowding_penalty(
        leaves=leaves,
        shoots=shoots,
        k_shoot_leaf=config.k_shoot_leaf,
        steepness=config.leaf_crowding_steepness,
        floor=config.leaf_crowding_floor,
    )
    # Apply crowding penalty to leaf efficiency
    effective_leaf_eta = config.eta_leaf * leaf_crowding

    # Flower crowding penalty: shoots are also scaffolding for flowers
    # When flowers exceed shoot capacity, flower growth efficiency drops
    # Combined with leaf crowding, this makes shoots doubly valuable
    flower_crowding = surrogates.flower_crowding_penalty(
        flowers=flowers,
        shoots=shoots,
        k_shoot_flower=config.k_shoot_flower,
        steepness=config.flower_crowding_steepness,
        floor=config.flower_crowding_floor,
    )
    # Apply crowding penalty to flower efficiency
    effective_flower_eta = config.eta_flower * flower_crowding

    root_growth, root_cost = compute_growth(allocation.roots, config.eta_root)
    trunk_growth, trunk_cost = compute_growth(allocation.trunk, config.eta_trunk)
    shoot_growth, shoot_cost = compute_growth(allocation.shoots, config.eta_shoot)
    leaf_growth, leaf_cost = compute_growth(allocation.leaves, effective_leaf_eta)
    flower_growth, flower_cost = compute_growth(
        gated_flower_alloc, effective_flower_eta
    )

    # Apply growth
    roots = roots + root_growth
    trunk = trunk + trunk_growth
    shoots = shoots + shoot_growth
    leaves = leaves + leaf_growth
    flowers = flowers + flower_growth

    # Deduct only the invested energy (not 100%)
    total_invested = root_cost + trunk_cost + shoot_cost + leaf_cost + flower_cost
    energy = energy - total_invested

    # 7. Consume water and nutrients for growth
    # Building new biomass requires water and nutrients
    total_growth = (
        root_growth + trunk_growth + shoot_growth + leaf_growth + flower_growth
    )
    water = water - config.c_water_growth * total_growth
    nutrients = nutrients - config.c_nutrient_growth * total_growth

    # 8. Wind damage to tender growth (shoots and leaves)
    # Uses effective_wind_damage which includes:
    # - Sigmoid damage ramp with threshold
    # - Damage cap (max_wind_damage) to prevent instant wipeout
    # - Wood protection (trunk provides structural support)
    effective_damage = surrogates.effective_wind_damage(
        wind=wind,
        trunk=trunk,
        threshold=config.wind_threshold,
        steepness=config.wind_steepness,
        max_damage=config.max_wind_damage,
        k_protection=config.k_wind_protection,
        max_protection=config.max_wind_protection,
    )

    # Apply compartment-specific damage coefficients
    shoot_damage = effective_damage * config.alpha_shoot
    shoots = shoots * (1.0 - shoot_damage)

    leaf_damage = effective_damage * config.alpha_leaf
    leaves = leaves * (1.0 - leaf_damage)

    # 8c. Flower wind damage with enhanced trunk protection
    # Flowers are tender (high alpha) but get MUCH more protection from trunk
    # This creates the "oak vs reeds" bifurcation:
    # - Without trunk: flowers get destroyed → no point in flowering under wind
    # - With trunk: flowers protected → trunk investment enables reproduction
    flower_wind_dmg = surrogates.flower_wind_damage(
        wind=wind,
        trunk=trunk,
        threshold=config.wind_threshold,
        steepness=config.wind_steepness,
        max_damage=config.max_wind_damage,
        alpha_flower=config.alpha_flower,
        k_protection=config.k_flower_protection,
        max_protection=config.max_flower_protection,
    )
    flowers = flowers * (1.0 - flower_wind_dmg)

    # 8d. Drought leaf senescence: when water is critically low, leaves die back
    # This is the "nuclear option" - the tree sacrifices leaves to survive drought
    drought_leaf_damage = surrogates.drought_damage(
        water=water,
        water_critical=config.drought_critical,
        steepness=config.drought_steepness,
        max_damage=config.drought_max_damage,
    )
    leaves = leaves * (1.0 - drought_leaf_damage)

    # 8e. Fruit dynamics: flowers → fruit → seeds
    # This creates the "bonsai economics" risk/reward:
    # - Flowers convert to fruit only when tree is mature
    # - Fruit conversion is SATURATING (diminishing returns to more flowers)
    # - Fruit conversion requires RESOURCES (leaves, water, light)
    # - Fruit can be destroyed by stress (wind, drought)
    # - Seeds = integral of fruit over season

    # Compute maturity gate: m(t) = σ(k(T-T0)) · σ(k(L-L0)) · σ(k(R-R0))
    # Only mature trees can convert flowers to fruit effectively
    maturity = surrogates.fruit_maturity_gate(
        trunk=trunk,
        leaves=leaves,
        roots=roots,
        trunk_threshold=config.fruit_maturity_trunk,
        leaves_threshold=config.fruit_maturity_leaves,
        roots_threshold=config.fruit_maturity_roots,
        steepness=config.fruit_maturity_steepness,
    )

    # Fruit accumulation with SATURATION and RESOURCE GATING:
    # dQ/dt = α · m(t) · (F/(F+K)) · f_L(L) · f_W(W) · f_I(I) - decay - damage
    #
    # Key insight: This forces policy to:
    # 1. Not go "all-in" on flowers (saturation makes 80% barely better than 40%)
    # 2. Keep infrastructure alive (need leaves/water/light to cash in on flowers)
    fruit_gain = surrogates.fruit_formation_rate(
        flowers=flowers,
        leaves=leaves,
        water=water,
        light=light,
        maturity=maturity,
        base_rate=config.fruit_conversion_rate,
        saturation_k=config.fruit_saturation_k,
        k_leaf=config.fruit_leaf_k,
        k_water=config.fruit_water_k,
        k_light=config.fruit_light_k,
    )

    # Natural fruit decay (ripening, falling, etc.)
    fruit_decay = config.fruit_decay_rate * fruit

    # Fruit stress damage: wind and drought can destroy developing fruit
    fruit_damage_rate = surrogates.fruit_stress_damage(
        wind=wind,
        water=water,
        wind_vulnerability=config.fruit_wind_vulnerability,
        drought_vulnerability=config.fruit_drought_vulnerability,
        drought_threshold=config.fruit_drought_threshold,
        wind_threshold=config.wind_threshold,
        wind_steepness=config.wind_steepness,
    )
    fruit_stress_loss = fruit_damage_rate * fruit

    # Update fruit
    fruit = fruit + fruit_gain - fruit_decay - fruit_stress_loss

    # 9. Structural penalty
    load = surrogates.compute_load(
        leaves=leaves,
        shoots=shoots,
        flowers=flowers,
        c_leaf=config.c_leaf,
        c_shoot=config.c_shoot,
        c_flower=config.c_flower,
    )
    capacity = surrogates.compute_capacity(
        trunk=trunk,
        c_trunk=config.c_trunk,
        gamma=config.gamma,
    )
    structural_drain = surrogates.structural_penalty(load, capacity)
    energy = energy - config.structural_penalty * structural_drain

    # 10. Resource decay (water and nutrients decay slightly)
    water = water * (1.0 - config.water_decay)
    nutrients = nutrients * (1.0 - config.nutrient_decay)

    # 10b. Soil water drainage (natural loss to deep soil / evaporation)
    soil_water = soil_water * (1.0 - config.soil_drain_rate)

    # 11. Clamp all values to nonnegative
    # Use jnp.maximum which has well-defined gradients
    energy = jnp.maximum(energy, 0.0)
    water = jnp.maximum(water, 0.0)
    nutrients = jnp.maximum(nutrients, 0.0)
    roots = jnp.maximum(roots, 0.0)
    trunk = jnp.maximum(trunk, 0.0)
    shoots = jnp.maximum(shoots, 0.0)
    leaves = jnp.maximum(leaves, 0.0)
    flowers = jnp.maximum(flowers, 0.0)
    fruit = jnp.maximum(fruit, 0.0)
    soil_water = jnp.maximum(soil_water, 0.0)

    return TreeState(
        energy=energy,
        water=water,
        nutrients=nutrients,
        roots=roots,
        trunk=trunk,
        shoots=shoots,
        leaves=leaves,
        flowers=flowers,
        fruit=fruit,
        soil_water=soil_water,
    )


def diagnose_energy_budget(
    state: TreeState,
    allocation: Allocation,  # noqa: ARG001
    light: float,
    moisture: float,
    wind: float,  # noqa: ARG001
    config: SimConfig,
) -> dict[str, float]:
    """
    Diagnostic function to inspect energy and resource budget components.

    Returns a dictionary with all flows for debugging.
    """
    # Soil water recharge
    recharge = surrogates.soil_water_recharge(
        moisture=moisture,
        soil_water=state.soil_water,
        soil_capacity=config.soil_water_capacity,
        recharge_rate=config.soil_recharge_rate,
    )
    soil_water_after_recharge = state.soil_water + recharge

    # Root uptake from soil reservoir (with inverted-U moisture response)
    water_uptake, nutrient_uptake, water_extracted = surrogates.root_uptake_from_soil(
        roots=state.roots,
        soil_water=soil_water_after_recharge,
        moisture=moisture,
        u_water_max=config.u_water_max,
        u_nutrient_max=config.u_nutrient_max,
        k_root=config.k_root,
        m_opt=config.moisture_optimum,
        m_sigma=config.moisture_sigma,
    )
    water = state.water + water_uptake
    nutrients = state.nutrients + nutrient_uptake

    # Transport bottleneck
    transport_cap = surrogates.transport_capacity(
        trunk=state.trunk,
        kappa=config.kappa_transport,
        beta=config.beta_transport,
    )
    water_available = min(float(water), float(transport_cap))

    # Photosynthesis (with transport-limited water and self-shading)
    photo_energy = surrogates.photosynthesis(
        leaves=state.leaves,
        light=light,
        water=water_available,
        nutrients=nutrients,
        p_max=config.p_max,
        k_light=config.k_light,
        k_water=config.k_water,
        k_nutrient=config.k_nutrient,
        k_leaf=config.k_leaf,
    )

    # Water/nutrient consumption for photosynthesis
    water_consumed_photo = config.c_water_photo * float(photo_energy)
    nutrient_consumed_photo = config.c_nutrient_photo * float(photo_energy)

    # Maintenance
    maintenance = surrogates.maintenance_cost(
        state,
        m_root=config.m_root,
        m_trunk=config.m_trunk,
        m_shoot=config.m_shoot,
        m_leaf=config.m_leaf,
        m_flower=config.m_flower,
    )

    # Investment
    energy_after_maintenance = state.energy + photo_energy - maintenance
    available = max(float(energy_after_maintenance), 0.0)
    investment = config.investment_rate * available

    # Structural penalty
    load = surrogates.compute_load(
        leaves=state.leaves,
        shoots=state.shoots,
        flowers=state.flowers,
        c_leaf=config.c_leaf,
        c_shoot=config.c_shoot,
        c_flower=config.c_flower,
    )
    capacity = surrogates.compute_capacity(
        trunk=state.trunk,
        c_trunk=config.c_trunk,
        gamma=config.gamma,
    )
    struct_penalty = float(surrogates.structural_penalty(load, capacity))

    # Self-shading efficiency (for debugging)
    leaf_eff = float(surrogates.leaf_area_efficiency(state.leaves, config.k_leaf))

    # Leaf crowding (shoot capacity constraint)
    leaf_crowding = surrogates.leaf_crowding_penalty(
        leaves=state.leaves,
        shoots=state.shoots,
        k_shoot_leaf=config.k_shoot_leaf,
        steepness=config.leaf_crowding_steepness,
        floor=config.leaf_crowding_floor,
    )
    leaf_capacity = config.k_shoot_leaf * float(state.shoots)

    # Flower crowding (shoot capacity constraint)
    flower_crowding = surrogates.flower_crowding_penalty(
        flowers=state.flowers,
        shoots=state.shoots,
        k_shoot_flower=config.k_shoot_flower,
        steepness=config.flower_crowding_steepness,
        floor=config.flower_crowding_floor,
    )
    flower_capacity = config.k_shoot_flower * float(state.shoots)

    # Drought mechanics
    stomatal = surrogates.stomatal_conductance(
        water=state.water,
        water_threshold=config.stomatal_threshold,
        steepness=config.stomatal_steepness,
        min_conductance=config.stomatal_min,
    )
    transpiration = config.transpiration_rate * float(state.leaves) * light
    drought_dmg = surrogates.drought_damage(
        water=state.water,
        water_critical=config.drought_critical,
        steepness=config.drought_steepness,
        max_damage=config.drought_max_damage,
    )

    return {
        "photosynthesis": float(photo_energy),
        "maintenance": float(maintenance),
        "investment": float(investment),
        "structural_penalty": float(struct_penalty) * config.structural_penalty,
        "energy_before": float(state.energy),
        "net_energy_flow": float(photo_energy) - float(maintenance) - float(investment),
        "water_uptake": float(water_uptake),
        "water_available": water_available,
        "transport_capacity": float(transport_cap),
        "water_consumed_photo": water_consumed_photo,
        "nutrient_consumed_photo": nutrient_consumed_photo,
        "leaf_efficiency": leaf_eff,
        "structural_load": float(load),
        "structural_capacity": float(capacity),
        # Drought mechanics
        "stomatal_conductance": float(stomatal),
        "transpiration": transpiration,
        "drought_damage": float(drought_dmg),
        "water_internal": float(state.water),
        # Soil water reservoir
        "soil_water": float(state.soil_water),
        "soil_recharge": float(recharge),
        "soil_water_extracted": float(water_extracted),
        # Leaf crowding (shoot capacity constraint)
        "leaf_crowding_penalty": float(leaf_crowding),
        "leaf_capacity": leaf_capacity,
        # Flower crowding (shoot capacity constraint)
        "flower_crowding_penalty": float(flower_crowding),
        "flower_capacity": flower_capacity,
    }


def compute_seeds(state: TreeState, config: SimConfig) -> Array:
    """
    Compute seed production from final state.

    Seeds = conversion * flowers * sigmoid(energy - threshold)

    Args:
        state: Final tree state
        config: Simulation configuration

    Returns:
        Number of seeds produced
    """
    return surrogates.seed_production(
        flowers=state.flowers,
        energy=state.energy,
        energy_threshold=config.seed_energy_threshold,
        conversion=config.seed_conversion,
    )


def compute_seeds_integral(
    flower_integral: Array,
    final_energy: Array,
    config: SimConfig,
) -> Array:
    """
    Compute seed production from integrated flower-days.

    DEPRECATED: Use compute_seeds_from_fruit instead.
    This is kept for backwards compatibility but fruit-based seeds are preferred.

    Seeds = conversion * flower_integral * sigmoid(energy - threshold)

    Args:
        flower_integral: Sum of flower biomass over season (flower-days)
        final_energy: Final energy level (gates seed conversion)
        config: Simulation configuration

    Returns:
        Number of seeds produced
    """
    normalized_integral = flower_integral / config.num_days
    return surrogates.seed_production(
        flowers=normalized_integral,
        energy=final_energy,
        energy_threshold=config.seed_energy_threshold,
        conversion=config.seed_conversion,
    )


def compute_seeds_from_fruit(
    fruit_integral: Array,
    final_energy: Array,
    config: SimConfig,
) -> Array:
    """
    Compute seed production from integrated fruit (the proper method).

    Seeds = conversion * (fruit_integral / num_days) * energy_gate

    This is the correct reward function:
    - Flowers must convert to fruit (requires maturity)
    - Fruit can be damaged by stress (gambling mechanic)
    - Seeds = integral of fruit (rewards sustained, protected reproduction)

    The fruit_integral is the sum of fruit biomass over all days.
    Dividing by num_days normalizes to "average fruit" for interpretability.

    Args:
        fruit_integral: Sum of fruit biomass over season (fruit-days)
        final_energy: Final energy level (gates seed viability)
        config: Simulation configuration

    Returns:
        Number of seeds produced
    """
    # Normalize by season length
    normalized_integral = fruit_integral / config.num_days

    return surrogates.seed_production(
        flowers=normalized_integral,  # reusing flowers param, it's just biomass
        energy=final_energy,
        energy_threshold=config.seed_energy_threshold,
        conversion=config.seed_conversion,
    )
