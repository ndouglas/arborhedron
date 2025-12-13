"""
Tree growth dynamics - the core simulation step.

This module implements the state update for one day of tree growth.
The update follows this sequence:

1. Root uptake of water and nutrients
2. Photosynthesis (energy production)
3. Maintenance costs
4. Resource allocation and growth
5. Wind damage
6. Structural penalty
7. Resource decay
8. Clamping to nonnegative values

All operations are differentiable for gradient-based optimization.
"""

import jax.numpy as jnp
from jax import Array

from sim import surrogates
from sim.config import Allocation, SimConfig, TreeState


def step(
    state: TreeState,
    allocation: Allocation,
    light: float,
    moisture: float,
    wind: float,
    config: SimConfig,
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

    # 1. Root uptake of water and nutrients
    water_uptake, nutrient_uptake = surrogates.root_uptake(
        roots=roots,
        moisture=moisture,
        u_water_max=config.u_water_max,
        u_nutrient_max=config.u_nutrient_max,
        k_root=config.k_root,
    )
    water = water + water_uptake
    nutrients = nutrients + nutrient_uptake

    # 2. Photosynthesis
    photo_energy = surrogates.photosynthesis(
        leaves=leaves,
        light=light,
        water=water,
        nutrients=nutrients,
        p_max=config.p_max,
        k_light=config.k_light,
        k_water=config.k_water,
        k_nutrient=config.k_nutrient,
    )
    energy = energy + photo_energy

    # 3. Maintenance costs
    maintenance = surrogates.maintenance_cost(
        state,
        m_root=config.m_root,
        m_trunk=config.m_trunk,
        m_shoot=config.m_shoot,
        m_leaf=config.m_leaf,
        m_flower=config.m_flower,
    )
    energy = energy - maintenance

    # 4. Resource allocation and growth
    # Compute available energy for growth (can't invest more than we have)
    available_energy = jnp.maximum(energy, 0.0)

    # Growth efficiency depends on water and nutrient availability
    # Each compartment has a base efficiency modified by resource availability
    def compute_growth(
        alloc_frac: Array, base_eta: float
    ) -> tuple[Array, Array]:
        """Compute growth and energy cost for a compartment."""
        efficiency = surrogates.growth_efficiency(
            water=water,
            nutrients=nutrients,
            base_efficiency=base_eta,
            k_water=config.k_water,
            k_nutrient=config.k_nutrient,
        )
        # Energy invested = fraction of available energy
        invested = alloc_frac * available_energy
        # Biomass gained = invested * efficiency
        growth = invested * efficiency
        return growth, invested

    root_growth, root_cost = compute_growth(allocation.roots, config.eta_root)
    trunk_growth, trunk_cost = compute_growth(allocation.trunk, config.eta_trunk)
    shoot_growth, shoot_cost = compute_growth(allocation.shoots, config.eta_shoot)
    leaf_growth, leaf_cost = compute_growth(allocation.leaves, config.eta_leaf)
    flower_growth, flower_cost = compute_growth(allocation.flowers, config.eta_flower)

    # Apply growth
    roots = roots + root_growth
    trunk = trunk + trunk_growth
    shoots = shoots + shoot_growth
    leaves = leaves + leaf_growth
    flowers = flowers + flower_growth

    # Deduct growth costs from energy
    total_cost = root_cost + trunk_cost + shoot_cost + leaf_cost + flower_cost
    energy = energy - total_cost

    # 5. Wind damage to tender growth (shoots and leaves)
    damage_factor = surrogates.wind_damage(
        jnp.array(wind),
        threshold=config.wind_threshold,
        steepness=config.wind_steepness,
    )

    # Shoots are damaged by wind
    shoot_damage = damage_factor * config.alpha_shoot
    shoots = shoots * (1.0 - shoot_damage)

    # Leaves are damaged by wind
    leaf_damage = damage_factor * config.alpha_leaf
    leaves = leaves * (1.0 - leaf_damage)

    # 6. Structural penalty
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

    # 7. Resource decay (water and nutrients decay slightly)
    water = water * (1.0 - config.water_decay)
    nutrients = nutrients * (1.0 - config.nutrient_decay)

    # 8. Clamp all values to nonnegative
    # Use jnp.maximum which has well-defined gradients
    energy = jnp.maximum(energy, 0.0)
    water = jnp.maximum(water, 0.0)
    nutrients = jnp.maximum(nutrients, 0.0)
    roots = jnp.maximum(roots, 0.0)
    trunk = jnp.maximum(trunk, 0.0)
    shoots = jnp.maximum(shoots, 0.0)
    leaves = jnp.maximum(leaves, 0.0)
    flowers = jnp.maximum(flowers, 0.0)

    return TreeState(
        energy=energy,
        water=water,
        nutrients=nutrients,
        roots=roots,
        trunk=trunk,
        shoots=shoots,
        leaves=leaves,
        flowers=flowers,
    )


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
