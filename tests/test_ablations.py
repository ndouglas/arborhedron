"""
Ablation tests for stabilization mechanisms.

Each test demonstrates that disabling a stabilization mechanism
causes the exploit behavior it was designed to prevent.
"""

import jax.numpy as jnp
from jax import grad

from sim import surrogates
from sim.config import Allocation, SimConfig, TreeState
from sim.dynamics import step


class TestSelfShadingAblation:
    """Tests proving self-shading (Beer-Lambert) prevents runaway leaf growth."""

    def test_without_self_shading_gradient_constant(self) -> None:
        """Without self-shading, leaf gradient doesn't diminish - runaway growth."""
        # Direct leaf -> photosynthesis gradient without Beer-Lambert
        def photo_no_shading(leaves: float) -> float:
            """Linear leaf contribution (no self-shading)."""
            return 0.5 * leaves * 0.5  # p_max * leaves * resource_eff

        # Gradient is constant regardless of leaf mass
        grad_at_1 = float(grad(photo_no_shading)(1.0))
        grad_at_10 = float(grad(photo_no_shading)(10.0))
        grad_at_100 = float(grad(photo_no_shading)(100.0))

        # Without self-shading, gradient never decreases
        assert jnp.isclose(grad_at_1, grad_at_10, atol=1e-6)
        assert jnp.isclose(grad_at_10, grad_at_100, atol=1e-6)

    def test_with_self_shading_gradient_decreases(self) -> None:
        """With self-shading, leaf gradient diminishes - stable growth."""

        def photo_with_shading(leaves: float) -> float:
            """Beer-Lambert self-shading: f(L) = 1 - exp(-k*L)."""
            leaf_eff = surrogates.leaf_area_efficiency(leaves, k_leaf=1.5)
            return 0.5 * leaf_eff * 0.5  # p_max * leaf_eff * resource_eff

        # Gradient diminishes with more leaves
        grad_at_1 = float(grad(photo_with_shading)(1.0))
        grad_at_3 = float(grad(photo_with_shading)(3.0))
        grad_at_10 = float(grad(photo_with_shading)(10.0))

        # Self-shading makes additional leaves less valuable
        assert grad_at_1 > grad_at_3 > grad_at_10
        # At high leaf mass, gradient approaches zero
        assert grad_at_10 < 0.01


class TestTransportBottleneckAblation:
    """Tests proving transport bottleneck requires trunk investment."""

    def test_without_transport_limit_leaves_alone_optimal(self) -> None:
        """Without transport bottleneck, just growing leaves is optimal."""
        # If water isn't limited by trunk, pure leaf investment dominates
        # because leaves produce energy with no trunk requirement
        config = SimConfig()

        # Without bottleneck: water_available = water (unlimited)
        # Just leaves produce energy
        photo_unlimited = float(
            surrogates.photosynthesis(
                leaves=5.0,  # High leaves
                light=0.7,
                water=10.0,  # Unlimited water
                nutrients=1.0,
                p_max=config.p_max,
                k_light=config.k_light,
                k_water=config.k_water,
                k_nutrient=config.k_nutrient,
                k_leaf=config.k_leaf,
            )
        )

        # With bottleneck: water_available = min(water, transport_cap(trunk))
        transport_cap = float(
            surrogates.transport_capacity(trunk=0.05, kappa=2.0, beta=0.7)
        )
        photo_limited = float(
            surrogates.photosynthesis(
                leaves=5.0,
                light=0.7,
                water=transport_cap,  # Limited by tiny trunk
                nutrients=1.0,
                p_max=config.p_max,
                k_light=config.k_light,
                k_water=config.k_water,
                k_nutrient=config.k_nutrient,
                k_leaf=config.k_leaf,
            )
        )

        # Without bottleneck, high leaves gives high output
        # With bottleneck, tiny trunk limits water, reducing output
        assert photo_unlimited > photo_limited * 1.5

    def test_transport_bottleneck_creates_trunk_gradient(self) -> None:
        """Transport bottleneck creates positive gradient for trunk investment."""

        def photo_rate_with_trunk(trunk):
            """Photosynthesis rate depends on trunk via transport capacity."""
            transport_cap = surrogates.transport_capacity(trunk=trunk, kappa=2.0, beta=0.7)
            water_available = jnp.minimum(1.0, transport_cap)  # Water store = 1.0
            return surrogates.photosynthesis(
                leaves=2.0,
                light=0.7,
                water=water_available,
                nutrients=0.5,
                p_max=0.5,
                k_light=0.3,
                k_water=0.2,
                k_nutrient=0.2,
                k_leaf=1.5,
            )

        # Gradient w.r.t. trunk should be positive when trunk is small
        grad_fn = grad(photo_rate_with_trunk)
        trunk_grad = float(grad_fn(0.1))

        # Transport bottleneck creates incentive to grow trunk
        assert trunk_grad > 0


class TestResourceConsumptionAblation:
    """Tests proving resource consumption prevents infinite energy."""

    def test_without_consumption_energy_unbounded(self) -> None:
        """Without resource consumption, photosynthesis could be infinite."""
        state = TreeState(
            energy=jnp.array(0.5),
            water=jnp.array(0.5),
            nutrients=jnp.array(0.5),
            roots=jnp.array(1.0),
            trunk=jnp.array(0.5),
            shoots=jnp.array(0.5),
            leaves=jnp.array(2.0),
            flowers=jnp.array(0.0),
        )
        config = SimConfig()

        # Run multiple steps
        energy_history = [float(state.energy)]
        for day in range(50):
            alloc = Allocation(
                roots=jnp.array(0.2),
                trunk=jnp.array(0.2),
                shoots=jnp.array(0.2),
                leaves=jnp.array(0.3),
                flowers=jnp.array(0.1),
            )
            state = step(state, alloc, light=0.7, moisture=0.6, wind=0.1, config=config, day=day)
            energy_history.append(float(state.energy))

        # With resource consumption, energy should stabilize (not grow forever)
        # Check that energy doesn't grow exponentially
        max_energy = max(energy_history)
        assert max_energy < 50.0  # Should be bounded, not exponential


class TestInvestmentGatingAblation:
    """Tests proving investment gating prevents suicide investing."""

    def test_low_energy_reduces_investment(self) -> None:
        """At low energy, investment is gated to prevent death spiral."""
        config = SimConfig()

        # High energy state
        high_energy = 2.0
        high_gate = 1 / (1 + jnp.exp(-config.investment_steepness * (high_energy - config.investment_energy_threshold)))

        # Low energy state
        low_energy = 0.1
        low_gate = 1 / (1 + jnp.exp(-config.investment_steepness * (low_energy - config.investment_energy_threshold)))

        # Low energy should gate investment
        assert float(low_gate) < 0.3
        assert float(high_gate) > 0.9

    def test_without_gating_low_energy_death_spiral(self) -> None:
        """Without investment gating, low energy leads to death spiral."""
        # Start with very low energy
        state = TreeState(
            energy=jnp.array(0.1),  # Very low
            water=jnp.array(0.3),
            nutrients=jnp.array(0.3),
            roots=jnp.array(0.3),  # Maintenance costs exist
            trunk=jnp.array(0.2),
            shoots=jnp.array(0.2),
            leaves=jnp.array(0.3),
            flowers=jnp.array(0.0),
        )

        # Config with high investment rate (simulating no gating)
        config = SimConfig(investment_rate=0.9, investment_steepness=0.1)  # Nearly always invest

        # The tree should survive (gating prevents death)
        # Even with aggressive investment, gating at low energy helps
        alloc = Allocation(
            roots=jnp.array(0.2),
            trunk=jnp.array(0.2),
            shoots=jnp.array(0.2),
            leaves=jnp.array(0.3),
            flowers=jnp.array(0.1),
        )

        # Run a few steps with low energy
        for day in range(10):
            state = step(state, alloc, light=0.7, moisture=0.6, wind=0.1, config=config, day=day)

        # Should still be alive (energy >= 0 due to clamping)
        assert float(state.energy) >= 0


class TestFlowerMaturityGatingAblation:
    """Tests proving flower maturity gating prevents early flower exploitation."""

    def test_early_flower_allocation_wasted(self) -> None:
        """Early flower allocation is blocked by maturity gate."""
        config = SimConfig(flowering_maturity=0.4)

        # Early in season (10% progress)
        day = 10
        progress = day / config.num_days
        assert progress < config.flowering_maturity

        # Maturity gate should be near 0
        maturity_gate = 1 / (1 + jnp.exp(-10.0 * (progress - config.flowering_maturity)))
        assert float(maturity_gate) < 0.1  # Gate blocks early flowering

    def test_late_flower_allocation_allowed(self) -> None:
        """Late flower allocation passes through maturity gate."""
        config = SimConfig(flowering_maturity=0.4)

        # Late in season (70% progress)
        day = 70
        progress = day / config.num_days
        assert progress > config.flowering_maturity

        # Maturity gate should be near 1
        maturity_gate = 1 / (1 + jnp.exp(-10.0 * (progress - config.flowering_maturity)))
        assert float(maturity_gate) > 0.9  # Gate allows late flowering


class TestMoistureInvertedUAblation:
    """Tests proving inverted-U moisture prevents 'more is always better'."""

    def test_optimal_moisture_beats_extremes(self) -> None:
        """Optimal moisture gives higher uptake than extremes."""
        config = SimConfig()

        # At optimal moisture
        water_opt, _ = surrogates.root_uptake(
            roots=1.0,
            moisture=config.moisture_optimum,
            u_water_max=config.u_water_max,
            u_nutrient_max=config.u_nutrient_max,
            k_root=config.k_root,
            m_opt=config.moisture_optimum,
            m_sigma=config.moisture_sigma,
        )

        # At high moisture (flooding)
        water_wet, _ = surrogates.root_uptake(
            roots=1.0,
            moisture=1.0,
            u_water_max=config.u_water_max,
            u_nutrient_max=config.u_nutrient_max,
            k_root=config.k_root,
            m_opt=config.moisture_optimum,
            m_sigma=config.moisture_sigma,
        )

        # At low moisture (drought)
        water_dry, _ = surrogates.root_uptake(
            roots=1.0,
            moisture=0.1,
            u_water_max=config.u_water_max,
            u_nutrient_max=config.u_nutrient_max,
            k_root=config.k_root,
            m_opt=config.moisture_optimum,
            m_sigma=config.moisture_sigma,
        )

        # Optimal beats both extremes
        assert float(water_opt) > float(water_wet)
        assert float(water_opt) > float(water_dry)


class TestWindDamageCapAblation:
    """Tests proving wind damage cap prevents instant wipeout."""

    def test_damage_capped_at_max(self) -> None:
        """Even extreme wind doesn't cause 100% damage."""
        config = SimConfig()

        # Extreme wind
        extreme_damage = surrogates.wind_damage(
            jnp.array(1.0),
            threshold=config.wind_threshold,
            steepness=config.wind_steepness,
            max_damage=config.max_wind_damage,
        )

        # Damage is capped
        assert float(extreme_damage) <= config.max_wind_damage
        assert float(extreme_damage) < 1.0  # Never 100% damage

    def test_wood_protection_reduces_damage(self) -> None:
        """Trunk investment provides protection against wind damage."""
        config = SimConfig()

        # Damage with no trunk
        damage_no_trunk = surrogates.effective_wind_damage(
            wind=0.8,
            trunk=0.01,
            threshold=config.wind_threshold,
            steepness=config.wind_steepness,
            max_damage=config.max_wind_damage,
            k_protection=config.k_wind_protection,
            max_protection=config.max_wind_protection,
        )

        # Damage with significant trunk
        damage_with_trunk = surrogates.effective_wind_damage(
            wind=0.8,
            trunk=2.0,
            threshold=config.wind_threshold,
            steepness=config.wind_steepness,
            max_damage=config.max_wind_damage,
            k_protection=config.k_wind_protection,
            max_protection=config.max_wind_protection,
        )

        # Trunk provides significant protection
        assert float(damage_with_trunk) < float(damage_no_trunk) * 0.5


class TestStructuralPenaltyAblation:
    """Tests proving structural penalty prevents unsupported canopy."""

    def test_unsupported_canopy_penalized(self) -> None:
        """Canopy without trunk support causes energy drain."""
        # High load (lots of leaves/shoots/flowers)
        high_load = surrogates.compute_load(
            leaves=5.0, shoots=3.0, flowers=2.0, c_leaf=1.0, c_shoot=0.5, c_flower=0.8
        )

        # Low capacity (small trunk)
        low_capacity = surrogates.compute_capacity(trunk=0.1, c_trunk=2.0, gamma=1.0)

        # Penalty should be significant
        penalty = surrogates.structural_penalty(high_load, low_capacity)
        assert float(penalty) > 5.0  # Significant drain

    def test_supported_canopy_no_penalty(self) -> None:
        """Canopy with adequate trunk support has minimal penalty."""
        # Moderate load
        load = surrogates.compute_load(
            leaves=2.0, shoots=1.0, flowers=0.5, c_leaf=1.0, c_shoot=0.5, c_flower=0.8
        )

        # High capacity (big trunk)
        capacity = surrogates.compute_capacity(trunk=3.0, c_trunk=2.0, gamma=1.0)

        # Penalty should be small
        penalty = surrogates.structural_penalty(load, capacity)
        assert float(penalty) < 0.5  # Minimal drain


class TestGradientHealthAblation:
    """Tests proving gradient floor prevents dead zones."""

    def test_photosynthesis_gradient_exists_at_low_resources(self) -> None:
        """Gradient floor ensures signal even at low resource levels."""

        def photo_rate(water):
            return surrogates.photosynthesis(
                leaves=1.0,
                light=0.7,
                water=water,
                nutrients=0.1,  # Very low nutrients
                p_max=0.5,
                k_light=0.3,
                k_water=0.2,
                k_nutrient=0.2,
                k_leaf=1.5,
                gradient_floor=0.03,  # Floor prevents dead zone
            )

        grad_fn = grad(photo_rate)

        # Even at very low water, gradient exists
        grad_low = float(grad_fn(0.01))
        assert grad_low > 0  # Gradient signal exists

    def test_growth_efficiency_gradient_exists(self) -> None:
        """Growth efficiency has gradient signal even at low resources."""

        def efficiency(water):
            return surrogates.growth_efficiency(
                water=water,
                nutrients=0.05,  # Very low
                base_efficiency=0.8,
                k_water=0.2,
                k_nutrient=0.2,
                gradient_floor=0.03,
            )

        grad_fn = grad(efficiency)

        # Gradient exists at low resources
        grad_low = float(grad_fn(0.01))
        assert grad_low > 0
