"""
Tests for biological surrogate functions.

These tests verify that our smooth mathematical approximations
of biological processes behave correctly at boundary conditions
and match expected shapes.
"""

import jax.numpy as jnp

from sim import surrogates


class TestSaturation:
    """Tests for Michaelis-Menten saturation function."""

    def test_zero_input_gives_zero(self) -> None:
        """f(0) = 0 for any positive K."""
        result = surrogates.saturation(jnp.array(0.0), k=0.5)
        assert jnp.isclose(result, 0.0, atol=1e-6)

    def test_approaches_one_at_high_input(self) -> None:
        """f(x) → 1 as x → ∞."""
        result = surrogates.saturation(jnp.array(100.0), k=0.5)
        assert jnp.isclose(result, 1.0, atol=0.01)

    def test_half_saturation_at_k(self) -> None:
        """f(K) = 0.5 by definition."""
        k = 0.3
        result = surrogates.saturation(jnp.array(k), k=k)
        assert jnp.isclose(result, 0.5, atol=1e-6)

    def test_monotonically_increasing(self) -> None:
        """Output increases with input."""
        x = jnp.linspace(0.0, 2.0, 20)
        y = surrogates.saturation(x, k=0.5)
        diffs = jnp.diff(y)
        assert jnp.all(diffs >= 0)

    def test_output_bounded_zero_one(self) -> None:
        """Output is always in [0, 1]."""
        x = jnp.linspace(0.0, 10.0, 100)
        y = surrogates.saturation(x, k=0.5)
        assert jnp.all(y >= 0.0)
        assert jnp.all(y <= 1.0)


class TestTemperatureWindow:
    """Tests for the temperature efficiency window."""

    def test_optimal_temperature_gives_high_efficiency(self) -> None:
        """Efficiency is high in the middle of the window."""
        # With t_min=10, t_max=30, optimal is around 20
        result = surrogates.temperature_window(
            jnp.array(20.0), t_min=10.0, t_max=30.0, steepness=1.0
        )
        assert result > 0.8

    def test_below_minimum_gives_low_efficiency(self) -> None:
        """Efficiency drops off below t_min."""
        result = surrogates.temperature_window(
            jnp.array(0.0), t_min=10.0, t_max=30.0, steepness=1.0
        )
        assert result < 0.1

    def test_above_maximum_gives_low_efficiency(self) -> None:
        """Efficiency drops off above t_max."""
        result = surrogates.temperature_window(
            jnp.array(40.0), t_min=10.0, t_max=30.0, steepness=1.0
        )
        assert result < 0.1

    def test_steepness_affects_transition(self) -> None:
        """Higher steepness gives sharper transitions."""
        t = jnp.array(8.0)  # Just below t_min=10
        soft = surrogates.temperature_window(t, t_min=10.0, t_max=30.0, steepness=0.5)
        sharp = surrogates.temperature_window(t, t_min=10.0, t_max=30.0, steepness=2.0)
        # Sharper transition means lower value just outside window
        assert sharp < soft

    def test_output_bounded_zero_one(self) -> None:
        """Output is always in [0, 1]."""
        temps = jnp.linspace(-10.0, 50.0, 100)
        y = surrogates.temperature_window(temps, t_min=10.0, t_max=30.0, steepness=1.0)
        assert jnp.all(y >= 0.0)
        assert jnp.all(y <= 1.0)


class TestWindDamage:
    """Tests for wind damage sigmoid."""

    def test_low_wind_no_damage(self) -> None:
        """Below threshold, damage is near zero."""
        result = surrogates.wind_damage(
            jnp.array(0.1), threshold=0.5, steepness=10.0
        )
        assert result < 0.02  # sigmoid(-4) ≈ 0.018

    def test_high_wind_full_damage(self) -> None:
        """Well above threshold, damage approaches 1."""
        result = surrogates.wind_damage(
            jnp.array(1.0), threshold=0.5, steepness=10.0
        )
        assert result > 0.99

    def test_threshold_gives_half_damage(self) -> None:
        """At threshold, damage is 0.5."""
        result = surrogates.wind_damage(
            jnp.array(0.5), threshold=0.5, steepness=10.0
        )
        assert jnp.isclose(result, 0.5, atol=0.01)

    def test_monotonically_increasing(self) -> None:
        """Damage increases with wind speed."""
        wind = jnp.linspace(0.0, 1.0, 20)
        damage = surrogates.wind_damage(wind, threshold=0.5, steepness=10.0)
        diffs = jnp.diff(damage)
        assert jnp.all(diffs >= 0)


class TestStructuralPenalty:
    """Tests for structural load/capacity penalty."""

    def test_no_penalty_at_zero_load(self) -> None:
        """Penalty is exactly zero when load is zero."""
        # Baseline subtraction ensures penalty(0, c) = 0
        for capacity in [0.5, 1.0, 2.0, 5.0]:
            penalty = surrogates.structural_penalty(load=0.0, capacity=capacity)
            assert jnp.isclose(penalty, 0.0, atol=1e-6)

    def test_small_penalty_when_supported(self) -> None:
        """When capacity > load, penalty is small."""
        penalty = surrogates.structural_penalty(load=1.0, capacity=2.0)
        assert penalty < 0.5  # Small but not necessarily near zero

    def test_penalty_when_unsupported(self) -> None:
        """When load > capacity, penalty is positive."""
        penalty = surrogates.structural_penalty(
            load=3.0, capacity=1.0
        )
        assert penalty > 1.5  # Should be roughly (3.0 - 1.0) = 2.0

    def test_smooth_transition(self) -> None:
        """Penalty transitions smoothly near equilibrium."""
        loads = jnp.linspace(0.5, 1.5, 20)
        penalties = jnp.array([
            surrogates.structural_penalty(load=float(l), capacity=1.0)
            for l in loads
        ])
        # Check monotonically increasing
        diffs = jnp.diff(penalties)
        assert jnp.all(diffs >= 0)


class TestPhotosynthesis:
    """Tests for the combined photosynthesis function."""

    def test_no_leaves_no_photosynthesis(self) -> None:
        """Zero leaf biomass means zero photosynthesis."""
        result = surrogates.photosynthesis(
            leaves=0.0,
            light=0.8,
            water=0.5,
            nutrients=0.5,
            p_max=0.5,
            k_light=0.3,
            k_water=0.2,
            k_nutrient=0.2,
        )
        assert jnp.isclose(result, 0.0, atol=1e-6)

    def test_no_light_low_photosynthesis(self) -> None:
        """Zero light means very low photosynthesis (gradient floor only)."""
        result = surrogates.photosynthesis(
            leaves=1.0,
            light=0.0,
            water=0.5,
            nutrients=0.5,
            p_max=0.5,
            k_light=0.3,
            k_water=0.2,
            k_nutrient=0.2,
        )
        # With gradient floor, result is small but nonzero (floor * p_max * leaves)
        assert result > 0  # Floor keeps gradient signal alive
        assert result < 0.1  # But still very small

    def test_no_water_low_photosynthesis(self) -> None:
        """Zero water means very low photosynthesis (gradient floor only)."""
        result = surrogates.photosynthesis(
            leaves=1.0,
            light=0.8,
            water=0.0,
            nutrients=0.5,
            p_max=0.5,
            k_light=0.3,
            k_water=0.2,
            k_nutrient=0.2,
        )
        # With gradient floor, result is small but nonzero
        assert result > 0  # Floor keeps gradient signal alive
        assert result < 0.1  # But still very small

    def test_scales_with_leaves(self) -> None:
        """More leaves means more photosynthesis."""
        base_params = dict(
            light=0.8,
            water=0.5,
            nutrients=0.5,
            p_max=0.5,
            k_light=0.3,
            k_water=0.2,
            k_nutrient=0.2,
        )
        p1 = surrogates.photosynthesis(leaves=1.0, **base_params)
        p2 = surrogates.photosynthesis(leaves=2.0, **base_params)
        assert p2 > p1

    def test_output_nonnegative(self) -> None:
        """Photosynthesis is never negative."""
        result = surrogates.photosynthesis(
            leaves=1.0,
            light=0.8,
            water=0.5,
            nutrients=0.5,
            p_max=0.5,
            k_light=0.3,
            k_water=0.2,
            k_nutrient=0.2,
        )
        assert result >= 0.0


class TestRootUptake:
    """Tests for root water/nutrient uptake."""

    def test_no_roots_no_uptake(self) -> None:
        """Zero root biomass means zero uptake."""
        water, nutrients = surrogates.root_uptake(
            roots=0.0,
            moisture=0.8,
            u_water_max=0.3,
            u_nutrient_max=0.2,
            k_root=0.5,
        )
        assert jnp.isclose(water, 0.0, atol=1e-6)
        assert jnp.isclose(nutrients, 0.0, atol=1e-6)

    def test_no_moisture_no_water_uptake(self) -> None:
        """Zero moisture means zero water uptake."""
        water, _ = surrogates.root_uptake(
            roots=1.0,
            moisture=0.0,
            u_water_max=0.3,
            u_nutrient_max=0.2,
            k_root=0.5,
        )
        assert jnp.isclose(water, 0.0, atol=1e-6)

    def test_more_roots_more_uptake(self) -> None:
        """More roots means more uptake."""
        w1, n1 = surrogates.root_uptake(
            roots=0.5,
            moisture=0.8,
            u_water_max=0.3,
            u_nutrient_max=0.2,
            k_root=0.5,
        )
        w2, n2 = surrogates.root_uptake(
            roots=1.5,
            moisture=0.8,
            u_water_max=0.3,
            u_nutrient_max=0.2,
            k_root=0.5,
        )
        assert w2 > w1
        assert n2 > n1

    def test_output_bounded(self) -> None:
        """Uptake is bounded by max rates."""
        water, nutrients = surrogates.root_uptake(
            roots=100.0,  # Lots of roots
            moisture=1.0,  # Maximum moisture
            u_water_max=0.3,
            u_nutrient_max=0.2,
            k_root=0.5,
        )
        assert water <= 0.3 + 1e-6
        assert nutrients <= 0.2 + 1e-6


class TestMaintenanceCost:
    """Tests for maintenance cost calculation."""

    def test_zero_biomass_zero_cost(self) -> None:
        """No biomass means no maintenance cost."""
        from sim.config import TreeState
        state = TreeState(
            energy=jnp.array(1.0),
            water=jnp.array(0.0),
            nutrients=jnp.array(0.0),
            roots=jnp.array(0.0),
            trunk=jnp.array(0.0),
            shoots=jnp.array(0.0),
            leaves=jnp.array(0.0),
            flowers=jnp.array(0.0),
        )
        cost = surrogates.maintenance_cost(
            state,
            m_root=0.01,
            m_trunk=0.005,
            m_shoot=0.02,
            m_leaf=0.03,
            m_flower=0.04,
        )
        assert jnp.isclose(cost, 0.0, atol=1e-6)

    def test_cost_increases_with_biomass(self) -> None:
        """More biomass means higher maintenance cost."""
        from sim.config import TreeState
        state1 = TreeState(
            energy=jnp.array(1.0),
            water=jnp.array(0.0),
            nutrients=jnp.array(0.0),
            roots=jnp.array(1.0),
            trunk=jnp.array(1.0),
            shoots=jnp.array(1.0),
            leaves=jnp.array(1.0),
            flowers=jnp.array(1.0),
        )
        state2 = TreeState(
            energy=jnp.array(1.0),
            water=jnp.array(0.0),
            nutrients=jnp.array(0.0),
            roots=jnp.array(2.0),
            trunk=jnp.array(2.0),
            shoots=jnp.array(2.0),
            leaves=jnp.array(2.0),
            flowers=jnp.array(2.0),
        )
        cost1 = surrogates.maintenance_cost(
            state1, m_root=0.01, m_trunk=0.005, m_shoot=0.02, m_leaf=0.03, m_flower=0.04
        )
        cost2 = surrogates.maintenance_cost(
            state2, m_root=0.01, m_trunk=0.005, m_shoot=0.02, m_leaf=0.03, m_flower=0.04
        )
        assert cost2 > cost1

    def test_cost_always_nonnegative(self) -> None:
        """Maintenance cost is never negative."""
        from sim.config import TreeState
        state = TreeState(
            energy=jnp.array(1.0),
            water=jnp.array(0.5),
            nutrients=jnp.array(0.5),
            roots=jnp.array(1.0),
            trunk=jnp.array(2.0),
            shoots=jnp.array(0.5),
            leaves=jnp.array(1.5),
            flowers=jnp.array(0.3),
        )
        cost = surrogates.maintenance_cost(
            state, m_root=0.01, m_trunk=0.005, m_shoot=0.02, m_leaf=0.03, m_flower=0.04
        )
        assert cost >= 0.0


class TestSeedProduction:
    """Tests for seed production calculation."""

    def test_no_flowers_no_seeds(self) -> None:
        """Zero flowers means zero seeds."""
        seeds = surrogates.seed_production(
            flowers=0.0,
            energy=1.0,
            energy_threshold=0.5,
            conversion=10.0,
        )
        assert jnp.isclose(seeds, 0.0, atol=1e-6)

    def test_low_energy_no_seeds(self) -> None:
        """Below energy threshold, seed production is suppressed."""
        # sigmoid((0.1 - 0.5) * 10) = sigmoid(-4) ≈ 0.018
        # So seeds ≈ 10.0 * 1.0 * 0.018 ≈ 0.18
        seeds = surrogates.seed_production(
            flowers=1.0,
            energy=0.1,  # Below threshold of 0.5
            energy_threshold=0.5,
            conversion=10.0,
        )
        assert seeds < 0.2  # Significantly suppressed but not zero (soft gate)

    def test_seeds_scale_with_flowers(self) -> None:
        """More flowers means more seeds."""
        s1 = surrogates.seed_production(
            flowers=1.0, energy=1.0, energy_threshold=0.5, conversion=10.0
        )
        s2 = surrogates.seed_production(
            flowers=2.0, energy=1.0, energy_threshold=0.5, conversion=10.0
        )
        assert s2 > s1

    def test_seeds_nonnegative(self) -> None:
        """Seed count is never negative."""
        seeds = surrogates.seed_production(
            flowers=1.0,
            energy=1.0,
            energy_threshold=0.5,
            conversion=10.0,
        )
        assert seeds >= 0.0
