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
        result = surrogates.wind_damage(jnp.array(0.1), threshold=0.5, steepness=10.0)
        assert result < 0.02  # sigmoid(-4) ≈ 0.018

    def test_high_wind_full_damage(self) -> None:
        """Well above threshold, damage approaches max_damage (0.5 by default)."""
        result = surrogates.wind_damage(jnp.array(1.0), threshold=0.5, steepness=10.0)
        # With max_damage=0.5, full damage approaches 0.5, not 1.0
        assert result > 0.49

    def test_threshold_gives_half_damage(self) -> None:
        """At threshold, damage is half of max_damage."""
        result = surrogates.wind_damage(jnp.array(0.5), threshold=0.5, steepness=10.0)
        # With max_damage=0.5, half damage is 0.25
        assert jnp.isclose(result, 0.25, atol=0.01)

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
        penalty = surrogates.structural_penalty(load=3.0, capacity=1.0)
        assert penalty > 1.5  # Should be roughly (3.0 - 1.0) = 2.0

    def test_smooth_transition(self) -> None:
        """Penalty transitions smoothly near equilibrium."""
        loads = jnp.linspace(0.5, 1.5, 20)
        penalties = jnp.array(
            [
                surrogates.structural_penalty(load=float(load_val), capacity=1.0)
                for load_val in loads
            ]
        )
        # Check monotonically increasing
        diffs = jnp.diff(penalties)
        assert jnp.all(diffs >= 0)


class TestLeafAreaEfficiency:
    """Tests for Beer-Lambert self-shading function."""

    def test_zero_leaves_zero_efficiency(self) -> None:
        """f(0) = 1 - exp(0) = 0."""
        result = surrogates.leaf_area_efficiency(jnp.array(0.0), k_leaf=1.5)
        assert jnp.isclose(result, 0.0, atol=1e-6)

    def test_approaches_one_at_high_leaves(self) -> None:
        """f(L) → 1 as L → ∞."""
        result = surrogates.leaf_area_efficiency(jnp.array(10.0), k_leaf=1.5)
        assert result > 0.99

    def test_monotonically_increasing(self) -> None:
        """Efficiency increases with leaf biomass."""
        leaves = jnp.linspace(0.0, 3.0, 20)
        eff = surrogates.leaf_area_efficiency(leaves, k_leaf=1.5)
        diffs = jnp.diff(eff)
        assert jnp.all(diffs >= 0)

    def test_higher_k_faster_saturation(self) -> None:
        """Higher k_leaf means faster saturation."""
        leaves = jnp.array(1.0)
        eff_low_k = surrogates.leaf_area_efficiency(leaves, k_leaf=0.5)
        eff_high_k = surrogates.leaf_area_efficiency(leaves, k_leaf=2.0)
        assert eff_high_k > eff_low_k

    def test_output_bounded_zero_one(self) -> None:
        """Output is always in [0, 1)."""
        leaves = jnp.linspace(0.0, 10.0, 100)
        eff = surrogates.leaf_area_efficiency(leaves, k_leaf=1.5)
        assert jnp.all(eff >= 0.0)
        assert jnp.all(eff < 1.0)


class TestPhotosynthesis:
    """Tests for the combined photosynthesis function."""

    def test_no_leaves_no_photosynthesis(self) -> None:
        """Zero leaf biomass means zero photosynthesis (via Beer-Lambert)."""
        result = surrogates.photosynthesis(
            leaves=0.0,
            light=0.8,
            water=0.5,
            nutrients=0.5,
            p_max=0.5,
            k_light=0.3,
            k_water=0.2,
            k_nutrient=0.2,
            k_leaf=1.5,
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
            k_leaf=1.5,
        )
        # With gradient floor, result is small but nonzero
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
            k_leaf=1.5,
        )
        # With gradient floor, result is small but nonzero
        assert result > 0  # Floor keeps gradient signal alive
        assert result < 0.1  # But still very small

    def test_scales_with_leaves_but_saturates(self) -> None:
        """More leaves means more photosynthesis, but with diminishing returns."""
        base_params = {
            "light": 0.8,
            "water": 0.5,
            "nutrients": 0.5,
            "p_max": 0.5,
            "k_light": 0.3,
            "k_water": 0.2,
            "k_nutrient": 0.2,
            "k_leaf": 1.5,
        }
        p1 = surrogates.photosynthesis(leaves=0.5, **base_params)
        p2 = surrogates.photosynthesis(leaves=1.0, **base_params)
        p3 = surrogates.photosynthesis(leaves=2.0, **base_params)
        p4 = surrogates.photosynthesis(leaves=4.0, **base_params)

        # More leaves = more photosynthesis
        assert p2 > p1
        assert p3 > p2
        assert p4 > p3

        # But with diminishing returns (due to self-shading)
        gain_1_to_2 = p2 - p1
        gain_2_to_3 = p3 - p2
        gain_3_to_4 = p4 - p3
        assert gain_2_to_3 < gain_1_to_2  # Diminishing returns
        assert gain_3_to_4 < gain_2_to_3  # Still diminishing

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
            k_leaf=1.5,
        )
        assert result >= 0.0

    def test_bounded_by_p_max(self) -> None:
        """Photosynthesis approaches p_max at high resources."""
        result = surrogates.photosynthesis(
            leaves=10.0,  # High leaves (saturated)
            light=1.0,  # Max light
            water=10.0,  # High water
            nutrients=10.0,  # High nutrients
            p_max=0.5,
            k_light=0.3,
            k_water=0.2,
            k_nutrient=0.2,
            k_leaf=1.5,
        )
        # Should approach p_max (0.5) but not exceed it
        # Note: actual max is p_max * leaf_eff * combined_eff
        # where combined_eff = floor + (1-floor) * product < 1
        # and leaf_eff = 1 - exp(-k*L) < 1
        assert result < 0.5 + 1e-6
        assert result > 0.35  # Should be reasonably high


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

    def test_extreme_moisture_low_uptake(self) -> None:
        """Extreme moisture (too low or too high) gives reduced uptake due to inverted-U."""
        # At moisture=0, far from optimum (0.6), efficiency is low but not zero
        # Gaussian: exp(-(0-0.6)^2 / (2*0.25^2)) = exp(-2.88) ≈ 0.056
        water_dry, _ = surrogates.root_uptake(
            roots=1.0,
            moisture=0.0,
            u_water_max=0.3,
            u_nutrient_max=0.2,
            k_root=0.5,
        )
        # At moisture=1.0, also far from optimum, efficiency is low
        water_wet, _ = surrogates.root_uptake(
            roots=1.0,
            moisture=1.0,
            u_water_max=0.3,
            u_nutrient_max=0.2,
            k_root=0.5,
        )
        # At optimal moisture, efficiency is high
        water_opt, _ = surrogates.root_uptake(
            roots=1.0,
            moisture=0.6,  # Default optimum
            u_water_max=0.3,
            u_nutrient_max=0.2,
            k_root=0.5,
        )
        # Both extremes should be much lower than optimal
        assert water_dry < water_opt * 0.2  # Dry: <20% of optimal
        assert water_wet < water_opt * 0.5  # Wet: <50% of optimal (closer to optimum)

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


class TestRootUptakeFromSoil:
    """Tests for root uptake from soil reservoir."""

    def test_no_roots_no_uptake(self) -> None:
        """Without roots, no uptake occurs."""
        water, nutrients, extracted = surrogates.root_uptake_from_soil(
            roots=0.0,
            soil_water=1.0,
            moisture=0.6,
            u_water_max=0.3,
            u_nutrient_max=0.2,
            k_root=0.5,
        )
        assert jnp.isclose(water, 0.0, atol=1e-6)
        assert jnp.isclose(nutrients, 0.0, atol=1e-6)
        assert jnp.isclose(extracted, 0.0, atol=1e-6)

    def test_no_soil_water_no_uptake(self) -> None:
        """Without soil water, no uptake occurs (drought scarcity)."""
        water, nutrients, extracted = surrogates.root_uptake_from_soil(
            roots=1.0,
            soil_water=0.0,
            moisture=0.6,
            u_water_max=0.3,
            u_nutrient_max=0.2,
            k_root=0.5,
        )
        assert jnp.isclose(water, 0.0, atol=1e-6)
        assert jnp.isclose(extracted, 0.0, atol=1e-6)

    def test_extreme_moisture_low_uptake(self) -> None:
        """Both very low and very high moisture reduce uptake (inverted-U)."""
        # Optimal moisture
        w_opt, _, _ = surrogates.root_uptake_from_soil(
            roots=1.0,
            soil_water=1.0,
            moisture=0.6,  # Optimal
            u_water_max=0.3,
            u_nutrient_max=0.2,
            k_root=0.5,
        )

        # Very low moisture (drought stress on roots)
        w_dry, _, _ = surrogates.root_uptake_from_soil(
            roots=1.0,
            soil_water=1.0,
            moisture=0.1,  # Too dry
            u_water_max=0.3,
            u_nutrient_max=0.2,
            k_root=0.5,
        )

        # Very high moisture (waterlogging/anoxia)
        w_wet, _, _ = surrogates.root_uptake_from_soil(
            roots=1.0,
            soil_water=1.0,
            moisture=1.0,  # Too wet
            u_water_max=0.3,
            u_nutrient_max=0.2,
            k_root=0.5,
        )

        # Optimal should beat both extremes (inverted-U / bell curve)
        assert float(w_opt) > float(w_dry)
        assert float(w_opt) > float(w_wet)

    def test_extraction_limited_by_soil_water(self) -> None:
        """Cannot extract more water than exists in soil."""
        # High demand but low soil water
        water, _, extracted = surrogates.root_uptake_from_soil(
            roots=10.0,  # High roots = high demand
            soil_water=0.05,  # Very little soil water
            moisture=0.6,
            u_water_max=0.3,
            u_nutrient_max=0.2,
            k_root=0.5,
        )
        # Extraction should be capped at soil water level
        assert float(extracted) <= 0.05

    def test_more_roots_more_uptake(self) -> None:
        """More roots means more uptake."""
        w1, n1, _ = surrogates.root_uptake_from_soil(
            roots=0.5,
            soil_water=1.0,
            moisture=0.6,
            u_water_max=0.3,
            u_nutrient_max=0.2,
            k_root=0.5,
        )
        w2, n2, _ = surrogates.root_uptake_from_soil(
            roots=2.0,
            soil_water=1.0,
            moisture=0.6,
            u_water_max=0.3,
            u_nutrient_max=0.2,
            k_root=0.5,
        )
        assert w2 > w1
        assert n2 > n1

    def test_more_soil_water_more_uptake(self) -> None:
        """More soil water means more uptake."""
        w1, _, _ = surrogates.root_uptake_from_soil(
            roots=1.0,
            soil_water=0.2,
            moisture=0.6,
            u_water_max=0.3,
            u_nutrient_max=0.2,
            k_root=0.5,
        )
        w2, _, _ = surrogates.root_uptake_from_soil(
            roots=1.0,
            soil_water=1.0,
            moisture=0.6,
            u_water_max=0.3,
            u_nutrient_max=0.2,
            k_root=0.5,
        )
        assert w2 > w1


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
            soil_water=jnp.array(0.0),
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
            soil_water=jnp.array(0.5),
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
            soil_water=jnp.array(0.5),
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
            soil_water=jnp.array(0.5),
        )
        cost = surrogates.maintenance_cost(
            state, m_root=0.01, m_trunk=0.005, m_shoot=0.02, m_leaf=0.03, m_flower=0.04
        )
        assert cost >= 0.0


class TestTransportCapacity:
    """Tests for trunk-based water transport capacity."""

    def test_zero_trunk_near_zero_capacity(self) -> None:
        """Very small trunk means very small transport capacity."""
        result = surrogates.transport_capacity(trunk=0.0, kappa=2.0, beta=0.7)
        # With 1e-8 floor, result is tiny but defined
        assert result < 1e-5

    def test_more_trunk_more_capacity(self) -> None:
        """More trunk means more transport capacity."""
        c1 = surrogates.transport_capacity(trunk=0.5, kappa=2.0, beta=0.7)
        c2 = surrogates.transport_capacity(trunk=1.0, kappa=2.0, beta=0.7)
        c3 = surrogates.transport_capacity(trunk=2.0, kappa=2.0, beta=0.7)
        assert c2 > c1
        assert c3 > c2

    def test_sublinear_scaling(self) -> None:
        """With beta < 1, capacity grows sublinearly with trunk."""
        # Doubling trunk should less than double capacity
        c1 = surrogates.transport_capacity(trunk=1.0, kappa=2.0, beta=0.7)
        c2 = surrogates.transport_capacity(trunk=2.0, kappa=2.0, beta=0.7)
        ratio = c2 / c1
        # For beta=0.7, ratio should be 2^0.7 ≈ 1.62, not 2.0
        assert ratio < 2.0
        assert ratio > 1.5

    def test_kappa_scales_linearly(self) -> None:
        """Kappa scales capacity linearly."""
        c1 = surrogates.transport_capacity(trunk=1.0, kappa=1.0, beta=0.7)
        c2 = surrogates.transport_capacity(trunk=1.0, kappa=2.0, beta=0.7)
        assert jnp.isclose(c2, 2 * c1, rtol=1e-5)

    def test_output_nonnegative(self) -> None:
        """Transport capacity is never negative."""
        trunks = jnp.linspace(0.0, 5.0, 50)
        for trunk in trunks:
            result = surrogates.transport_capacity(
                trunk=float(trunk), kappa=2.0, beta=0.7
            )
            assert result >= 0.0


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


class TestStomatalConductance:
    """Tests for stomatal conductance (drought response A)."""

    def test_high_water_full_conductance(self) -> None:
        """With plenty of water, stomata are fully open."""
        result = surrogates.stomatal_conductance(
            water=1.0,
            water_threshold=0.25,
            steepness=10.0,
            min_conductance=0.1,
        )
        assert float(result) > 0.95  # Nearly 1.0

    def test_low_water_reduced_conductance(self) -> None:
        """With low water, stomata close (reduced conductance)."""
        result = surrogates.stomatal_conductance(
            water=0.05,
            water_threshold=0.25,
            steepness=10.0,
            min_conductance=0.1,
        )
        # Should be close to min_conductance (soft sigmoid, not hard cutoff)
        assert float(result) < 0.25  # Significantly reduced
        assert float(result) >= 0.1  # Never below min

    def test_threshold_gives_half_range(self) -> None:
        """At threshold, conductance should be about midway."""
        threshold = 0.25
        result = surrogates.stomatal_conductance(
            water=threshold,
            water_threshold=threshold,
            steepness=10.0,
            min_conductance=0.1,
        )
        # sigmoid(0) = 0.5, so result = 0.1 + 0.9 * 0.5 = 0.55
        assert jnp.isclose(result, 0.55, atol=0.05)

    def test_min_conductance_preserved(self) -> None:
        """Even at zero water, conductance is at minimum."""
        result = surrogates.stomatal_conductance(
            water=0.0,
            water_threshold=0.25,
            steepness=10.0,
            min_conductance=0.1,
        )
        assert float(result) >= 0.1

    def test_output_bounded(self) -> None:
        """Conductance is always in [min, 1]."""
        water_levels = jnp.linspace(0.0, 1.0, 50)
        for w in water_levels:
            result = surrogates.stomatal_conductance(
                water=float(w),
                water_threshold=0.25,
                steepness=10.0,
                min_conductance=0.1,
            )
            assert float(result) >= 0.1
            assert float(result) <= 1.0


class TestDroughtDamage:
    """Tests for drought leaf damage (drought response B)."""

    def test_high_water_no_damage(self) -> None:
        """With plenty of water, no drought damage."""
        result = surrogates.drought_damage(
            water=0.5,
            water_critical=0.15,
            steepness=15.0,
            max_damage=0.25,
        )
        assert float(result) < 0.01  # Essentially zero

    def test_critical_water_gives_half_damage(self) -> None:
        """At critical threshold, damage is about half max."""
        result = surrogates.drought_damage(
            water=0.15,  # At threshold
            water_critical=0.15,
            steepness=15.0,
            max_damage=0.25,
        )
        # sigmoid(0) = 0.5, so damage = 0.25 * 0.5 = 0.125
        assert jnp.isclose(result, 0.125, atol=0.02)

    def test_very_low_water_full_damage(self) -> None:
        """At very low water, damage approaches max."""
        result = surrogates.drought_damage(
            water=0.01,
            water_critical=0.15,
            steepness=15.0,
            max_damage=0.25,
        )
        # Should be close to max_damage
        assert float(result) > 0.2

    def test_max_damage_capped(self) -> None:
        """Damage never exceeds max_damage."""
        result = surrogates.drought_damage(
            water=0.0,
            water_critical=0.15,
            steepness=15.0,
            max_damage=0.25,
        )
        assert float(result) <= 0.25 + 1e-6

    def test_damage_nonnegative(self) -> None:
        """Damage is never negative."""
        water_levels = jnp.linspace(0.0, 1.0, 50)
        for w in water_levels:
            result = surrogates.drought_damage(
                water=float(w),
                water_critical=0.15,
                steepness=15.0,
                max_damage=0.25,
            )
            assert float(result) >= 0.0


class TestWaterStress:
    """Tests for water stress signal."""

    def test_high_water_no_stress(self) -> None:
        """With plenty of water, stress is low."""
        result = surrogates.water_stress(
            water=0.5,
            water_threshold=0.2,
            steepness=10.0,
        )
        assert float(result) < 0.1

    def test_low_water_high_stress(self) -> None:
        """With low water, stress is high."""
        result = surrogates.water_stress(
            water=0.05,
            water_threshold=0.2,
            steepness=10.0,
        )
        assert float(result) > 0.8

    def test_threshold_gives_half_stress(self) -> None:
        """At threshold, stress is 0.5."""
        result = surrogates.water_stress(
            water=0.2,
            water_threshold=0.2,
            steepness=10.0,
        )
        assert jnp.isclose(result, 0.5, atol=0.01)

    def test_output_bounded(self) -> None:
        """Stress is always in [0, 1]."""
        water_levels = jnp.linspace(0.0, 1.0, 50)
        for w in water_levels:
            result = surrogates.water_stress(
                water=float(w),
                water_threshold=0.2,
                steepness=10.0,
            )
            assert float(result) >= 0.0
            assert float(result) <= 1.0


class TestMaturityGate:
    """Tests for phenology maturity gate (trunk + leaves requirement)."""

    def test_both_below_threshold_gate_closed(self) -> None:
        """With both trunk and leaves below threshold, gate is nearly closed."""
        result = surrogates.maturity_gate(
            trunk=0.05,
            leaves=0.1,
            trunk_threshold=0.2,
            leaves_threshold=0.3,
            steepness=10.0,
        )
        assert float(result) < 0.1  # Gate nearly closed

    def test_trunk_only_gate_closed(self) -> None:
        """With only trunk above threshold, gate still closed (need both)."""
        result = surrogates.maturity_gate(
            trunk=0.5,  # Above threshold
            leaves=0.1,  # Below threshold
            trunk_threshold=0.2,
            leaves_threshold=0.3,
            steepness=10.0,
        )
        assert float(result) < 0.2  # Gate mostly closed

    def test_leaves_only_gate_closed(self) -> None:
        """With only leaves above threshold, gate still closed (need both)."""
        result = surrogates.maturity_gate(
            trunk=0.05,  # Below threshold
            leaves=0.5,  # Above threshold
            trunk_threshold=0.2,
            leaves_threshold=0.3,
            steepness=10.0,
        )
        assert float(result) < 0.2  # Gate mostly closed

    def test_both_above_threshold_gate_open(self) -> None:
        """With both trunk and leaves above threshold, gate is open."""
        result = surrogates.maturity_gate(
            trunk=0.5,  # Above threshold
            leaves=0.6,  # Above threshold
            trunk_threshold=0.2,
            leaves_threshold=0.3,
            steepness=10.0,
        )
        assert float(result) > 0.8  # Gate open

    def test_output_bounded(self) -> None:
        """Gate is always in [0, 1]."""
        for trunk in [0.0, 0.2, 0.5, 1.0]:
            for leaves in [0.0, 0.3, 0.6, 1.0]:
                result = surrogates.maturity_gate(
                    trunk=trunk,
                    leaves=leaves,
                    trunk_threshold=0.2,
                    leaves_threshold=0.3,
                    steepness=10.0,
                )
                assert float(result) >= 0.0
                assert float(result) <= 1.0


class TestFlowerWindDamage:
    """Tests for flower-specific wind damage with trunk protection."""

    def test_low_wind_no_damage(self) -> None:
        """Below wind threshold, flowers are safe."""
        result = surrogates.flower_wind_damage(
            wind=0.2,
            trunk=0.0,
            threshold=0.5,
            steepness=8.0,
            max_damage=0.5,
            alpha_flower=0.7,
            k_protection=2.0,
            max_protection=0.9,
        )
        assert float(result) < 0.1  # Minimal damage

    def test_high_wind_no_trunk_severe_damage(self) -> None:
        """High wind with no trunk causes severe flower damage."""
        result = surrogates.flower_wind_damage(
            wind=0.8,
            trunk=0.0,
            threshold=0.5,
            steepness=8.0,
            max_damage=0.5,
            alpha_flower=0.7,
            k_protection=2.0,
            max_protection=0.9,
        )
        # With no trunk: damage ≈ 0.7 * 0.5 * sigmoid(2.4) ≈ 0.7 * 0.5 * 0.92 ≈ 0.32
        assert float(result) > 0.25  # Significant damage

    def test_high_wind_with_trunk_protected(self) -> None:
        """High wind with trunk provides substantial protection to flowers."""
        damage_no_trunk = surrogates.flower_wind_damage(
            wind=0.8,
            trunk=0.0,
            threshold=0.5,
            steepness=8.0,
            max_damage=0.5,
            alpha_flower=0.7,
            k_protection=2.0,
            max_protection=0.9,
        )
        damage_with_trunk = surrogates.flower_wind_damage(
            wind=0.8,
            trunk=1.0,
            threshold=0.5,
            steepness=8.0,
            max_damage=0.5,
            alpha_flower=0.7,
            k_protection=2.0,
            max_protection=0.9,
        )
        # Trunk should provide substantial protection (90% max with k=2.0)
        assert float(damage_with_trunk) < float(damage_no_trunk) * 0.3
        # With trunk=1.0, k=2.0: protection ≈ 0.9 * (1 - exp(-2)) ≈ 0.78
        # So damage reduced by ~78%

    def test_flower_more_vulnerable_than_leaves(self) -> None:
        """Flowers have higher alpha than leaves (more vulnerable base)."""
        # Compare with effective_wind_damage for leaves
        flower_damage = surrogates.flower_wind_damage(
            wind=0.7,
            trunk=0.0,  # No protection
            threshold=0.5,
            alpha_flower=0.7,  # High vulnerability
        )
        leaf_damage_base = surrogates.effective_wind_damage(
            wind=0.7,
            trunk=0.0,
            threshold=0.5,
        )
        # alpha_leaf (0.2) < alpha_flower (0.7)
        # So flower damage should be higher than leaf damage
        assert float(flower_damage) > float(leaf_damage_base) * 0.2  # leaf alpha

    def test_trunk_protection_stronger_for_flowers(self) -> None:
        """Trunk protection is stronger for flowers (higher k, higher max)."""
        # Flower protection at trunk=0.5
        flower_damage_protected = surrogates.flower_wind_damage(
            wind=0.7,
            trunk=0.5,
            k_protection=2.0,  # Higher k
            max_protection=0.9,  # Higher max
        )
        flower_damage_unprotected = surrogates.flower_wind_damage(
            wind=0.7,
            trunk=0.0,
        )

        # General protection at trunk=0.5
        general_damage_protected = surrogates.effective_wind_damage(
            wind=0.7,
            trunk=0.5,
            k_protection=1.0,  # Lower k
            max_protection=0.7,  # Lower max
        )
        general_damage_unprotected = surrogates.effective_wind_damage(
            wind=0.7,
            trunk=0.0,
        )

        # Flower protection ratio should be better
        flower_ratio = float(flower_damage_protected) / float(flower_damage_unprotected)
        general_ratio = float(general_damage_protected) / float(
            general_damage_unprotected
        )
        # Flower protection more effective (lower ratio means more protection)
        assert flower_ratio < general_ratio

    def test_output_bounded(self) -> None:
        """Flower damage is always in reasonable bounds."""
        for wind in [0.0, 0.5, 1.0]:
            for trunk in [0.0, 0.5, 1.0]:
                result = surrogates.flower_wind_damage(
                    wind=wind,
                    trunk=trunk,
                )
                assert float(result) >= 0.0
                # Max possible: alpha_flower * max_damage = 0.7 * 0.5 = 0.35
                assert float(result) <= 0.7 * 0.5 + 0.01


class TestLeafCrowdingPenalty:
    """Tests for leaf crowding penalty (shoot capacity constraint)."""

    def test_few_leaves_full_efficiency(self) -> None:
        """With leaves well below shoot capacity, growth efficiency is high."""
        # Capacity = 5.0 * 1.0 = 5.0 leaves; current leaves = 0.5
        result = surrogates.leaf_crowding_penalty(
            leaves=0.5,
            shoots=1.0,
            k_shoot_leaf=5.0,
            steepness=5.0,
            floor=0.1,
        )
        assert float(result) > 0.9  # Nearly full efficiency

    def test_overcrowded_reduced_efficiency(self) -> None:
        """With leaves exceeding shoot capacity, growth efficiency is reduced."""
        # Capacity = 5.0 * 1.0 = 5.0 leaves; current leaves = 10.0 (overcrowded)
        result = surrogates.leaf_crowding_penalty(
            leaves=10.0,
            shoots=1.0,
            k_shoot_leaf=5.0,
            steepness=5.0,
            floor=0.1,
        )
        # Crowding ratio = 10.0 / 5.0 = 2.0 → well overcrowded
        assert float(result) < 0.3  # Significantly reduced

    def test_floor_preserved_when_overcrowded(self) -> None:
        """Even when severely overcrowded, efficiency doesn't go below floor."""
        result = surrogates.leaf_crowding_penalty(
            leaves=100.0,  # Severely overcrowded
            shoots=1.0,
            k_shoot_leaf=5.0,
            steepness=5.0,
            floor=0.1,
        )
        assert float(result) >= 0.1  # Floor preserved

    def test_more_shoots_more_capacity(self) -> None:
        """More shoots means more leaf capacity."""
        # Same leaves, different shoot amounts
        eff_few_shoots = surrogates.leaf_crowding_penalty(
            leaves=5.0,
            shoots=0.5,  # Capacity = 2.5
            k_shoot_leaf=5.0,
        )
        eff_more_shoots = surrogates.leaf_crowding_penalty(
            leaves=5.0,
            shoots=2.0,  # Capacity = 10.0
            k_shoot_leaf=5.0,
        )
        # More shoots = more capacity = less crowding = higher efficiency
        assert float(eff_more_shoots) > float(eff_few_shoots)

    def test_at_capacity_half_penalty(self) -> None:
        """At exactly capacity (crowding=1), penalty should be around midpoint."""
        # Capacity = 5.0 * 1.0 = 5.0 leaves; current leaves = 5.0 (at capacity)
        result = surrogates.leaf_crowding_penalty(
            leaves=5.0,
            shoots=1.0,
            k_shoot_leaf=5.0,
            steepness=5.0,
            floor=0.1,
        )
        # At crowding=1.0: sigmoid(5.0 * 0) = 0.5
        # Efficiency = 0.1 + 0.9 * 0.5 = 0.55
        assert jnp.isclose(result, 0.55, atol=0.1)

    def test_zero_shoots_minimum_efficiency(self) -> None:
        """With no shoots, capacity is minimal, so efficiency is at floor."""
        result = surrogates.leaf_crowding_penalty(
            leaves=1.0,
            shoots=0.0,  # No shoots!
            k_shoot_leaf=5.0,
            steepness=5.0,
            floor=0.1,
        )
        # capacity ≈ 1e-6 (epsilon), crowding → infinity
        # sigmoid(5 * (1 - infinity)) → 0
        # efficiency → floor
        assert float(result) < 0.15  # Close to floor

    def test_steepness_affects_transition(self) -> None:
        """Higher steepness makes sharper transition at capacity."""
        leaves = 6.0  # Slightly over capacity of 5.0
        shoots = 1.0
        k = 5.0

        eff_soft = surrogates.leaf_crowding_penalty(
            leaves=leaves, shoots=shoots, k_shoot_leaf=k, steepness=2.0
        )
        eff_sharp = surrogates.leaf_crowding_penalty(
            leaves=leaves, shoots=shoots, k_shoot_leaf=k, steepness=10.0
        )
        # Sharper transition → lower efficiency when slightly overcrowded
        assert float(eff_sharp) < float(eff_soft)

    def test_output_bounded(self) -> None:
        """Efficiency is always in [floor, 1]."""
        for leaves in [0.0, 1.0, 5.0, 20.0]:
            for shoots in [0.0, 0.5, 1.0, 2.0]:
                result = surrogates.leaf_crowding_penalty(
                    leaves=leaves,
                    shoots=shoots,
                    k_shoot_leaf=5.0,
                    steepness=5.0,
                    floor=0.1,
                )
                assert float(result) >= 0.1  # Never below floor
                assert float(result) <= 1.0  # Never above 1


class TestFlowerCrowdingPenalty:
    """Tests for flower crowding penalty (shoot capacity constraint for flowers)."""

    def test_few_flowers_full_efficiency(self) -> None:
        """With flowers well below shoot capacity, growth efficiency is high."""
        # Capacity = 8.0 * 1.0 = 8.0 flowers; current flowers = 0.5
        result = surrogates.flower_crowding_penalty(
            flowers=0.5,
            shoots=1.0,
            k_shoot_flower=8.0,
            steepness=5.0,
            floor=0.1,
        )
        assert float(result) > 0.9  # Nearly full efficiency

    def test_overcrowded_reduced_efficiency(self) -> None:
        """With flowers exceeding shoot capacity, growth efficiency is reduced."""
        # Capacity = 8.0 * 1.0 = 8.0 flowers; current flowers = 16.0 (overcrowded)
        result = surrogates.flower_crowding_penalty(
            flowers=16.0,
            shoots=1.0,
            k_shoot_flower=8.0,
            steepness=5.0,
            floor=0.1,
        )
        # Crowding ratio = 16.0 / 8.0 = 2.0 → well overcrowded
        assert float(result) < 0.3  # Significantly reduced

    def test_floor_preserved_when_overcrowded(self) -> None:
        """Even when severely overcrowded, efficiency doesn't go below floor."""
        result = surrogates.flower_crowding_penalty(
            flowers=100.0,  # Severely overcrowded
            shoots=1.0,
            k_shoot_flower=8.0,
            steepness=5.0,
            floor=0.1,
        )
        assert float(result) >= 0.1  # Floor preserved

    def test_more_shoots_more_flower_capacity(self) -> None:
        """More shoots means more flower capacity."""
        # Same flowers, different shoot amounts
        eff_few_shoots = surrogates.flower_crowding_penalty(
            flowers=8.0,
            shoots=0.5,  # Capacity = 4.0
            k_shoot_flower=8.0,
        )
        eff_more_shoots = surrogates.flower_crowding_penalty(
            flowers=8.0,
            shoots=2.0,  # Capacity = 16.0
            k_shoot_flower=8.0,
        )
        # More shoots = more capacity = less crowding = higher efficiency
        assert float(eff_more_shoots) > float(eff_few_shoots)

    def test_zero_shoots_minimum_efficiency(self) -> None:
        """With no shoots, capacity is minimal, so efficiency is at floor."""
        result = surrogates.flower_crowding_penalty(
            flowers=1.0,
            shoots=0.0,  # No shoots!
            k_shoot_flower=8.0,
            steepness=5.0,
            floor=0.1,
        )
        # capacity ≈ 1e-6 (epsilon), crowding → infinity
        # efficiency → floor
        assert float(result) < 0.15  # Close to floor

    def test_output_bounded(self) -> None:
        """Efficiency is always in [floor, 1]."""
        for flowers in [0.0, 1.0, 8.0, 30.0]:
            for shoots in [0.0, 0.5, 1.0, 2.0]:
                result = surrogates.flower_crowding_penalty(
                    flowers=flowers,
                    shoots=shoots,
                    k_shoot_flower=8.0,
                    steepness=5.0,
                    floor=0.1,
                )
                assert float(result) >= 0.1  # Never below floor
                assert float(result) <= 1.0  # Never above 1
