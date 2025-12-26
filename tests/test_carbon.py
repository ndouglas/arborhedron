"""
Tests for carbon sequestration metrics.

These tests verify that carbon computations are correct, differentiable,
and produce sensible values for tree growth simulation.
"""

import jax
import jax.numpy as jnp

from sim import carbon
from sim.config import SimConfig, TreeState


def make_test_state(
    energy: float = 1.0,
    water: float = 0.5,
    nutrients: float = 0.5,
    roots: float = 0.5,
    trunk: float = 0.3,
    shoots: float = 0.2,
    leaves: float = 0.5,
    flowers: float = 0.1,
    fruit: float = 0.0,
    soil_water: float = 0.5,
) -> TreeState:
    """Create a test state with given values."""
    return TreeState(
        energy=jnp.array(energy),
        water=jnp.array(water),
        nutrients=jnp.array(nutrients),
        roots=jnp.array(roots),
        trunk=jnp.array(trunk),
        shoots=jnp.array(shoots),
        leaves=jnp.array(leaves),
        flowers=jnp.array(flowers),
        fruit=jnp.array(fruit),
        soil_water=jnp.array(soil_water),
    )


class TestCarbonContent:
    """Tests for carbon content computation."""

    def test_carbon_content_nonnegative(self) -> None:
        """Carbon content should be nonnegative for valid states."""
        config = SimConfig()
        state = make_test_state()

        content = carbon.compute_carbon_content(state, config)

        assert float(content["trunk_carbon"]) >= 0
        assert float(content["roots_carbon"]) >= 0
        assert float(content["shoots_carbon"]) >= 0
        assert float(content["leaves_carbon"]) >= 0
        assert float(content["flowers_carbon"]) >= 0
        assert float(content["total_carbon"]) >= 0
        assert float(content["permanent_carbon"]) >= 0
        assert float(content["seasonal_carbon"]) >= 0

    def test_carbon_content_sums_correctly(self) -> None:
        """Total carbon should equal sum of compartments."""
        config = SimConfig()
        state = make_test_state()

        content = carbon.compute_carbon_content(state, config)

        total_from_parts = (
            content["trunk_carbon"]
            + content["roots_carbon"]
            + content["shoots_carbon"]
            + content["leaves_carbon"]
            + content["flowers_carbon"]
        )

        assert jnp.isclose(content["total_carbon"], total_from_parts)

    def test_permanent_seasonal_partition(self) -> None:
        """Permanent + seasonal should equal total carbon."""
        config = SimConfig()
        state = make_test_state()

        content = carbon.compute_carbon_content(state, config)

        total_from_partition = content["permanent_carbon"] + content["seasonal_carbon"]

        assert jnp.isclose(content["total_carbon"], total_from_partition)

    def test_carbon_fractions_reasonable(self) -> None:
        """Carbon should be 40-50% of biomass."""
        config = SimConfig()
        state = make_test_state(trunk=1.0, roots=1.0, leaves=1.0)

        content = carbon.compute_carbon_content(state, config)

        # Trunk is 50% carbon
        assert jnp.isclose(content["trunk_carbon"], 0.5 * state.trunk)
        # Roots are 45% carbon
        assert jnp.isclose(content["roots_carbon"], 0.45 * state.roots)
        # Leaves are 45% carbon
        assert jnp.isclose(content["leaves_carbon"], 0.45 * state.leaves)

    def test_zero_biomass_zero_carbon(self) -> None:
        """Zero biomass should produce zero carbon."""
        config = SimConfig()
        state = make_test_state(
            roots=0.0, trunk=0.0, shoots=0.0, leaves=0.0, flowers=0.0
        )

        content = carbon.compute_carbon_content(state, config)

        assert float(content["total_carbon"]) == 0.0


class TestCarbonScore:
    """Tests for permanence-weighted carbon score."""

    def test_carbon_score_nonnegative(self) -> None:
        """Carbon score should be nonnegative for valid states."""
        config = SimConfig()
        state = make_test_state()

        score = carbon.compute_carbon_score(state, config)

        assert float(score) >= 0

    def test_trunk_contributes_more_than_leaves(self) -> None:
        """Equal mass in trunk should contribute more than in leaves."""
        config = SimConfig()

        # States with ONLY trunk or leaves (zero everything else)
        trunk_state = make_test_state(
            trunk=1.0, roots=0.0, shoots=0.0, leaves=0.0, flowers=0.0
        )
        leaf_state = make_test_state(
            trunk=0.0, roots=0.0, shoots=0.0, leaves=1.0, flowers=0.0
        )

        trunk_score = carbon.compute_carbon_score(trunk_state, config)
        leaf_score = carbon.compute_carbon_score(leaf_state, config)

        # Trunk has higher permanence (1.0 vs 0.1)
        assert float(trunk_score) > float(leaf_score)
        # Specifically: trunk contributes 0.5*1.0 = 0.5, leaves contribute 0.45*0.1 = 0.045
        assert float(trunk_score) > 10 * float(leaf_score)

    def test_carbon_score_increases_with_biomass(self) -> None:
        """More biomass should mean higher carbon score."""
        config = SimConfig()

        small_state = make_test_state(trunk=0.1, roots=0.1)
        large_state = make_test_state(trunk=1.0, roots=1.0)

        small_score = carbon.compute_carbon_score(small_state, config)
        large_score = carbon.compute_carbon_score(large_state, config)

        assert float(large_score) > float(small_score)

    def test_carbon_score_differentiable(self) -> None:
        """Gradients should flow through carbon_score."""
        config = SimConfig()

        def score_from_trunk(trunk_val: float) -> float:
            state = make_test_state(trunk=trunk_val)
            return carbon.compute_carbon_score(state, config)

        # Compute gradient
        grad_fn = jax.grad(score_from_trunk)
        gradient = grad_fn(0.5)

        # Gradient should be positive (more trunk = more score)
        assert float(gradient) > 0
        # Gradient should be finite
        assert jnp.isfinite(gradient)
        # Gradient should equal carbon_fraction * permanence
        expected_grad = config.carbon_fraction_trunk * config.permanence_trunk
        assert jnp.isclose(gradient, expected_grad)


class TestCarbonEfficiency:
    """Tests for carbon efficiency computation."""

    def test_efficiency_positive(self) -> None:
        """Efficiency should be positive for positive inputs."""
        efficiency = carbon.compute_carbon_efficiency(
            carbon_score=jnp.array(1.0),
            energy_invested=jnp.array(2.0),
        )

        assert float(efficiency) > 0

    def test_efficiency_scales_correctly(self) -> None:
        """Doubling carbon should double efficiency."""
        eff1 = carbon.compute_carbon_efficiency(
            carbon_score=jnp.array(1.0),
            energy_invested=jnp.array(2.0),
        )
        eff2 = carbon.compute_carbon_efficiency(
            carbon_score=jnp.array(2.0),
            energy_invested=jnp.array(2.0),
        )

        assert jnp.isclose(eff2, 2 * eff1)

    def test_efficiency_handles_zero_energy(self) -> None:
        """Should handle zero energy without NaN (uses epsilon)."""
        efficiency = carbon.compute_carbon_efficiency(
            carbon_score=jnp.array(1.0),
            energy_invested=jnp.array(0.0),
        )

        assert jnp.isfinite(efficiency)


class TestCarbonObjective:
    """Tests for carbon optimization objective."""

    def test_objective_zero_when_dead(self) -> None:
        """Dead tree (zero energy) should have ~zero objective."""
        config = SimConfig()
        state = make_test_state()

        objective = carbon.carbon_objective(
            final_state=state,
            config=config,
            carbon_integral=jnp.array(100.0),
            final_energy=jnp.array(0.0),  # Dead
        )

        # Energy gate should be near zero
        assert float(objective) < 0.1

    def test_objective_positive_when_alive(self) -> None:
        """Living tree should have positive objective."""
        config = SimConfig()
        state = make_test_state()

        objective = carbon.carbon_objective(
            final_state=state,
            config=config,
            carbon_integral=jnp.array(100.0),
            final_energy=jnp.array(1.0),  # Healthy
        )

        assert float(objective) > 0

    def test_objective_scales_with_integral(self) -> None:
        """More carbon integral should mean higher objective."""
        config = SimConfig()
        state = make_test_state()

        obj1 = carbon.carbon_objective(
            final_state=state,
            config=config,
            carbon_integral=jnp.array(50.0),
            final_energy=jnp.array(1.0),
        )
        obj2 = carbon.carbon_objective(
            final_state=state,
            config=config,
            carbon_integral=jnp.array(100.0),
            final_energy=jnp.array(1.0),
        )

        assert float(obj2) > float(obj1)

    def test_objective_differentiable(self) -> None:
        """Gradients should flow through objective."""
        config = SimConfig()
        state = make_test_state()

        def obj_from_integral(integral: float) -> float:
            return carbon.carbon_objective(
                final_state=state,
                config=config,
                carbon_integral=integral,
                final_energy=jnp.array(1.0),
            )

        grad_fn = jax.grad(obj_from_integral)
        gradient = grad_fn(50.0)

        assert jnp.isfinite(gradient)
        assert float(gradient) > 0  # More integral = higher objective


class TestCarbonSeedTradeoff:
    """Tests for multi-objective carbon/seed combination."""

    def test_weight_zero_is_pure_seeds(self) -> None:
        """Weight 0 should return only seeds."""
        result = carbon.carbon_seed_tradeoff(
            carbon_score=jnp.array(10.0),
            seeds=jnp.array(5.0),
            carbon_weight=0.0,
        )

        assert jnp.isclose(result, 5.0)

    def test_weight_one_is_pure_carbon(self) -> None:
        """Weight 1 should return only carbon."""
        result = carbon.carbon_seed_tradeoff(
            carbon_score=jnp.array(10.0),
            seeds=jnp.array(5.0),
            carbon_weight=1.0,
        )

        assert jnp.isclose(result, 10.0)

    def test_weight_half_is_average(self) -> None:
        """Weight 0.5 should return average."""
        result = carbon.carbon_seed_tradeoff(
            carbon_score=jnp.array(10.0),
            seeds=jnp.array(6.0),
            carbon_weight=0.5,
        )

        assert jnp.isclose(result, 8.0)  # 0.5*10 + 0.5*6 = 8


class TestCarbonSummary:
    """Tests for trajectory carbon summary."""

    def test_summary_handles_empty_trajectory(self) -> None:
        """Should return zeros for empty trajectory."""
        config = SimConfig()
        summary = carbon.compute_carbon_summary([], config)

        assert summary["FinalTotalCarbon"] == 0.0
        assert summary["CarbonIntegral"] == 0.0

    def test_summary_computes_all_metrics(self) -> None:
        """Should compute all expected metrics."""
        config = SimConfig()
        states = [make_test_state() for _ in range(10)]

        summary = carbon.compute_carbon_summary(states, config)

        # Check all expected keys exist
        expected_keys = [
            "FinalTotalCarbon",
            "FinalPermanentCarbon",
            "FinalSeasonalCarbon",
            "FinalCarbonScore",
            "PeakTotalCarbon",
            "PeakPermanentCarbon",
            "MeanCarbonScore",
            "CarbonIntegral",
            "TrunkCarbonFraction",
        ]
        for key in expected_keys:
            assert key in summary

    def test_summary_values_finite(self) -> None:
        """All summary values should be finite."""
        config = SimConfig()
        states = [make_test_state() for _ in range(10)]

        summary = carbon.compute_carbon_summary(states, config)

        for key, value in summary.items():
            assert jnp.isfinite(value), f"{key} is not finite"

    def test_integral_equals_sum_of_scores(self) -> None:
        """Carbon integral should equal sum of scores."""
        config = SimConfig()
        states = [make_test_state() for _ in range(5)]

        summary = carbon.compute_carbon_summary(states, config)

        # Manually compute sum
        expected_sum = sum(
            float(carbon.compute_carbon_score(s, config)) for s in states
        )

        assert jnp.isclose(summary["CarbonIntegral"], expected_sum)
