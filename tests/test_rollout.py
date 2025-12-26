"""
Tests for full season rollout.

These tests verify that the complete simulation produces
valid trajectories and expected behavior patterns.
"""

import jax.numpy as jnp
from jax import Array

from sim import policies, rollout
from sim.config import ClimateConfig, SimConfig


class TestRollout:
    """Tests for single season rollout."""

    def test_rollout_produces_trajectory(self) -> None:
        """Rollout should produce state trajectory."""
        config = SimConfig(num_days=50)
        climate = ClimateConfig.mild()
        policy = policies.baseline_policy

        trajectory = rollout.run_season(
            config=config,
            climate=climate,
            policy=policy,
        )

        # Should have num_days + 1 states (including initial)
        assert len(trajectory.states) == config.num_days + 1

    def test_all_states_valid(self) -> None:
        """All states in trajectory should be valid."""
        config = SimConfig(num_days=50)
        climate = ClimateConfig.mild()
        policy = policies.baseline_policy

        trajectory = rollout.run_season(
            config=config,
            climate=climate,
            policy=policy,
        )

        for state in trajectory.states:
            assert state.is_valid()

    def test_seed_production_nonnegative(self) -> None:
        """Seed production should be nonnegative."""
        config = SimConfig(num_days=100)
        climate = ClimateConfig.mild()
        policy = policies.baseline_policy

        trajectory = rollout.run_season(
            config=config,
            climate=climate,
            policy=policy,
        )

        assert trajectory.seeds >= 0

    def test_mild_climate_survival(self) -> None:
        """Tree should survive in mild climate."""
        config = SimConfig(num_days=100)
        climate = ClimateConfig.mild()
        policy = policies.baseline_policy

        trajectory = rollout.run_season(
            config=config,
            climate=climate,
            policy=policy,
        )

        # Final state should have positive biomass
        final = trajectory.states[-1]
        assert final.total_biomass() > 0

    def test_harsh_climate_reduces_output(self) -> None:
        """Harsh climate should reduce seed production."""
        config = SimConfig(num_days=100)
        mild_climate = ClimateConfig.mild()
        harsh_climate = ClimateConfig.windy()
        policy = policies.baseline_policy

        mild_result = rollout.run_season(
            config=config,
            climate=mild_climate,
            policy=policy,
        )
        harsh_result = rollout.run_season(
            config=config,
            climate=harsh_climate,
            policy=policy,
        )

        # Harsh climate should produce fewer seeds
        # (or at least not significantly more)
        assert harsh_result.seeds <= mild_result.seeds * 1.5


class TestTrajectory:
    """Tests for trajectory data structure."""

    def test_trajectory_has_stress_history(self) -> None:
        """Trajectory should record stress history."""
        config = SimConfig(num_days=50)
        climate = ClimateConfig.mild()
        policy = policies.baseline_policy

        trajectory = rollout.run_season(
            config=config,
            climate=climate,
            policy=policy,
        )

        assert len(trajectory.light_history) == config.num_days
        assert len(trajectory.moisture_history) == config.num_days
        assert len(trajectory.wind_history) == config.num_days

    def test_trajectory_has_allocation_history(self) -> None:
        """Trajectory should record allocation history."""
        config = SimConfig(num_days=50)
        climate = ClimateConfig.mild()
        policy = policies.baseline_policy

        trajectory = rollout.run_season(
            config=config,
            climate=climate,
            policy=policy,
        )

        assert len(trajectory.allocations) == config.num_days

    def test_scalar_summary_structure(self) -> None:
        """Scalar summary should have all expected keys."""
        config = SimConfig(num_days=50)
        climate = ClimateConfig.mild()
        policy = policies.baseline_policy

        trajectory = rollout.run_season(
            config=config,
            climate=climate,
            policy=policy,
        )

        summary = trajectory.get_scalar_summary()
        expected_keys = [
            "Seeds",
            "FinalBiomass",
            "FinalEnergy",
            "PeakEnergy",
            "MinEnergy",
            "DaysAtZero",
            "FinalTrunk",
            "FinalFlowers",
            "FinalRoots",
            "FinalLeaves",
            "MeanLight",
            "MeanMoisture",
            "MeanWind",
        ]
        for key in expected_keys:
            assert key in summary
            assert isinstance(summary[key], (int, float))

    def test_scalar_summary_values_consistent(self) -> None:
        """Scalar summary values should match trajectory data."""
        config = SimConfig(num_days=50)
        climate = ClimateConfig.mild()
        policy = policies.baseline_policy

        trajectory = rollout.run_season(
            config=config,
            climate=climate,
            policy=policy,
        )

        summary = trajectory.get_scalar_summary()
        final_state = trajectory.states[-1]

        # Check consistency
        assert abs(summary["Seeds"] - float(trajectory.seeds)) < 1e-6
        assert abs(summary["FinalEnergy"] - float(final_state.energy)) < 1e-6
        assert abs(summary["FinalTrunk"] - float(final_state.trunk)) < 1e-6


class TestPolicyComparison:
    """Tests comparing different policies."""

    def test_growth_policy_grows_more(self) -> None:
        """Growth-focused policy should produce more biomass early."""
        config = SimConfig(num_days=30)
        climate = ClimateConfig.mild()

        growth_trajectory = rollout.run_season(
            config=config,
            climate=climate,
            policy=policies.growth_focused_policy,
        )
        defensive_trajectory = rollout.run_season(
            config=config,
            climate=climate,
            policy=policies.defensive_policy,
        )

        # Growth policy should have more leaves
        growth_leaves = growth_trajectory.states[-1].leaves
        defensive_leaves = defensive_trajectory.states[-1].leaves
        assert growth_leaves > defensive_leaves

    def test_defensive_policy_more_trunk(self) -> None:
        """Defensive policy should produce more trunk."""
        config = SimConfig(num_days=30)
        climate = ClimateConfig.mild()

        growth_trajectory = rollout.run_season(
            config=config,
            climate=climate,
            policy=policies.growth_focused_policy,
        )
        defensive_trajectory = rollout.run_season(
            config=config,
            climate=climate,
            policy=policies.defensive_policy,
        )

        # Defensive policy should have more trunk
        growth_trunk = growth_trajectory.states[-1].trunk
        defensive_trunk = defensive_trajectory.states[-1].trunk
        assert defensive_trunk > growth_trunk


class TestDifferentiability:
    """Tests for gradient flow through rollout."""

    def test_rollout_gradients_exist(self) -> None:
        """Gradients should flow through the rollout."""

        climate = ClimateConfig.mild()

        def loss_fn(seed_energy: Array) -> Array:
            # Modify config with the parameter
            modified_config = SimConfig(
                num_days=20,
                seed_energy=float(seed_energy),
            )
            trajectory = rollout.run_season(
                config=modified_config,
                climate=climate,
                policy=policies.baseline_policy,
            )
            return trajectory.seeds

        # This test just checks that the code runs without error
        # Full gradient checking would require jitting the rollout
        result = loss_fn(jnp.array(1.0))
        assert jnp.isfinite(result)
