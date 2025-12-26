"""
Tests for resilience and tipping point analysis.

These tests verify that the resilience analysis tools work correctly
and produce sensible results for the tree growth simulation.
"""

import jax.numpy as jnp
import numpy as np

from sim import resilience
from sim.config import ClimateConfig, SimConfig


def simple_fitness_fn(climate: ClimateConfig) -> float:
    """
    Simple test fitness function.

    Higher moisture and lower wind = better fitness.
    This creates a predictable gradient for testing.
    """
    # Fitness increases with moisture, decreases with wind
    base = 1.0
    moisture_bonus = climate.moisture.offset * 0.5
    wind_penalty = climate.wind.offset * 0.8
    return max(0.0, base + moisture_bonus - wind_penalty)


def quadratic_fitness_fn(climate: ClimateConfig) -> float:
    """
    Fitness function with a clear optimum.

    Has a peak at moisture=0.6, wind=0.3
    """
    moisture_opt = 0.6
    wind_opt = 0.3

    moisture_dist = (climate.moisture.offset - moisture_opt) ** 2
    wind_dist = (climate.wind.offset - wind_opt) ** 2

    return max(0.0, 1.0 - moisture_dist - wind_dist)


class TestMakeClimateWithOffsets:
    """Tests for climate modification function."""

    def test_creates_valid_climate(self) -> None:
        """Should create a valid ClimateConfig."""
        base = ClimateConfig.mild()
        modified = resilience.make_climate_with_offsets(base, 0.5, 0.3)

        assert isinstance(modified, ClimateConfig)
        assert modified.moisture.offset == 0.5
        assert modified.wind.offset == 0.3

    def test_preserves_other_params(self) -> None:
        """Should preserve light and other parameters."""
        base = ClimateConfig.mild()
        modified = resilience.make_climate_with_offsets(base, 0.5, 0.3)

        # Light should be unchanged
        assert modified.light.offset == base.light.offset
        assert modified.light.amplitude == base.light.amplitude

        # Amplitudes and frequencies preserved
        assert modified.moisture.amplitude == base.moisture.amplitude
        assert modified.wind.frequency == base.wind.frequency


class TestParameterSweep2D:
    """Tests for 2D parameter sweep."""

    def test_returns_correct_shape(self) -> None:
        """Sweep should return resolution × resolution grid."""
        base = ClimateConfig.mild()
        result = resilience.parameter_sweep_2d(
            simple_fitness_fn,
            base,
            resolution=5,
        )

        assert result["fitness_grid"].shape == (5, 5)
        assert result["moisture_grid"].shape == (5, 5)
        assert result["wind_grid"].shape == (5, 5)

    def test_fitness_values_in_range(self) -> None:
        """Fitness values should be nonnegative."""
        base = ClimateConfig.mild()
        result = resilience.parameter_sweep_2d(
            simple_fitness_fn,
            base,
            resolution=5,
        )

        assert np.all(result["fitness_grid"] >= 0)

    def test_fitness_varies_with_params(self) -> None:
        """Fitness should vary across the grid."""
        base = ClimateConfig.mild()
        result = resilience.parameter_sweep_2d(
            simple_fitness_fn,
            base,
            resolution=10,
        )

        # Should have variation (not all same value)
        fitness_std = np.std(result["fitness_grid"])
        assert fitness_std > 0

    def test_respects_custom_ranges(self) -> None:
        """Should use specified parameter ranges."""
        base = ClimateConfig.mild()
        result = resilience.parameter_sweep_2d(
            simple_fitness_fn,
            base,
            moisture_range=(0.3, 0.7),
            wind_range=(0.2, 0.6),
            resolution=5,
        )

        assert result["moisture_vals"].min() == 0.3
        assert result["moisture_vals"].max() == 0.7
        assert result["wind_vals"].min() == 0.2
        assert result["wind_vals"].max() == 0.6


class TestComputeSensitivity:
    """Tests for gradient-based sensitivity computation."""

    def test_moisture_sensitivity_positive(self) -> None:
        """More moisture should increase fitness (for simple_fitness_fn)."""
        base = ClimateConfig.mild()
        sensitivity = resilience.compute_sensitivity(
            simple_fitness_fn, base, "moisture_offset"
        )

        # simple_fitness_fn has positive moisture coefficient
        assert float(sensitivity) > 0

    def test_wind_sensitivity_negative(self) -> None:
        """More wind should decrease fitness (for simple_fitness_fn)."""
        base = ClimateConfig.mild()
        sensitivity = resilience.compute_sensitivity(
            simple_fitness_fn, base, "wind_offset"
        )

        # simple_fitness_fn has negative wind coefficient
        assert float(sensitivity) < 0

    def test_sensitivity_finite(self) -> None:
        """Sensitivity should be finite."""
        base = ClimateConfig.mild()
        moisture_sens = resilience.compute_sensitivity(
            simple_fitness_fn, base, "moisture_offset"
        )
        wind_sens = resilience.compute_sensitivity(
            simple_fitness_fn, base, "wind_offset"
        )

        assert jnp.isfinite(moisture_sens)
        assert jnp.isfinite(wind_sens)

    def test_invalid_param_raises(self) -> None:
        """Should raise error for invalid parameter name."""
        import pytest

        base = ClimateConfig.mild()
        with pytest.raises(ValueError, match="Unknown parameter"):
            resilience.compute_sensitivity(simple_fitness_fn, base, "invalid_param")


class TestSensitivitySweep:
    """Tests for sensitivity sweep across parameter range."""

    def test_returns_correct_length(self) -> None:
        """Should return arrays of specified resolution."""
        base = ClimateConfig.mild()
        result = resilience.sensitivity_sweep(
            simple_fitness_fn,
            base,
            "moisture_offset",
            (0.2, 0.8),
            resolution=10,
        )

        assert len(result["param_values"]) == 10
        assert len(result["fitness_values"]) == 10
        assert len(result["gradient_values"]) == 10

    def test_gradient_sign_consistent(self) -> None:
        """Gradient signs should be consistent with fitness function."""
        base = ClimateConfig.mild()

        # Moisture sweep
        moisture_result = resilience.sensitivity_sweep(
            simple_fitness_fn,
            base,
            "moisture_offset",
            (0.2, 0.8),
            resolution=10,
        )

        # For simple_fitness_fn, gradient should be positive everywhere
        assert np.all(np.array(moisture_result["gradient_values"]) >= -0.1)

    def test_fitness_monotonic_for_simple_fn(self) -> None:
        """Fitness should be monotonic for simple fitness function."""
        base = ClimateConfig.mild()

        moisture_result = resilience.sensitivity_sweep(
            simple_fitness_fn,
            base,
            "moisture_offset",
            (0.2, 0.8),
            resolution=10,
        )

        # Fitness should generally increase with moisture
        fitness = np.array(moisture_result["fitness_values"])
        # Allow some numerical noise
        assert fitness[-1] > fitness[0]


class TestFindTippingPoints:
    """Tests for tipping point detection."""

    def test_returns_list(self) -> None:
        """Should return a list (possibly empty)."""
        sweep = {
            "param_values": np.linspace(0, 1, 10),
            "fitness_values": np.linspace(1, 0, 10),  # Linear decrease
            "gradient_values": np.full(10, -1.0),  # Constant gradient
        }

        tipping_points = resilience.find_tipping_points(sweep)
        assert isinstance(tipping_points, list)

    def test_detects_gradient_spike(self) -> None:
        """Should detect when gradient spikes."""
        param_values = np.linspace(0, 1, 20)
        fitness_values = np.ones(20)
        gradient_values = np.zeros(20)

        # Add a gradient spike in the middle
        gradient_values[10] = 10.0  # Big spike

        sweep = {
            "param_values": param_values,
            "fitness_values": fitness_values,
            "gradient_values": gradient_values,
        }

        tipping_points = resilience.find_tipping_points(sweep, gradient_threshold=2.0)
        assert len(tipping_points) >= 1

    def test_detects_fitness_drop(self) -> None:
        """Should detect significant fitness drops."""
        param_values = np.linspace(0, 1, 20)
        fitness_values = np.ones(20)
        gradient_values = np.zeros(20)

        # Add a fitness drop
        fitness_values[10:] = 0.2  # Big drop

        sweep = {
            "param_values": param_values,
            "fitness_values": fitness_values,
            "gradient_values": gradient_values,
        }

        tipping_points = resilience.find_tipping_points(
            sweep, fitness_drop_threshold=0.3
        )
        assert len(tipping_points) >= 1


class TestComputeResilienceBoundary:
    """Tests for resilience boundary computation."""

    def test_returns_all_required_keys(self) -> None:
        """Should return all expected keys."""
        base = ClimateConfig.mild()
        result = resilience.compute_resilience_boundary(
            simple_fitness_fn,
            base,
            resolution=5,
        )

        assert "fitness_grid" in result
        assert "survival_mask" in result
        assert "boundary_points" in result
        assert "survival_threshold" in result

    def test_survival_mask_is_boolean(self) -> None:
        """Survival mask should be boolean array."""
        base = ClimateConfig.mild()
        result = resilience.compute_resilience_boundary(
            simple_fitness_fn,
            base,
            resolution=5,
        )

        assert result["survival_mask"].dtype == bool

    def test_boundary_points_tuple(self) -> None:
        """Boundary points should be tuple of arrays."""
        base = ClimateConfig.mild()
        result = resilience.compute_resilience_boundary(
            quadratic_fitness_fn,
            base,
            resolution=10,
        )

        boundary_points = result["boundary_points"]
        assert isinstance(boundary_points, tuple)
        assert len(boundary_points) == 2


class TestResilienceReport:
    """Tests for comprehensive resilience report."""

    def test_returns_all_sections(self) -> None:
        """Should return all expected sections."""
        base = ClimateConfig.mild()
        config = SimConfig()

        report = resilience.resilience_report(
            simple_fitness_fn,
            base,
            config,
        )

        assert "landscape" in report
        assert "moisture_sensitivity" in report
        assert "wind_sensitivity" in report
        assert "moisture_tipping_points" in report
        assert "wind_tipping_points" in report
        assert "boundary" in report
        assert "summary" in report

    def test_summary_has_statistics(self) -> None:
        """Summary should contain key statistics."""
        base = ClimateConfig.mild()
        config = SimConfig()

        report = resilience.resilience_report(
            simple_fitness_fn,
            base,
            config,
        )

        summary = report["summary"]
        assert "max_fitness" in summary
        assert "min_fitness" in summary
        assert "mean_fitness" in summary
        assert "survival_fraction" in summary

    def test_values_are_finite(self) -> None:
        """All numeric values should be finite."""
        base = ClimateConfig.mild()
        config = SimConfig()

        report = resilience.resilience_report(
            simple_fitness_fn,
            base,
            config,
        )

        for key, value in report["summary"].items():
            assert np.isfinite(value), f"{key} is not finite"


class TestWithRealSimulation:
    """Tests using the actual simulation (slower but more realistic)."""

    def test_sweep_with_real_simulation(self) -> None:
        """Should work with actual run_season."""
        from sim import policies, rollout

        config = SimConfig(num_days=20)  # Short for speed

        def real_fitness_fn(climate: ClimateConfig) -> float:
            trajectory = rollout.run_season(config, climate, policies.baseline_policy)
            return float(trajectory.seeds)

        base = ClimateConfig.mild()
        result = resilience.parameter_sweep_2d(
            real_fitness_fn,
            base,
            resolution=3,  # Very small for speed
        )

        assert result["fitness_grid"].shape == (3, 3)
        assert np.all(np.isfinite(result["fitness_grid"]))
