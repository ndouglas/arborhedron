"""
Tests for environmental stress signal generation.

These tests verify that our sine-wave based stress signals
behave correctly and produce values in the expected ranges.
"""

import jax.numpy as jnp

from sim import stress
from sim.config import ClimateConfig, StressParams


class TestStressSignal:
    """Tests for individual stress signal generation."""

    def test_signal_at_zero_time(self) -> None:
        """Signal at t=0 depends on phase."""
        params = StressParams(offset=0.5, amplitude=0.2, frequency=1.0, phase=0.0)
        result = stress.compute_signal(params, t=0.0)
        # sin(0) = 0, so result = 0.5 + 0.2 * 0 = 0.5
        assert jnp.isclose(result, 0.5, atol=1e-5)

    def test_signal_with_phase_shift(self) -> None:
        """Phase shift affects the signal."""
        params = StressParams(
            offset=0.5, amplitude=0.2, frequency=1.0, phase=jnp.pi / 2
        )
        result = stress.compute_signal(params, t=0.0)
        # sin(π/2) = 1, so result = 0.5 + 0.2 * 1 = 0.7
        assert jnp.isclose(result, 0.7, atol=1e-5)

    def test_signal_bounded_zero_one(self) -> None:
        """Signal is clipped to [0, 1] range."""
        # Signal would go negative without clipping
        params = StressParams(offset=0.1, amplitude=0.5, frequency=1.0, phase=0.0)
        times = jnp.linspace(0, 10, 100)
        for t in times:
            result = stress.compute_signal(params, t=float(t))
            assert 0.0 <= float(result) <= 1.0

    def test_signal_oscillates(self) -> None:
        """Signal oscillates over time."""
        params = StressParams(offset=0.5, amplitude=0.2, frequency=0.5, phase=0.0)
        results = [stress.compute_signal(params, t=float(t)) for t in range(20)]
        # Check that there's variation (not constant)
        assert max(results) > min(results)

    def test_zero_amplitude_constant(self) -> None:
        """Zero amplitude gives constant signal."""
        params = StressParams(offset=0.6, amplitude=0.0, frequency=1.0, phase=0.0)
        results = [stress.compute_signal(params, t=float(t)) for t in range(10)]
        assert all(jnp.isclose(r, 0.6, atol=1e-5) for r in results)


class TestEnvironmentState:
    """Tests for combined environment state."""

    def test_environment_returns_three_values(self) -> None:
        """Environment state has light, moisture, wind."""
        config = ClimateConfig.mild()
        light, moisture, wind = stress.compute_environment(config, t=0.0)
        assert light is not None
        assert moisture is not None
        assert wind is not None

    def test_all_values_bounded(self) -> None:
        """All environmental values are in [0, 1]."""
        config = ClimateConfig.mild()
        for t in range(100):
            light, moisture, wind = stress.compute_environment(config, t=float(t))
            assert 0.0 <= float(light) <= 1.0
            assert 0.0 <= float(moisture) <= 1.0
            assert 0.0 <= float(wind) <= 1.0

    def test_droughty_has_low_moisture(self) -> None:
        """Droughty climate has lower average moisture."""
        mild = ClimateConfig.mild()
        droughty = ClimateConfig.droughty()

        mild_moisture = (
            sum(stress.compute_environment(mild, t=float(t))[1] for t in range(100))
            / 100
        )
        droughty_moisture = (
            sum(stress.compute_environment(droughty, t=float(t))[1] for t in range(100))
            / 100
        )

        assert droughty_moisture < mild_moisture

    def test_windy_has_high_wind(self) -> None:
        """Windy climate has higher average wind."""
        mild = ClimateConfig.mild()
        windy = ClimateConfig.windy()

        mild_wind = (
            sum(stress.compute_environment(mild, t=float(t))[2] for t in range(100))
            / 100
        )
        windy_wind = (
            sum(stress.compute_environment(windy, t=float(t))[2] for t in range(100))
            / 100
        )

        assert windy_wind > mild_wind


class TestEnvironmentBatch:
    """Tests for batched environment computation."""

    def test_batch_returns_arrays(self) -> None:
        """Batch computation returns JAX arrays."""
        config = ClimateConfig.mild()
        light, moisture, wind = stress.compute_environment_batch(config, num_days=10)
        assert light.shape == (10,)
        assert moisture.shape == (10,)
        assert wind.shape == (10,)

    def test_batch_matches_sequential(self) -> None:
        """Batch computation matches sequential computation."""
        config = ClimateConfig.mild()
        num_days = 20

        # Batch
        batch_light, batch_moisture, batch_wind = stress.compute_environment_batch(
            config, num_days=num_days
        )

        # Sequential
        for t in range(num_days):
            light, moisture, wind = stress.compute_environment(config, t=float(t))
            assert jnp.isclose(batch_light[t], light, atol=1e-5)
            assert jnp.isclose(batch_moisture[t], moisture, atol=1e-5)
            assert jnp.isclose(batch_wind[t], wind, atol=1e-5)


class TestPresetClimates:
    """Tests for preset climate configurations."""

    def test_mild_climate_values(self) -> None:
        """Mild climate has moderate values."""
        mild = ClimateConfig.mild()
        assert mild.light.offset > 0.5
        assert mild.moisture.offset > 0.4
        assert mild.wind.offset < 0.4

    def test_droughty_climate_values(self) -> None:
        """Droughty climate has low moisture offset."""
        droughty = ClimateConfig.droughty()
        assert droughty.moisture.offset < 0.4

    def test_windy_climate_values(self) -> None:
        """Windy climate has high wind offset and amplitude."""
        windy = ClimateConfig.windy()
        assert windy.wind.offset > 0.4
        assert windy.wind.amplitude > 0.2
