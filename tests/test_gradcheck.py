"""
Tests for gradient health checks.
"""

import jax.random as jr

from sim.config import SimConfig
from sim.gradcheck import gradient_health_report, sample_episode_distribution


class TestGradientHealth:
    """Tests for gradient health utilities."""

    def test_sample_distribution_shapes(self) -> None:
        """Sample distribution should have correct shapes."""
        key = jr.PRNGKey(0)
        samples = sample_episode_distribution(key, num_samples=100)

        assert samples["light"].shape == (100,)
        assert samples["water"].shape == (100,)
        assert samples["nutrients"].shape == (100,)
        assert samples["wind"].shape == (100,)
        assert samples["leaves"].shape == (100,)
        assert samples["energy"].shape == (100,)

    def test_gradient_report_structure(self) -> None:
        """Gradient report should have expected structure."""
        key = jr.PRNGKey(42)
        report = gradient_health_report(key, SimConfig(), num_samples=100)

        expected_inputs = ["light", "water", "nutrients", "leaves"]
        expected_metrics = ["mean", "std", "min", "max", "pct_near_zero", "pct_healthy"]

        for input_name in expected_inputs:
            assert input_name in report
            for metric in expected_metrics:
                assert metric in report[input_name]

    def test_gradient_floor_prevents_dead_zones(self) -> None:
        """Gradient floor should prevent excessive dead zones."""
        key = jr.PRNGKey(42)
        report = gradient_health_report(key, SimConfig(), num_samples=500)

        # With the floor, dead zones should be < 5%
        for name, metrics in report.items():
            assert metrics["pct_near_zero"] < 10.0, f"{name} has too many dead zones"

        # Healthy gradients should be > 70%
        for name, metrics in report.items():
            assert metrics["pct_healthy"] > 70.0, f"{name} has insufficient healthy gradients"

    def test_leaves_gradient_always_positive(self) -> None:
        """Leaves gradient should always be positive due to floor."""
        key = jr.PRNGKey(42)
        report = gradient_health_report(key, SimConfig(), num_samples=500)

        # Leaves min gradient should be > 0 (floor guarantees this)
        assert report["leaves"]["min"] > 0.01
        assert report["leaves"]["pct_healthy"] == 100.0
