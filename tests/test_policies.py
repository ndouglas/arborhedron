"""
Tests for allocation policies.

These tests verify that policies produce valid allocation vectors
and exhibit expected behavior patterns.
"""

import jax.numpy as jnp
from jax import Array

from sim import policies
from sim.config import TreeState


def make_test_state(
    energy: float = 1.0,
    water: float = 0.5,
    nutrients: float = 0.5,
    roots: float = 0.5,
    trunk: float = 0.3,
    shoots: float = 0.2,
    leaves: float = 0.5,
    flowers: float = 0.0,
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
        soil_water=jnp.array(soil_water),
    )


class TestBaselinePolicy:
    """Tests for the hand-coded baseline policy."""

    def test_returns_valid_allocation(self) -> None:
        """Policy should return a valid allocation."""
        state = make_test_state()
        allocation = policies.baseline_policy(state, day=0, num_days=100)
        assert allocation.is_valid()

    def test_allocations_sum_to_one(self) -> None:
        """Allocations should sum to 1."""
        state = make_test_state()
        for day in range(100):
            allocation = policies.baseline_policy(state, day=day, num_days=100)
            total = (
                allocation.roots
                + allocation.trunk
                + allocation.shoots
                + allocation.leaves
                + allocation.flowers
            )
            assert jnp.isclose(total, 1.0, atol=1e-5)

    def test_all_allocations_nonnegative(self) -> None:
        """All allocation fractions should be nonnegative."""
        state = make_test_state()
        for day in range(100):
            allocation = policies.baseline_policy(state, day=day, num_days=100)
            assert allocation.roots >= 0
            assert allocation.trunk >= 0
            assert allocation.shoots >= 0
            assert allocation.leaves >= 0
            assert allocation.flowers >= 0

    def test_early_days_favor_roots_and_leaves(self) -> None:
        """Early days should prioritize roots and leaves."""
        state = make_test_state()
        allocation = policies.baseline_policy(state, day=0, num_days=100)

        # Early: roots + leaves should dominate
        root_leaf = allocation.roots + allocation.leaves
        trunk_flower = allocation.trunk + allocation.flowers
        assert root_leaf > trunk_flower

    def test_late_days_favor_trunk_and_flowers(self) -> None:
        """Later days should shift toward trunk and flowers."""
        state = make_test_state()
        early = policies.baseline_policy(state, day=10, num_days=100)
        late = policies.baseline_policy(state, day=90, num_days=100)

        # Late should have more trunk and flowers than early
        assert late.trunk + late.flowers > early.trunk + early.flowers

    def test_wind_response(self) -> None:
        """Policy should respond to high wind conditions."""
        state = make_test_state()
        low_wind = policies.baseline_policy(state, day=50, num_days=100, wind=0.2)
        high_wind = policies.baseline_policy(state, day=50, num_days=100, wind=0.8)

        # High wind should increase trunk allocation for protection
        assert high_wind.trunk >= low_wind.trunk


class TestSoftmaxAllocation:
    """Tests for the softmax allocation helper."""

    def test_produces_valid_allocation(self) -> None:
        """Softmax should produce valid allocations."""
        logits = jnp.array([1.0, 0.5, 0.3, 0.2, 0.0])
        allocation = policies.softmax_allocation(logits)
        assert allocation.is_valid()

    def test_equal_logits_equal_allocation(self) -> None:
        """Equal logits should give equal allocations."""
        logits = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        allocation = policies.softmax_allocation(logits)

        # All should be approximately 0.2
        assert jnp.isclose(allocation.roots, 0.2, atol=0.01)
        assert jnp.isclose(allocation.trunk, 0.2, atol=0.01)
        assert jnp.isclose(allocation.shoots, 0.2, atol=0.01)
        assert jnp.isclose(allocation.leaves, 0.2, atol=0.01)
        assert jnp.isclose(allocation.flowers, 0.2, atol=0.01)

    def test_higher_logit_higher_allocation(self) -> None:
        """Higher logit should give higher allocation."""
        logits = jnp.array([10.0, 0.0, 0.0, 0.0, 0.0])  # Strong preference for roots
        allocation = policies.softmax_allocation(logits)

        # Roots should dominate
        assert allocation.roots > 0.9


class TestConstantPolicy:
    """Tests for constant (fixed) allocation policies."""

    def test_growth_focused_policy(self) -> None:
        """Growth-focused policy should allocate to roots and leaves."""
        state = make_test_state()
        allocation = policies.growth_focused_policy(state, day=0, num_days=100)

        assert allocation.is_valid()
        # Roots and leaves should dominate
        assert allocation.roots + allocation.leaves > 0.6

    def test_defensive_policy(self) -> None:
        """Defensive policy should allocate to trunk."""
        state = make_test_state()
        allocation = policies.defensive_policy(state, day=0, num_days=100)

        assert allocation.is_valid()
        # Trunk should be significant
        assert allocation.trunk > 0.3

    def test_reproduction_policy(self) -> None:
        """Reproduction policy should allocate to flowers."""
        state = make_test_state()
        allocation = policies.reproduction_policy(state, day=0, num_days=100)

        assert allocation.is_valid()
        # Flowers should be significant
        assert allocation.flowers > 0.3


class TestPolicyDifferentiability:
    """Tests for gradient flow through policies."""

    def test_softmax_gradients_exist(self) -> None:
        """Gradients should flow through softmax allocation."""
        import jax

        def loss_fn(logit: Array) -> Array:
            logits = jnp.array([logit, 0.0, 0.0, 0.0, 0.0])
            allocation = policies.softmax_allocation(logits)
            return allocation.roots

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(jnp.array(0.0))

        assert jnp.isfinite(grad)
        assert grad > 0  # Higher logit should increase roots
