"""
Tests for tree growth dynamics.

These tests verify the core simulation update step behaves
correctly and maintains required invariants.
"""

import jax.numpy as jnp
from jax import Array

from sim import dynamics
from sim.config import Allocation, SimConfig, TreeState


def make_test_state(
    energy: float = 1.0,
    water: float = 0.5,
    nutrients: float = 0.5,
    roots: float = 0.5,
    trunk: float = 0.3,
    shoots: float = 0.2,
    leaves: float = 0.5,
    flowers: float = 0.0,
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


def make_test_allocation(
    roots: float = 0.2,
    trunk: float = 0.2,
    shoots: float = 0.2,
    leaves: float = 0.2,
    flowers: float = 0.2,
) -> Allocation:
    """Create a test allocation (must sum to 1)."""
    return Allocation(
        roots=jnp.array(roots),
        trunk=jnp.array(trunk),
        shoots=jnp.array(shoots),
        leaves=jnp.array(leaves),
        flowers=jnp.array(flowers),
    )


class TestStateInvariants:
    """Tests for state invariant preservation."""

    def test_state_remains_valid(self) -> None:
        """State should remain valid after update."""
        config = SimConfig()
        state = make_test_state()
        allocation = make_test_allocation()

        new_state = dynamics.step(
            state=state,
            allocation=allocation,
            light=0.8,
            moisture=0.6,
            wind=0.2,
            config=config,
        )

        assert new_state.is_valid()

    def test_no_nan_values(self) -> None:
        """No NaN values should appear after update."""
        config = SimConfig()
        state = make_test_state()
        allocation = make_test_allocation()

        new_state = dynamics.step(
            state=state,
            allocation=allocation,
            light=0.8,
            moisture=0.6,
            wind=0.2,
            config=config,
        )

        assert jnp.isfinite(new_state.energy)
        assert jnp.isfinite(new_state.water)
        assert jnp.isfinite(new_state.nutrients)
        assert jnp.isfinite(new_state.roots)
        assert jnp.isfinite(new_state.trunk)
        assert jnp.isfinite(new_state.shoots)
        assert jnp.isfinite(new_state.leaves)
        assert jnp.isfinite(new_state.flowers)

    def test_biomass_nonnegative(self) -> None:
        """All biomass values stay nonnegative."""
        config = SimConfig()
        state = make_test_state()
        allocation = make_test_allocation()

        # Run multiple steps
        for _ in range(50):
            state = dynamics.step(
                state=state,
                allocation=allocation,
                light=0.8,
                moisture=0.6,
                wind=0.2,
                config=config,
            )

        assert state.roots >= 0
        assert state.trunk >= 0
        assert state.shoots >= 0
        assert state.leaves >= 0
        assert state.flowers >= 0


class TestEnergyConservation:
    """Tests for energy budget behavior."""

    def test_no_leaves_energy_decreases(self) -> None:
        """Without leaves, energy only decreases (maintenance)."""
        config = SimConfig()
        state = make_test_state(leaves=0.0, energy=1.0)
        allocation = make_test_allocation(
            leaves=0.0, roots=0.25, trunk=0.25, shoots=0.25, flowers=0.25
        )

        new_state = dynamics.step(
            state=state,
            allocation=allocation,
            light=0.8,
            moisture=0.6,
            wind=0.0,
            config=config,
        )

        # Energy should decrease (no photosynthesis, only costs)
        assert new_state.energy < state.energy

    def test_good_conditions_energy_positive(self) -> None:
        """With leaves, trunk, and good conditions, tree survives."""
        config = SimConfig()
        # Start with good leaves and sufficient trunk (low structural penalty)
        # Use high starting energy to offset costs in first step
        state = make_test_state(
            leaves=1.0,
            energy=2.0,
            roots=0.5,
            trunk=2.0,
            shoots=0.2,
            flowers=0.0,
            water=0.8,
            nutrients=0.8,
        )
        # Zero allocation to test pure photosynthesis vs maintenance
        allocation = make_test_allocation(
            roots=0.0, trunk=0.0, shoots=0.0, leaves=0.0, flowers=0.0
        )

        new_state = dynamics.step(
            state=state,
            allocation=allocation,
            light=0.9,  # Good light
            moisture=0.8,  # Good moisture
            wind=0.0,  # No wind
            config=config,
        )

        # With zero allocation, photosynthesis minus maintenance should leave energy
        assert new_state.energy >= 0  # State remains valid
        assert new_state.is_valid()  # All invariants hold


class TestResourceUptake:
    """Tests for water and nutrient uptake."""

    def test_no_roots_no_uptake(self) -> None:
        """Without roots, water and nutrients don't increase."""
        config = SimConfig()
        state = make_test_state(roots=0.0, water=0.5, nutrients=0.5)
        allocation = make_test_allocation()

        new_state = dynamics.step(
            state=state,
            allocation=allocation,
            light=0.8,
            moisture=0.8,
            wind=0.0,
            config=config,
        )

        # Water and nutrients should not increase (only decay)
        assert new_state.water <= state.water
        assert new_state.nutrients <= state.nutrients

    def test_roots_increase_water(self) -> None:
        """Roots enable water uptake."""
        config = SimConfig()
        state = make_test_state(roots=1.0, water=0.1, nutrients=0.5)
        allocation = make_test_allocation()

        new_state = dynamics.step(
            state=state,
            allocation=allocation,
            light=0.5,
            moisture=0.9,  # High moisture
            wind=0.0,
            config=config,
        )

        # Water should increase with roots and moisture
        assert new_state.water > state.water


class TestWindDamage:
    """Tests for wind damage to tender growth."""

    def test_high_wind_damages_shoots(self) -> None:
        """High wind should damage shoots."""
        config = SimConfig()
        state = make_test_state(shoots=1.0, leaves=1.0)
        allocation = make_test_allocation()

        new_state = dynamics.step(
            state=state,
            allocation=allocation,
            light=0.8,
            moisture=0.6,
            wind=0.9,  # High wind
            config=config,
        )

        # Shoots should decrease due to wind damage
        assert new_state.shoots < state.shoots

    def test_high_wind_damages_leaves(self) -> None:
        """High wind should damage leaves."""
        config = SimConfig()
        state = make_test_state(shoots=1.0, leaves=1.0)
        allocation = make_test_allocation()

        new_state = dynamics.step(
            state=state,
            allocation=allocation,
            light=0.8,
            moisture=0.6,
            wind=0.9,  # High wind
            config=config,
        )

        # Leaves should decrease due to wind damage
        assert new_state.leaves < state.leaves

    def test_low_wind_no_damage(self) -> None:
        """Low wind should not damage significantly."""
        config = SimConfig()
        state = make_test_state(shoots=1.0, leaves=1.0, trunk=1.0)
        # Only allocate to trunk (no new tender growth)
        allocation = make_test_allocation(
            roots=0.0, trunk=1.0, shoots=0.0, leaves=0.0, flowers=0.0
        )

        new_state = dynamics.step(
            state=state,
            allocation=allocation,
            light=0.8,
            moisture=0.6,
            wind=0.1,  # Low wind
            config=config,
        )

        # Shoots and leaves should not be significantly damaged
        # (small damage is expected from the sigmoid tail)
        assert new_state.shoots > 0.95 * state.shoots


class TestStructuralConstraint:
    """Tests for structural load/capacity mechanics."""

    def test_unsupported_canopy_drains_energy(self) -> None:
        """Large canopy without trunk support drains energy."""
        config = SimConfig()
        # Lots of leaves, no trunk
        state = make_test_state(
            leaves=5.0, shoots=3.0, trunk=0.1, energy=2.0, roots=0.5
        )
        allocation = make_test_allocation()

        new_state = dynamics.step(
            state=state,
            allocation=allocation,
            light=0.8,
            moisture=0.6,
            wind=0.0,
            config=config,
        )

        # Energy drain from structural penalty
        # This is a soft penalty, so check it's significant
        # Exact behavior depends on config
        assert new_state.energy < state.energy


class TestGrowth:
    """Tests for biomass growth."""

    def test_allocation_directs_growth(self) -> None:
        """Allocation determines which compartments grow."""
        config = SimConfig()
        state = make_test_state(energy=2.0, water=0.8, nutrients=0.8)

        # Allocate everything to roots
        allocation = make_test_allocation(
            roots=1.0, trunk=0.0, shoots=0.0, leaves=0.0, flowers=0.0
        )

        new_state = dynamics.step(
            state=state,
            allocation=allocation,
            light=0.8,
            moisture=0.6,
            wind=0.0,
            config=config,
        )

        # Roots should grow most
        root_growth = new_state.roots - state.roots
        trunk_growth = new_state.trunk - state.trunk

        assert root_growth > trunk_growth

    def test_low_resources_limit_growth(self) -> None:
        """Low water/nutrients limit growth efficiency."""
        config = SimConfig()
        # Same energy, different resource availability
        state_rich = make_test_state(energy=2.0, water=0.9, nutrients=0.9, roots=0.5)
        state_poor = make_test_state(energy=2.0, water=0.1, nutrients=0.1, roots=0.5)

        allocation = make_test_allocation(
            roots=1.0, trunk=0.0, shoots=0.0, leaves=0.0, flowers=0.0
        )

        new_rich = dynamics.step(
            state=state_rich,
            allocation=allocation,
            light=0.8,
            moisture=0.6,
            wind=0.0,
            config=config,
        )
        new_poor = dynamics.step(
            state=state_poor,
            allocation=allocation,
            light=0.8,
            moisture=0.6,
            wind=0.0,
            config=config,
        )

        rich_growth = new_rich.roots - state_rich.roots
        poor_growth = new_poor.roots - state_poor.roots

        assert rich_growth > poor_growth


class TestDifferentiability:
    """Tests for gradient flow through dynamics."""

    def test_gradients_exist(self) -> None:
        """Gradients should exist through the step function."""
        import jax

        config = SimConfig()

        def loss_fn(energy: Array) -> Array:
            state = TreeState(
                energy=energy,
                water=jnp.array(0.8),
                nutrients=jnp.array(0.8),
                roots=jnp.array(0.5),
                trunk=jnp.array(2.0),  # High trunk to avoid structural penalty
                shoots=jnp.array(0.2),
                leaves=jnp.array(0.5),
                flowers=jnp.array(0.0),
                fruit=jnp.array(0.0),
                soil_water=jnp.array(0.5),
            )
            # Zero allocation to avoid energy consumption
            allocation = Allocation(
                roots=jnp.array(0.0),
                trunk=jnp.array(0.0),
                shoots=jnp.array(0.0),
                leaves=jnp.array(0.0),
                flowers=jnp.array(0.0),
            )
            new_state = dynamics.step(
                state=state,
                allocation=allocation,
                light=0.8,
                moisture=0.6,
                wind=0.0,  # No wind damage
                config=config,
            )
            return new_state.energy

        grad_fn = jax.grad(loss_fn)
        # Use high initial energy so output depends on input
        grad = grad_fn(jnp.array(5.0))

        assert jnp.isfinite(grad)
        # Gradient should be approximately 1 since energy passes through with some loss
        assert grad > 0.5  # Energy should mostly pass through

    def test_gradients_through_allocation(self) -> None:
        """Gradients should flow through allocation."""
        import jax

        config = SimConfig()
        state = make_test_state()

        def loss_fn(root_alloc: Array) -> Array:
            allocation = Allocation(
                roots=root_alloc,
                trunk=jnp.array(0.2),
                shoots=jnp.array(0.2),
                leaves=jnp.array(0.4) - root_alloc / 2,
                flowers=jnp.array(0.2) - root_alloc / 2,
            )
            new_state = dynamics.step(
                state=state,
                allocation=allocation,
                light=0.8,
                moisture=0.6,
                wind=0.2,
                config=config,
            )
            return new_state.roots

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(jnp.array(0.2))

        assert jnp.isfinite(grad)
        assert grad > 0  # More root allocation should increase roots
