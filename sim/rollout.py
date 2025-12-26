"""
Full season rollout simulation.

This module runs a complete season of tree growth, combining:
- Environmental stress signals
- Allocation policy
- Dynamics updates

The result is a trajectory containing the full history of
states, allocations, and environmental conditions.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from sim import dynamics, stress
from sim.config import Allocation, ClimateConfig, SimConfig, TreeState

# Type alias for policy functions
PolicyFn = Callable[[TreeState, int, int, float], Allocation]


@dataclass
class Trajectory:
    """
    Complete record of a season's simulation.

    Contains:
    - states: List of TreeState at each timestep (including initial)
    - allocations: List of Allocation made each day
    - stress histories: Light, moisture, wind at each day
    - seeds: Final seed production
    """

    states: list[TreeState]
    allocations: list[Allocation]
    light_history: list[float]
    moisture_history: list[float]
    wind_history: list[float]
    seeds: Array

    def get_state_arrays(self) -> dict[str, Array]:
        """Convert state history to arrays for plotting."""
        return {
            "energy": jnp.array([float(s.energy) for s in self.states]),
            "water": jnp.array([float(s.water) for s in self.states]),
            "nutrients": jnp.array([float(s.nutrients) for s in self.states]),
            "roots": jnp.array([float(s.roots) for s in self.states]),
            "trunk": jnp.array([float(s.trunk) for s in self.states]),
            "shoots": jnp.array([float(s.shoots) for s in self.states]),
            "leaves": jnp.array([float(s.leaves) for s in self.states]),
            "flowers": jnp.array([float(s.flowers) for s in self.states]),
            "fruit": jnp.array([float(s.fruit) for s in self.states]),
        }

    def get_allocation_arrays(self) -> dict[str, Array]:
        """Convert allocation history to arrays for plotting."""
        return {
            "roots": jnp.array([float(a.roots) for a in self.allocations]),
            "trunk": jnp.array([float(a.trunk) for a in self.allocations]),
            "shoots": jnp.array([float(a.shoots) for a in self.allocations]),
            "leaves": jnp.array([float(a.leaves) for a in self.allocations]),
            "flowers": jnp.array([float(a.flowers) for a in self.allocations]),
        }

    def get_scalar_summary(self) -> dict[str, float]:
        """
        Compute scalar diagnostic summary of the simulation.

        Returns a dictionary with key metrics for quick evaluation:
        - Seeds: Final seed count (primary fitness metric)
        - FinalBiomass: Total biomass at end of season
        - FinalEnergy: Energy reserves at end
        - PeakEnergy: Maximum energy achieved during season
        - MinEnergy: Minimum energy during season
        - DaysAtZero: Number of days with energy < 0.01 (stress indicator)
        - FinalTrunk: Trunk biomass (structural investment)
        - FinalFlowers: Flower biomass (reproductive investment)
        - MeanLight/Moisture/Wind: Average environmental conditions
        """
        state_arrays = self.get_state_arrays()
        energy = state_arrays["energy"]

        # Final state metrics
        final_state = self.states[-1]
        final_biomass = float(final_state.total_biomass())

        # Energy dynamics
        peak_energy = float(jnp.max(energy))
        min_energy = float(jnp.min(energy))
        days_at_zero = int(jnp.sum(energy < 0.01))

        # Environmental averages
        mean_light = sum(self.light_history) / len(self.light_history)
        mean_moisture = sum(self.moisture_history) / len(self.moisture_history)
        mean_wind = sum(self.wind_history) / len(self.wind_history)

        return {
            "Seeds": float(self.seeds),
            "FinalBiomass": final_biomass,
            "FinalEnergy": float(final_state.energy),
            "PeakEnergy": peak_energy,
            "MinEnergy": min_energy,
            "DaysAtZero": days_at_zero,
            "FinalTrunk": float(final_state.trunk),
            "FinalFlowers": float(final_state.flowers),
            "FinalFruit": float(final_state.fruit),
            "FinalRoots": float(final_state.roots),
            "FinalLeaves": float(final_state.leaves),
            "MeanLight": mean_light,
            "MeanMoisture": mean_moisture,
            "MeanWind": mean_wind,
        }

    def print_summary(self) -> None:
        """Print a formatted summary table to stdout."""
        summary = self.get_scalar_summary()
        print("\n" + "=" * 40)
        print("SIMULATION SUMMARY")
        print("=" * 40)
        for key, value in summary.items():
            if isinstance(value, int) or key == "DaysAtZero":
                print(f"{key:20s}: {int(value):>10d}")
            else:
                print(f"{key:20s}: {value:>10.3f}")
        print("=" * 40)


def run_season(
    config: SimConfig,
    climate: ClimateConfig,
    policy: PolicyFn,
    initial_state: TreeState | None = None,
) -> Trajectory:
    """
    Run a complete season simulation.

    Args:
        config: Simulation configuration
        climate: Climate configuration for stress signals
        policy: Policy function that returns allocations
        initial_state: Optional starting state (defaults to seed)

    Returns:
        Trajectory containing full simulation history
    """
    # Initialize
    if initial_state is None:
        state = TreeState.initial(energy=config.seed_energy)
    else:
        state = initial_state

    # Storage for history
    states: list[TreeState] = [state]
    allocations: list[Allocation] = []
    light_history: list[float] = []
    moisture_history: list[float] = []
    wind_history: list[float] = []

    # Run simulation
    for day in range(config.num_days):
        # Get environmental conditions
        light, moisture, wind = stress.compute_environment(climate, t=float(day))
        light_history.append(float(light))
        moisture_history.append(float(moisture))
        wind_history.append(float(wind))

        # Get allocation from policy
        allocation = policy(state, day, config.num_days, float(wind))
        allocations.append(allocation)

        # Update state
        state = dynamics.step(
            state=state,
            allocation=allocation,
            light=float(light),
            moisture=float(moisture),
            wind=float(wind),
            config=config,
            day=day,
        )
        states.append(state)

    # Compute final seeds
    seeds = dynamics.compute_seeds(state, config)

    return Trajectory(
        states=states,
        allocations=allocations,
        light_history=light_history,
        moisture_history=moisture_history,
        wind_history=wind_history,
        seeds=seeds,
    )


def evaluate_policy(
    policy: PolicyFn,
    config: SimConfig,
    climate: ClimateConfig,
    num_runs: int = 1,
) -> dict[str, float]:
    """
    Evaluate a policy over multiple runs.

    Args:
        policy: Policy function to evaluate
        config: Simulation configuration
        climate: Climate configuration
        num_runs: Number of runs for averaging

    Returns:
        Dictionary with evaluation metrics
    """
    total_seeds = 0.0
    total_biomass = 0.0
    survived = 0

    for _ in range(num_runs):
        trajectory = run_season(config, climate, policy)
        total_seeds += float(trajectory.seeds)
        final_biomass = float(trajectory.states[-1].total_biomass())
        total_biomass += final_biomass
        if final_biomass > 0:
            survived += 1

    return {
        "mean_seeds": total_seeds / num_runs,
        "mean_biomass": total_biomass / num_runs,
        "survival_rate": survived / num_runs,
    }


def compare_policies(
    policies_dict: dict[str, PolicyFn],
    config: SimConfig,
    climate: ClimateConfig,
) -> dict[str, dict[str, float]]:
    """
    Compare multiple policies on the same climate.

    Args:
        policies_dict: Dictionary mapping policy names to functions
        config: Simulation configuration
        climate: Climate configuration

    Returns:
        Dictionary mapping policy names to their metrics
    """
    results = {}
    for name, policy in policies_dict.items():
        results[name] = evaluate_policy(policy, config, climate)
    return results
