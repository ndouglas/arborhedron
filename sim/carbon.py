"""
Carbon sequestration metrics for tree growth simulation.

This module computes carbon content and sequestration metrics from tree state,
enabling climate-focused optimization objectives. Carbon metrics provide a
scientifically grounded way to evaluate tree growth strategies for their
climate impact.

Key concepts:
- Carbon content: Total carbon mass in each compartment (biomass × C fraction)
- Permanent carbon: Carbon in long-lived tissues (trunk, roots)
- Seasonal carbon: Carbon in ephemeral tissues (leaves, flowers)
- Carbon score: Permanence-weighted sum for optimization (rewards durable storage)

Climate relevance:
Trees sequester atmospheric CO2 via photosynthesis, storing carbon in biomass.
However, not all storage is equal: trunk wood persists for decades/centuries,
while leaves decompose within a year, releasing their carbon back to atmosphere.
The permanence-weighted carbon score captures this distinction.

All functions are JAX-compatible for end-to-end differentiability.
"""

from typing import TYPE_CHECKING

from jax import Array
from jax.nn import sigmoid

if TYPE_CHECKING:
    from sim.config import SimConfig, TreeState


def compute_carbon_content(state: "TreeState", config: "SimConfig") -> dict[str, Array]:
    """
    Compute carbon content in each compartment.

    Carbon content = biomass × carbon fraction for each tissue type.

    Args:
        state: Current tree state with biomass values
        config: Simulation configuration with carbon fractions

    Returns:
        Dictionary with carbon content per compartment:
        - trunk_carbon, roots_carbon, shoots_carbon, leaves_carbon, flowers_carbon
        - total_carbon: Sum of all compartments
        - permanent_carbon: trunk + roots (long-lived storage)
        - seasonal_carbon: shoots + leaves + flowers (ephemeral)
    """
    trunk_carbon = state.trunk * config.carbon_fraction_trunk
    roots_carbon = state.roots * config.carbon_fraction_roots
    shoots_carbon = state.shoots * config.carbon_fraction_shoots
    leaves_carbon = state.leaves * config.carbon_fraction_leaves
    flowers_carbon = state.flowers * config.carbon_fraction_flowers

    total_carbon = (
        trunk_carbon + roots_carbon + shoots_carbon + leaves_carbon + flowers_carbon
    )

    # Permanent vs seasonal split (important for climate impact)
    permanent_carbon = trunk_carbon + roots_carbon
    seasonal_carbon = shoots_carbon + leaves_carbon + flowers_carbon

    return {
        "trunk_carbon": trunk_carbon,
        "roots_carbon": roots_carbon,
        "shoots_carbon": shoots_carbon,
        "leaves_carbon": leaves_carbon,
        "flowers_carbon": flowers_carbon,
        "total_carbon": total_carbon,
        "permanent_carbon": permanent_carbon,
        "seasonal_carbon": seasonal_carbon,
    }


def compute_carbon_score(state: "TreeState", config: "SimConfig") -> Array:
    """
    Compute permanence-weighted carbon score.

    This is the primary metric for carbon-focused optimization.
    Higher permanence tissues contribute more to the score, reflecting
    their greater climate value as long-term carbon sinks.

    score = Σ (carbon_i × permanence_i)
          = Σ (biomass_i × carbon_fraction_i × permanence_i)

    The score rewards:
    - Trunk investment (permanence=1.0, highest value)
    - Root development (permanence=0.7)
    - Less emphasis on leaves/flowers (permanence=0.1/0.05)

    This creates different optimal strategies than seed maximization:
    - Seed-focused policy: maximize flowers late in season
    - Carbon-focused policy: maximize trunk throughout season

    Args:
        state: Current tree state
        config: Simulation configuration with permanence weights

    Returns:
        Permanence-weighted carbon score (scalar, differentiable)
    """
    trunk_contrib = state.trunk * config.carbon_fraction_trunk * config.permanence_trunk
    roots_contrib = state.roots * config.carbon_fraction_roots * config.permanence_roots
    shoots_contrib = (
        state.shoots * config.carbon_fraction_shoots * config.permanence_shoots
    )
    leaves_contrib = (
        state.leaves * config.carbon_fraction_leaves * config.permanence_leaves
    )
    flowers_contrib = (
        state.flowers * config.carbon_fraction_flowers * config.permanence_flowers
    )

    return (
        trunk_contrib
        + roots_contrib
        + shoots_contrib
        + leaves_contrib
        + flowers_contrib
    )


def compute_carbon_efficiency(
    carbon_score: Array,
    energy_invested: Array,
    eps: float = 1e-6,
) -> Array:
    """
    Compute carbon sequestration efficiency.

    efficiency = carbon_score / energy_invested

    This measures how effectively the tree converts energy into
    permanent carbon storage. Higher efficiency means more carbon
    sequestered per unit of photosynthetic energy.

    Useful for comparing strategies: a policy that produces
    less total carbon but uses less energy might be more efficient.

    Args:
        carbon_score: Permanence-weighted carbon (from compute_carbon_score)
        energy_invested: Total energy invested in growth over season
        eps: Small constant for numerical stability

    Returns:
        Carbon efficiency ratio (dimensionless)
    """
    return carbon_score / (energy_invested + eps)


def carbon_objective(
    final_state: "TreeState",  # noqa: ARG001
    config: "SimConfig",
    carbon_integral: Array,
    final_energy: Array,
    energy_threshold: float = 0.3,
) -> Array:
    """
    Carbon-focused objective for optimization.

    Similar structure to seed production objective, but rewards
    carbon accumulation with an energy gate (tree must survive).

    objective = (carbon_integral / num_days) × energy_gate

    The energy gate ensures that only viable trees count:
    a dead tree releases its carbon back to atmosphere.

    Args:
        final_state: Tree state at end of season (unused but kept for API consistency)
        config: Simulation configuration
        carbon_integral: Integral of carbon_score over season (Σ carbon_score(t))
        final_energy: Final energy level (survival indicator)
        energy_threshold: Minimum energy for full credit (default 0.3)

    Returns:
        Carbon objective value (scalar, differentiable)
    """
    # Normalize by season length for comparability
    normalized_integral = carbon_integral / config.num_days

    # Energy gate: must survive to count carbon
    # sigmoid(10 * (E - threshold)) → 0 if dead, 1 if healthy
    energy_gate = sigmoid(10.0 * (final_energy - energy_threshold))

    return normalized_integral * energy_gate


def carbon_seed_tradeoff(
    carbon_score: Array,
    seeds: Array,
    carbon_weight: float = 0.5,
) -> Array:
    """
    Multi-objective combination of carbon and reproduction.

    objective = carbon_weight × carbon_score + (1 - carbon_weight) × seeds

    This enables exploring the Pareto frontier between carbon sequestration
    and reproductive fitness. In nature, trees face this tradeoff:
    - Investing in wood → more carbon storage, fewer seeds
    - Investing in flowers → more seeds, less wood

    Args:
        carbon_score: Permanence-weighted carbon from compute_carbon_score
        seeds: Seed production from seed objective
        carbon_weight: Weight for carbon (0-1), remainder goes to seeds
            - 0.0: Pure seed optimization (original objective)
            - 0.5: Balanced (default)
            - 1.0: Pure carbon optimization

    Returns:
        Combined objective value (scalar, differentiable)
    """
    return carbon_weight * carbon_score + (1.0 - carbon_weight) * seeds


def compute_carbon_summary(
    states: list["TreeState"],
    config: "SimConfig",
) -> dict[str, float]:
    """
    Compute comprehensive carbon metrics from a trajectory.

    This provides all relevant carbon statistics for a completed season,
    suitable for analysis and comparison across policies/climates.

    Args:
        states: List of TreeState objects from a season trajectory
        config: Simulation configuration

    Returns:
        Dictionary with summary metrics:
        - FinalTotalCarbon: Total carbon at end of season
        - FinalPermanentCarbon: Permanent (trunk+roots) carbon at end
        - FinalSeasonalCarbon: Seasonal (shoots+leaves+flowers) carbon at end
        - FinalCarbonScore: Permanence-weighted score at end
        - PeakTotalCarbon: Maximum total carbon reached during season
        - PeakPermanentCarbon: Maximum permanent carbon during season
        - MeanCarbonScore: Average carbon score over season
        - CarbonIntegral: Sum of carbon scores (for optimization)
        - TrunkCarbonFraction: Fraction of total carbon in trunk
    """
    if not states:
        return {
            "FinalTotalCarbon": 0.0,
            "FinalPermanentCarbon": 0.0,
            "FinalSeasonalCarbon": 0.0,
            "FinalCarbonScore": 0.0,
            "PeakTotalCarbon": 0.0,
            "PeakPermanentCarbon": 0.0,
            "MeanCarbonScore": 0.0,
            "CarbonIntegral": 0.0,
            "TrunkCarbonFraction": 0.0,
        }

    # Compute carbon metrics for each state
    carbon_scores = []
    total_carbons = []
    permanent_carbons = []

    for state in states:
        content = compute_carbon_content(state, config)
        score = compute_carbon_score(state, config)

        carbon_scores.append(float(score))
        total_carbons.append(float(content["total_carbon"]))
        permanent_carbons.append(float(content["permanent_carbon"]))

    # Final state metrics
    final_content = compute_carbon_content(states[-1], config)
    final_score = compute_carbon_score(states[-1], config)

    # Compute trunk fraction (how much of carbon is in most permanent form)
    final_total = float(final_content["total_carbon"])
    trunk_fraction = (
        float(final_content["trunk_carbon"]) / final_total if final_total > 0 else 0.0
    )

    return {
        "FinalTotalCarbon": final_total,
        "FinalPermanentCarbon": float(final_content["permanent_carbon"]),
        "FinalSeasonalCarbon": float(final_content["seasonal_carbon"]),
        "FinalCarbonScore": float(final_score),
        "PeakTotalCarbon": max(total_carbons),
        "PeakPermanentCarbon": max(permanent_carbons),
        "MeanCarbonScore": sum(carbon_scores) / len(carbon_scores),
        "CarbonIntegral": sum(carbon_scores),
        "TrunkCarbonFraction": trunk_fraction,
    }
