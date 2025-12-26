"""
Tipping point and resilience analysis for tree growth simulation.

This module provides tools to identify critical environmental thresholds
where tree survival or fitness drops sharply, using JAX autodiff for
efficient gradient computation.

Key analyses:
- Parameter sweeps: Grid search over stress parameters
- Sensitivity analysis: ∂fitness/∂param via JAX grad
- Tipping point detection: Where gradients spike or fitness collapses
- Resilience boundaries: 2D heatmaps of survival regions

Climate relevance:
Climate tipping points are a major concern for ecosystem management.
Understanding where trees fail under stress helps inform:
- Species selection for climate adaptation
- Risk assessment for forest management
- Identification of vulnerable ecosystems

All functions are JAX-compatible for end-to-end differentiability.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from sim.config import ClimateConfig, StressParams

if TYPE_CHECKING:
    from sim.config import SimConfig

# Type alias for fitness function
FitnessFn = Callable[[ClimateConfig], float]


def make_climate_with_offsets(
    base_climate: ClimateConfig,
    moisture_offset: float,
    wind_offset: float,
) -> ClimateConfig:
    """
    Create a climate config with modified moisture and wind offsets.

    This is the primary way to explore the stress parameter space.
    Keeps other parameters (amplitude, frequency, phase) fixed.

    Args:
        base_climate: Base climate configuration to modify
        moisture_offset: New moisture offset (0-1, higher = wetter)
        wind_offset: New wind offset (0-1, higher = windier)

    Returns:
        Modified ClimateConfig with new offsets
    """
    return ClimateConfig(
        light=base_climate.light,
        moisture=StressParams(
            offset=moisture_offset,
            amplitude=base_climate.moisture.amplitude,
            frequency=base_climate.moisture.frequency,
            phase=base_climate.moisture.phase,
        ),
        wind=StressParams(
            offset=wind_offset,
            amplitude=base_climate.wind.amplitude,
            frequency=base_climate.wind.frequency,
            phase=base_climate.wind.phase,
        ),
    )


def parameter_sweep_2d(
    fitness_fn: FitnessFn,
    base_climate: ClimateConfig,
    moisture_range: tuple[float, float] = (0.2, 0.9),
    wind_range: tuple[float, float] = (0.1, 0.8),
    resolution: int = 20,
) -> dict[str, np.ndarray]:
    """
    Sweep over moisture and wind offsets to map fitness landscape.

    Creates a 2D grid of (moisture, wind) values and evaluates fitness
    at each point. This reveals the "viable region" and identifies
    where fitness drops sharply.

    Args:
        fitness_fn: Function that takes ClimateConfig and returns fitness (scalar)
        base_climate: Base climate to modify (keeps light, amplitudes, etc.)
        moisture_range: (min, max) for moisture offset sweep
        wind_range: (min, max) for wind offset sweep
        resolution: Number of points per dimension (total = resolution²)

    Returns:
        Dictionary with:
        - moisture_grid: 2D array of moisture values (shape: resolution × resolution)
        - wind_grid: 2D array of wind values
        - fitness_grid: 2D array of fitness values
        - moisture_vals: 1D array of moisture values used
        - wind_vals: 1D array of wind values used
    """
    moisture_vals = np.linspace(moisture_range[0], moisture_range[1], resolution)
    wind_vals = np.linspace(wind_range[0], wind_range[1], resolution)

    fitness_grid = np.zeros((resolution, resolution))

    for i, moisture in enumerate(moisture_vals):
        for j, wind in enumerate(wind_vals):
            climate = make_climate_with_offsets(base_climate, moisture, wind)
            fitness_grid[i, j] = float(fitness_fn(climate))

    moisture_grid, wind_grid = np.meshgrid(moisture_vals, wind_vals, indexing="ij")

    return {
        "moisture_grid": moisture_grid,
        "wind_grid": wind_grid,
        "fitness_grid": fitness_grid,
        "moisture_vals": moisture_vals,
        "wind_vals": wind_vals,
    }


def compute_sensitivity_finite_diff(
    fitness_fn: FitnessFn,
    base_climate: ClimateConfig,
    param_name: str = "moisture_offset",
    eps: float = 1e-4,
) -> float:
    """
    Compute sensitivity of fitness to a stress parameter using finite differences.

    This is a more robust method than autodiff for functions that involve
    non-JAX operations (like creating new ClimateConfig objects).

    Args:
        fitness_fn: Fitness function (ClimateConfig → scalar)
        base_climate: Climate at which to compute sensitivity
        param_name: Which parameter to differentiate w.r.t.
            Options: "moisture_offset", "wind_offset"
        eps: Finite difference step size

    Returns:
        Gradient value (scalar) - positive means fitness increases with param
    """
    if param_name == "moisture_offset":
        base_val = base_climate.moisture.offset

        # Central difference
        climate_plus = make_climate_with_offsets(
            base_climate, base_val + eps, base_climate.wind.offset
        )
        climate_minus = make_climate_with_offsets(
            base_climate, base_val - eps, base_climate.wind.offset
        )

        fitness_plus = fitness_fn(climate_plus)
        fitness_minus = fitness_fn(climate_minus)

        return (fitness_plus - fitness_minus) / (2 * eps)

    elif param_name == "wind_offset":
        base_val = base_climate.wind.offset

        climate_plus = make_climate_with_offsets(
            base_climate, base_climate.moisture.offset, base_val + eps
        )
        climate_minus = make_climate_with_offsets(
            base_climate, base_climate.moisture.offset, base_val - eps
        )

        fitness_plus = fitness_fn(climate_plus)
        fitness_minus = fitness_fn(climate_minus)

        return (fitness_plus - fitness_minus) / (2 * eps)

    else:
        raise ValueError(
            f"Unknown parameter: {param_name}. Use 'moisture_offset' or 'wind_offset'."
        )


def compute_sensitivity(
    fitness_fn: FitnessFn,
    base_climate: ClimateConfig,
    param_name: str = "moisture_offset",
) -> float:
    """
    Compute sensitivity of fitness to a stress parameter.

    Uses finite differences for robustness with non-JAX fitness functions.

    Args:
        fitness_fn: Fitness function (ClimateConfig → scalar)
        base_climate: Climate at which to compute sensitivity
        param_name: Which parameter to differentiate w.r.t.
            Options: "moisture_offset", "wind_offset"

    Returns:
        Gradient value (scalar) - positive means fitness increases with param
    """
    return compute_sensitivity_finite_diff(fitness_fn, base_climate, param_name)


def sensitivity_sweep(
    fitness_fn: FitnessFn,
    base_climate: ClimateConfig,
    param_name: str,
    param_range: tuple[float, float],
    resolution: int = 50,
) -> dict[str, np.ndarray]:
    """
    Compute fitness and sensitivity across a range of parameter values.

    This produces the data for tipping point visualization:
    - Fitness curve shows where performance drops
    - Gradient curve shows where sensitivity spikes (potential tipping point)

    Args:
        fitness_fn: Fitness function
        base_climate: Base climate to modify
        param_name: Parameter to sweep ("moisture_offset" or "wind_offset")
        param_range: (min, max) range for parameter
        resolution: Number of points to evaluate

    Returns:
        Dictionary with:
        - param_values: 1D array of parameter values
        - fitness_values: 1D array of fitness at each point
        - gradient_values: 1D array of ∂fitness/∂param at each point
    """
    param_values = np.linspace(param_range[0], param_range[1], resolution)
    fitness_values = np.zeros(resolution)
    gradient_values = np.zeros(resolution)

    for i, param in enumerate(param_values):
        # Create climate with this parameter value
        if param_name == "moisture_offset":
            climate = make_climate_with_offsets(
                base_climate, param, base_climate.wind.offset
            )
        elif param_name == "wind_offset":
            climate = make_climate_with_offsets(
                base_climate, base_climate.moisture.offset, param
            )
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        # Evaluate fitness
        fitness_values[i] = float(fitness_fn(climate))

        # Compute gradient at this point
        try:
            gradient_values[i] = float(
                compute_sensitivity(fitness_fn, climate, param_name)
            )
        except Exception:
            # Gradient computation can fail at extreme values
            gradient_values[i] = 0.0

    return {
        "param_values": param_values,
        "fitness_values": fitness_values,
        "gradient_values": gradient_values,
    }


def find_tipping_points(
    sweep_result: dict[str, np.ndarray],
    gradient_threshold: float = 2.0,
    fitness_drop_threshold: float = 0.3,
) -> list[dict]:
    """
    Identify tipping points from a sensitivity sweep.

    Tipping points are identified where:
    1. Gradient magnitude spikes (rapid fitness change)
    2. Fitness drops significantly over a small parameter change

    Args:
        sweep_result: Output from sensitivity_sweep
        gradient_threshold: Z-score threshold for "high" gradient magnitude
        fitness_drop_threshold: Fraction drop to count as "significant"

    Returns:
        List of tipping point dictionaries, each with:
        - param_value: Parameter value at tipping point
        - fitness: Fitness at that point
        - gradient: Gradient at that point
        - gradient_z_score: How many std devs above mean gradient
        - fitness_drop_pct: Percent fitness drop around this point
    """
    param_values = sweep_result["param_values"]
    fitness_values = sweep_result["fitness_values"]
    gradient_values = sweep_result["gradient_values"]

    tipping_points = []

    # Compute gradient statistics
    grad_magnitude = np.abs(gradient_values)
    grad_mean = np.mean(grad_magnitude)
    grad_std = np.std(grad_magnitude)

    for i in range(1, len(param_values) - 1):
        # Check for gradient spike
        z_score = (grad_magnitude[i] - grad_mean) / (grad_std + 1e-6)
        is_gradient_spike = z_score > gradient_threshold

        # Check for fitness drop (looking at neighbors)
        if fitness_values[i - 1] > 0:
            fitness_drop = fitness_values[i - 1] - fitness_values[i + 1]
            fitness_drop_pct = fitness_drop / fitness_values[i - 1]
        else:
            fitness_drop_pct = 0.0

        is_fitness_drop = fitness_drop_pct > fitness_drop_threshold

        if is_gradient_spike or is_fitness_drop:
            tipping_points.append(
                {
                    "param_value": float(param_values[i]),
                    "fitness": float(fitness_values[i]),
                    "gradient": float(gradient_values[i]),
                    "gradient_z_score": float(z_score),
                    "fitness_drop_pct": float(fitness_drop_pct * 100),
                }
            )

    return tipping_points


def compute_resilience_boundary(
    fitness_fn: FitnessFn,
    base_climate: ClimateConfig,
    survival_threshold: float = 0.1,
    moisture_range: tuple[float, float] = (0.15, 0.95),
    wind_range: tuple[float, float] = (0.05, 0.85),
    resolution: int = 30,
) -> dict[str, np.ndarray]:
    """
    Find the boundary in parameter space where fitness drops below threshold.

    This identifies the "resilience boundary" - the edge of the viable
    region where trees can survive and reproduce.

    Args:
        fitness_fn: Fitness function
        base_climate: Base climate configuration
        survival_threshold: Minimum fitness to count as "surviving"
        moisture_range: Range for moisture sweep
        wind_range: Range for wind sweep
        resolution: Grid resolution

    Returns:
        Dictionary with:
        - All outputs from parameter_sweep_2d
        - survival_mask: Boolean array where fitness >= threshold
        - boundary_points: (moisture, wind) coordinates on boundary
        - survival_threshold: The threshold used
    """
    sweep = parameter_sweep_2d(
        fitness_fn,
        base_climate,
        moisture_range=moisture_range,
        wind_range=wind_range,
        resolution=resolution,
    )

    survival_mask = sweep["fitness_grid"] >= survival_threshold

    # Find boundary points (where survival transitions)
    boundary_moisture = []
    boundary_wind = []

    for i in range(1, resolution - 1):
        for j in range(1, resolution - 1):
            if survival_mask[i, j]:
                # Check if any neighbor is non-surviving
                neighbors = [
                    survival_mask[i - 1, j],
                    survival_mask[i + 1, j],
                    survival_mask[i, j - 1],
                    survival_mask[i, j + 1],
                ]
                if not all(neighbors):
                    boundary_moisture.append(sweep["moisture_vals"][i])
                    boundary_wind.append(sweep["wind_vals"][j])

    return {
        **sweep,
        "survival_mask": survival_mask,
        "boundary_points": (np.array(boundary_moisture), np.array(boundary_wind)),
        "survival_threshold": survival_threshold,
    }


def resilience_report(
    fitness_fn: FitnessFn,
    base_climate: ClimateConfig,
    config: "SimConfig",  # noqa: ARG001
) -> dict:
    """
    Generate a comprehensive resilience analysis report.

    This is the main entry point for resilience analysis - it runs
    all analyses and returns a complete report.

    Args:
        fitness_fn: Fitness function (ClimateConfig → scalar)
        base_climate: Climate to analyze around
        config: Simulation configuration (for reference)

    Returns:
        Dictionary with:
        - landscape: 2D fitness sweep results
        - moisture_sensitivity: Sensitivity sweep for moisture
        - wind_sensitivity: Sensitivity sweep for wind
        - moisture_tipping_points: List of moisture tipping points
        - wind_tipping_points: List of wind tipping points
        - boundary: Resilience boundary analysis
        - summary: Key statistics
    """
    # 2D fitness landscape (lower resolution for speed)
    landscape = parameter_sweep_2d(fitness_fn, base_climate, resolution=20)

    # Sensitivity sweeps for each parameter
    moisture_sensitivity = sensitivity_sweep(
        fitness_fn, base_climate, "moisture_offset", (0.2, 0.9), resolution=30
    )
    wind_sensitivity = sensitivity_sweep(
        fitness_fn, base_climate, "wind_offset", (0.1, 0.8), resolution=30
    )

    # Find tipping points
    moisture_tipping = find_tipping_points(moisture_sensitivity)
    wind_tipping = find_tipping_points(wind_sensitivity)

    # Resilience boundary
    boundary = compute_resilience_boundary(fitness_fn, base_climate, resolution=20)

    # Summary statistics
    fitness_grid = landscape["fitness_grid"]
    survival_mask = boundary["survival_mask"]

    return {
        "landscape": landscape,
        "moisture_sensitivity": moisture_sensitivity,
        "wind_sensitivity": wind_sensitivity,
        "moisture_tipping_points": moisture_tipping,
        "wind_tipping_points": wind_tipping,
        "boundary": boundary,
        "summary": {
            "max_fitness": float(np.max(fitness_grid)),
            "min_fitness": float(np.min(fitness_grid)),
            "mean_fitness": float(np.mean(fitness_grid)),
            "survival_fraction": float(np.mean(survival_mask)),
            "num_moisture_tipping_points": len(moisture_tipping),
            "num_wind_tipping_points": len(wind_tipping),
            "viable_area_fraction": float(np.mean(survival_mask)),
        },
    }


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_fitness_landscape(
    sweep_result: dict[str, np.ndarray],
    title: str = "Fitness Landscape",
    ax=None,
):
    """
    Plot 2D heatmap of fitness across parameter space.

    Args:
        sweep_result: Output from parameter_sweep_2d
        title: Plot title
        ax: Matplotlib axis (optional, creates new figure if None)

    Returns:
        Matplotlib axis with the plot
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        sweep_result["fitness_grid"].T,  # Transpose for correct orientation
        extent=[
            sweep_result["moisture_vals"].min(),
            sweep_result["moisture_vals"].max(),
            sweep_result["wind_vals"].min(),
            sweep_result["wind_vals"].max(),
        ],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )

    ax.set_xlabel("Moisture Offset (higher = wetter)")
    ax.set_ylabel("Wind Offset (higher = windier)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Fitness (seeds)")

    return ax


def plot_sensitivity_curve(
    sweep_result: dict[str, np.ndarray],
    param_name: str,
    axes=None,
):
    """
    Plot fitness and gradient vs parameter value.

    Creates a two-panel figure showing:
    - Left: Fitness curve
    - Right: Gradient (sensitivity) curve

    Args:
        sweep_result: Output from sensitivity_sweep
        param_name: Name for x-axis label
        axes: Tuple of two matplotlib axes (optional)

    Returns:
        Tuple of matplotlib axes with the plots
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax1, ax2 = axes

    # Fitness curve
    ax1.plot(
        sweep_result["param_values"],
        sweep_result["fitness_values"],
        linewidth=2,
        color="blue",
    )
    ax1.set_xlabel(param_name)
    ax1.set_ylabel("Fitness")
    ax1.set_title(f"Fitness vs {param_name}")
    ax1.grid(True, alpha=0.3)

    # Gradient curve
    ax2.plot(
        sweep_result["param_values"],
        sweep_result["gradient_values"],
        linewidth=2,
        color="red",
    )
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel(param_name)
    ax2.set_ylabel("Gradient (∂F/∂param)")
    ax2.set_title(f"Sensitivity vs {param_name}")
    ax2.grid(True, alpha=0.3)

    return axes


def plot_resilience_boundary(
    boundary_result: dict[str, np.ndarray],
    title: str = "Resilience Boundary",
    ax=None,
):
    """
    Plot survival region and boundary in parameter space.

    Shows fitness as background color, with survival boundary overlaid.
    The boundary marks the transition from viable to non-viable conditions.

    Args:
        boundary_result: Output from compute_resilience_boundary
        title: Plot title
        ax: Matplotlib axis (optional)

    Returns:
        Matplotlib axis with the plot
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    # Plot fitness as background
    im = ax.imshow(
        boundary_result["fitness_grid"].T,
        extent=[
            boundary_result["moisture_vals"].min(),
            boundary_result["moisture_vals"].max(),
            boundary_result["wind_vals"].min(),
            boundary_result["wind_vals"].max(),
        ],
        origin="lower",
        aspect="auto",
        cmap="RdYlGn",
        alpha=0.7,
    )

    # Overlay survival boundary
    boundary_moisture, boundary_wind = boundary_result["boundary_points"]
    if len(boundary_moisture) > 0:
        ax.scatter(
            boundary_moisture,
            boundary_wind,
            c="black",
            s=20,
            marker="o",
            label=f"Survival boundary (F={boundary_result['survival_threshold']:.2f})",
        )
        ax.legend()

    ax.set_xlabel("Moisture Offset (higher = wetter)")
    ax.set_ylabel("Wind Offset (higher = windier)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Fitness")

    return ax
