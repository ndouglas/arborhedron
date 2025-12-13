"""
Gradient health checks for the simulation.

This module provides utilities to verify gradient flow through
the simulation components. Key concern: multiplicative limiting
factors can kill gradients in low-resource regimes.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from sim import surrogates
from sim.config import SimConfig


def sample_episode_distribution(
    key: Array,
    num_samples: int = 1000,
) -> dict[str, Array]:
    """
    Sample from the intended episode distribution.

    This covers the range of states a learning agent will encounter,
    including the critical low-resource regime at seed stage.

    Returns:
        Dictionary with arrays for each input dimension
    """
    keys = jr.split(key, 6)

    # Sample inputs across realistic ranges
    # Include many low-resource samples (seed stage is critical)
    light = jr.uniform(keys[0], (num_samples,), minval=0.0, maxval=1.0)
    water = jr.uniform(keys[1], (num_samples,), minval=0.0, maxval=1.0)
    nutrients = jr.uniform(keys[2], (num_samples,), minval=0.0, maxval=1.0)
    wind = jr.uniform(keys[3], (num_samples,), minval=0.0, maxval=1.0)
    leaves = jr.uniform(keys[4], (num_samples,), minval=0.01, maxval=2.0)
    energy = jr.uniform(keys[5], (num_samples,), minval=0.1, maxval=3.0)

    return {
        "light": light,
        "water": water,
        "nutrients": nutrients,
        "wind": wind,
        "leaves": leaves,
        "energy": energy,
    }


def compute_photosynthesis_gradients(
    samples: dict[str, Array],
    config: SimConfig,
) -> dict[str, Array]:
    """
    Compute gradient magnitudes for photosynthesis wrt each input.

    Returns:
        Dictionary mapping input names to gradient magnitude arrays
    """

    def photo_fn(light: float, water: float, nutrients: float, leaves: float) -> Array:
        return surrogates.photosynthesis(
            leaves=leaves,
            light=light,
            water=water,
            nutrients=nutrients,
            p_max=config.p_max,
            k_light=config.k_light,
            k_water=config.k_water,
            k_nutrient=config.k_nutrient,
            k_leaf=config.k_leaf,
        )

    # Compute gradients for each input
    grad_light = jax.vmap(jax.grad(photo_fn, argnums=0))(
        samples["light"], samples["water"], samples["nutrients"], samples["leaves"]
    )
    grad_water = jax.vmap(jax.grad(photo_fn, argnums=1))(
        samples["light"], samples["water"], samples["nutrients"], samples["leaves"]
    )
    grad_nutrients = jax.vmap(jax.grad(photo_fn, argnums=2))(
        samples["light"], samples["water"], samples["nutrients"], samples["leaves"]
    )
    grad_leaves = jax.vmap(jax.grad(photo_fn, argnums=3))(
        samples["light"], samples["water"], samples["nutrients"], samples["leaves"]
    )

    return {
        "light": jnp.abs(grad_light),
        "water": jnp.abs(grad_water),
        "nutrients": jnp.abs(grad_nutrients),
        "leaves": jnp.abs(grad_leaves),
    }


def gradient_health_report(
    key: Array,
    config: SimConfig | None = None,
    num_samples: int = 1000,
) -> dict[str, dict[str, float]]:
    """
    Generate a gradient health report.

    Reports statistics on gradient magnitudes across the episode distribution.
    Key metrics:
    - Mean: typical gradient magnitude
    - Min: worst-case (potential dead zone)
    - % near zero: fraction of samples with |grad| < 0.001

    Args:
        key: JAX random key
        config: Simulation config (uses defaults if None)
        num_samples: Number of samples to draw

    Returns:
        Nested dict: {input_name: {metric_name: value}}
    """
    if config is None:
        config = SimConfig()

    samples = sample_episode_distribution(key, num_samples)
    gradients = compute_photosynthesis_gradients(samples, config)

    report = {}
    for name, grads in gradients.items():
        report[name] = {
            "mean": float(jnp.mean(grads)),
            "std": float(jnp.std(grads)),
            "min": float(jnp.min(grads)),
            "max": float(jnp.max(grads)),
            "pct_near_zero": float(jnp.mean(grads < 0.001) * 100),
            "pct_healthy": float(jnp.mean(grads > 0.01) * 100),
        }

    return report


def print_gradient_report(report: dict[str, dict[str, float]]) -> None:
    """Pretty-print a gradient health report."""
    print("\n" + "=" * 60)
    print("GRADIENT HEALTH REPORT - Photosynthesis")
    print("=" * 60)
    print(f"{'Input':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'%Dead':>8} {'%OK':>8}")
    print("-" * 60)

    for name, metrics in report.items():
        print(
            f"{name:<12} "
            f"{metrics['mean']:>8.4f} "
            f"{metrics['std']:>8.4f} "
            f"{metrics['min']:>8.4f} "
            f"{metrics['pct_near_zero']:>7.1f}% "
            f"{metrics['pct_healthy']:>7.1f}%"
        )

    print("=" * 60)
    print("Dead = |grad| < 0.001, OK = |grad| > 0.01")
    print("Target: <5% dead, >80% OK")
    print()
