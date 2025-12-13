"""
Environmental stress signal generation.

This module generates time-varying environmental conditions using
sinusoidal signals. Each stressor (light, moisture, wind) follows:

    signal(t) = offset + amplitude * sin(frequency * t + phase)

Signals are clipped to [0, 1] to represent normalized conditions.
"""

import jax.numpy as jnp
from jax import Array

from sim.config import ClimateConfig, StressParams


def compute_signal(params: StressParams, t: float) -> Array:
    """
    Compute a single stress signal value at time t.

    signal(t) = clip(offset + amplitude * sin(frequency * t + phase), 0, 1)

    Args:
        params: Signal parameters (offset, amplitude, frequency, phase)
        t: Time (in days)

    Returns:
        Signal value in [0, 1]
    """
    raw = params.offset + params.amplitude * jnp.sin(
        params.frequency * t + params.phase
    )
    return jnp.clip(raw, 0.0, 1.0)


def compute_environment(config: ClimateConfig, t: float) -> tuple[Array, Array, Array]:
    """
    Compute all environmental stress values at time t.

    Args:
        config: Climate configuration with light, moisture, wind params
        t: Time (in days)

    Returns:
        Tuple of (light, moisture, wind), each in [0, 1]
    """
    light = compute_signal(config.light, t)
    moisture = compute_signal(config.moisture, t)
    wind = compute_signal(config.wind, t)
    return light, moisture, wind


def compute_environment_batch(
    config: ClimateConfig, num_days: int
) -> tuple[Array, Array, Array]:
    """
    Compute environmental stress values for multiple days.

    This is more efficient than calling compute_environment repeatedly
    as it uses vectorized operations.

    Args:
        config: Climate configuration
        num_days: Number of days to compute

    Returns:
        Tuple of (light, moisture, wind) arrays, each of shape (num_days,)
    """
    t = jnp.arange(num_days, dtype=jnp.float32)

    # Light
    light_raw = config.light.offset + config.light.amplitude * jnp.sin(
        config.light.frequency * t + config.light.phase
    )
    light = jnp.clip(light_raw, 0.0, 1.0)

    # Moisture
    moisture_raw = config.moisture.offset + config.moisture.amplitude * jnp.sin(
        config.moisture.frequency * t + config.moisture.phase
    )
    moisture = jnp.clip(moisture_raw, 0.0, 1.0)

    # Wind
    wind_raw = config.wind.offset + config.wind.amplitude * jnp.sin(
        config.wind.frequency * t + config.wind.phase
    )
    wind = jnp.clip(wind_raw, 0.0, 1.0)

    return light, moisture, wind


def random_climate(key: Array, base: ClimateConfig | None = None) -> ClimateConfig:
    """
    Generate a random climate configuration.

    Useful for training policies that generalize across climates.

    Args:
        key: JAX random key
        base: Optional base config to perturb (defaults to mild)

    Returns:
        Randomized ClimateConfig
    """
    import jax.random as jr

    if base is None:
        base = ClimateConfig.mild()

    # Split keys for each parameter
    keys = jr.split(key, 9)

    def perturb_params(
        params: StressParams, k1: Array, k2: Array, k3: Array
    ) -> StressParams:
        """Perturb stress parameters within reasonable bounds."""
        offset = jnp.clip(params.offset + 0.2 * (jr.uniform(k1) - 0.5), 0.1, 0.9)
        amplitude = jnp.clip(params.amplitude + 0.1 * (jr.uniform(k2) - 0.5), 0.05, 0.4)
        phase = params.phase + jnp.pi * (jr.uniform(k3) - 0.5)
        return StressParams(
            offset=float(offset),
            amplitude=float(amplitude),
            frequency=params.frequency,
            phase=float(phase),
        )

    return ClimateConfig(
        light=perturb_params(base.light, keys[0], keys[1], keys[2]),
        moisture=perturb_params(base.moisture, keys[3], keys[4], keys[5]),
        wind=perturb_params(base.wind, keys[6], keys[7], keys[8]),
    )
