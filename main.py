"""
Arborhedron - Differentiable Tree Growth Pipeline

Demonstrates three Tesseracts working together:
1. growth_step - Single day tree growth simulation
2. neural_policy - Neural network allocation decisions
3. seed_production - Fitness computation from fruit integral

This pipeline composes differentiable Tesseracts to:
- Run a full growing season by looping growth_step
- Use neural_policy for allocation decisions
- Compute seeds via seed_production
- Optimize policy weights via gradient descent through the entire pipeline
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract


def compute_environment(day: int, num_days: int) -> tuple[Array, Array, Array]:
    """
    Compute environmental conditions for a given day.

    Uses simple sinusoidal patterns for demonstration.
    In practice, these could come from another Tesseract.
    """
    t = jnp.array(day, dtype=jnp.float32)

    # Light: high during middle of season
    light = 0.7 + 0.2 * jnp.sin(0.1 * t)
    light = jnp.clip(light, 0.0, 1.0)

    # Moisture: varies with phase offset
    moisture = 0.6 + 0.15 * jnp.sin(0.08 * t + 1.0)
    moisture = jnp.clip(moisture, 0.0, 1.0)

    # Wind: occasional gusts
    wind = 0.2 + 0.1 * jnp.sin(0.15 * t + 0.5)
    wind = jnp.clip(wind, 0.0, 1.0)

    return light, moisture, wind


def run_season_manual(
    growth_step: Tesseract,
    neural_policy: Tesseract,
    seed_production: Tesseract,
    policy_weights: dict,
    num_days: int = 100,
) -> dict:
    """
    Run a full growing season using Tesseract SDK calls.

    This demonstrates the composition pattern without JAX tracing,
    useful for debugging and understanding the pipeline.
    """
    # Initial tree state (seedling)
    state = {
        "energy": jnp.array(1.0),
        "water": jnp.array(0.3),
        "nutrients": jnp.array(0.3),
        "roots": jnp.array(0.1),
        "trunk": jnp.array(0.05),
        "shoots": jnp.array(0.05),
        "leaves": jnp.array(0.2),
        "flowers": jnp.array(1e-4),
        "fruit": jnp.array(0.0),
        "soil_water": jnp.array(0.5),
    }

    # Track fruit integral for seed computation
    fruit_integral = jnp.array(0.0)

    print(f"Starting season simulation ({num_days} days)...")

    for day in range(num_days):
        # Get environment
        light, moisture, wind = compute_environment(day, num_days)

        # Get allocation from neural policy
        policy_input = {
            "state": {
                "energy": state["energy"],
                "water": state["water"],
                "nutrients": state["nutrients"],
                "roots": state["roots"],
                "trunk": state["trunk"],
                "shoots": state["shoots"],
                "leaves": state["leaves"],
                "flowers": state["flowers"],
            },
            "light": light,
            "moisture": moisture,
            "wind": wind,
            "day": day,
            "num_days": num_days,
            "weights": policy_weights,
        }
        policy_output = neural_policy.apply(policy_input)
        allocation = policy_output["allocation"]

        # Run growth step
        growth_input = {
            "state": state,
            "allocation": allocation,
            "light": light,
            "moisture": moisture,
            "wind": wind,
            "day": day,
            "num_days": num_days,
        }
        growth_output = growth_step.apply(growth_input)
        state = growth_output["state"]

        # Accumulate fruit
        fruit_integral = fruit_integral + state["fruit"]

        # Progress indicator
        if day % 25 == 0:
            print(f"  Day {day}: energy={float(state['energy']):.2f}, "
                  f"leaves={float(state['leaves']):.2f}, "
                  f"flowers={float(state['flowers']):.2f}")

    # Compute seeds
    seed_input = {
        "fruit_integral": fruit_integral,
        "final_energy": state["energy"],
        "num_days": num_days,
        "seed_energy_threshold": 0.5,
        "seed_conversion": 10.0,
    }
    seed_output = seed_production.apply(seed_input)

    print(f"\nSeason complete!")
    print(f"  Final energy: {float(state['energy']):.3f}")
    print(f"  Final flowers: {float(state['flowers']):.3f}")
    print(f"  Fruit integral: {float(fruit_integral):.3f}")
    print(f"  Seeds produced: {float(seed_output['seeds']):.3f}")

    return {
        "final_state": state,
        "fruit_integral": fruit_integral,
        "seeds": seed_output["seeds"],
    }


def run_season_jax(
    growth_step: Tesseract,
    neural_policy: Tesseract,
    seed_production: Tesseract,
    policy_weights: dict,
    num_days: int = 100,
) -> Array:
    """
    Run a full growing season using Tesseract-JAX.

    This version is fully differentiable - we can compute gradients
    of seed production with respect to policy weights.
    """
    # Initial tree state
    state = {
        "energy": jnp.array(1.0),
        "water": jnp.array(0.3),
        "nutrients": jnp.array(0.3),
        "roots": jnp.array(0.1),
        "trunk": jnp.array(0.05),
        "shoots": jnp.array(0.05),
        "leaves": jnp.array(0.2),
        "flowers": jnp.array(1e-4),
        "fruit": jnp.array(0.0),
        "soil_water": jnp.array(0.5),
    }

    fruit_integral = jnp.array(0.0)

    # Run season using Tesseract-JAX composition
    for day in range(num_days):
        light, moisture, wind = compute_environment(day, num_days)

        # Neural policy
        policy_input = {
            "state": {
                "energy": state["energy"],
                "water": state["water"],
                "nutrients": state["nutrients"],
                "roots": state["roots"],
                "trunk": state["trunk"],
                "shoots": state["shoots"],
                "leaves": state["leaves"],
                "flowers": state["flowers"],
            },
            "light": light,
            "moisture": moisture,
            "wind": wind,
            "day": day,
            "num_days": num_days,
            "weights": policy_weights,
        }
        policy_output = apply_tesseract(neural_policy, policy_input)
        allocation = policy_output["allocation"]

        # Growth step
        growth_input = {
            "state": state,
            "allocation": allocation,
            "light": light,
            "moisture": moisture,
            "wind": wind,
            "day": day,
            "num_days": num_days,
        }
        growth_output = apply_tesseract(growth_step, growth_input)
        state = growth_output["state"]

        fruit_integral = fruit_integral + state["fruit"]

    # Seed production
    seed_input = {
        "fruit_integral": fruit_integral,
        "final_energy": state["energy"],
        "num_days": num_days,
        "seed_energy_threshold": 0.5,
        "seed_conversion": 10.0,
    }
    seed_output = apply_tesseract(seed_production, seed_input)

    return seed_output["seeds"]


def init_policy_weights(key: Array, hidden_size: int = 32) -> dict:
    """Initialize random neural policy weights."""
    input_size = 12  # 8 state + 1 progress + 3 environment
    output_size = 5  # allocation logits

    k1, k2 = jr.split(key)

    # Xavier initialization
    w1_scale = jnp.sqrt(2.0 / (input_size + hidden_size))
    w2_scale = jnp.sqrt(2.0 / (hidden_size + output_size))

    return {
        "w1": jr.normal(k1, (hidden_size, input_size)) * w1_scale,
        "b1": jnp.zeros(hidden_size),
        "w2": jr.normal(k2, (output_size, hidden_size)) * w2_scale,
        "b2": jnp.zeros(output_size),
    }


def main() -> None:
    print("\n" + "=" * 60)
    print("  ARBORHEDRON: Differentiable Tree Growth Pipeline")
    print("=" * 60)

    # Load Tesseracts
    growth_step = Tesseract.from_image("growth_step")
    neural_policy = Tesseract.from_image("neural_policy")
    seed_production = Tesseract.from_image("seed_production")

    with growth_step, neural_policy, seed_production:
        # Initialize random policy
        key = jr.PRNGKey(42)
        policy_weights = init_policy_weights(key)

        # PATH 1: Manual Tesseract calls (for debugging)
        print("\n" + "=" * 60)
        print("PATH 1: Season Simulation via Tesseract SDK")
        print("=" * 60)
        result = run_season_manual(
            growth_step, neural_policy, seed_production,
            policy_weights, num_days=100
        )

        # PATH 2: Differentiable composition with Tesseract-JAX
        print("\n" + "=" * 60)
        print("PATH 2: Differentiable Pipeline via Tesseract-JAX")
        print("=" * 60)

        # Define loss function (negative seeds for minimization)
        def loss_fn(weights: dict) -> Array:
            seeds = run_season_jax(
                growth_step, neural_policy, seed_production,
                weights, num_days=100
            )
            return -seeds  # Negative because we want to maximize seeds

        # Compute gradient of seeds with respect to policy weights
        print("\nComputing gradient of seed production w.r.t. policy weights...")
        grad_fn = jax.grad(loss_fn)
        gradients = grad_fn(policy_weights)

        print(f"Gradient norms:")
        print(f"  w1: {jnp.linalg.norm(gradients['w1']):.4f}")
        print(f"  b1: {jnp.linalg.norm(gradients['b1']):.4f}")
        print(f"  w2: {jnp.linalg.norm(gradients['w2']):.4f}")
        print(f"  b2: {jnp.linalg.norm(gradients['b2']):.4f}")

        # One gradient descent step
        print("\nPerforming one gradient descent step...")
        learning_rate = 0.01
        updated_weights = jax.tree.map(
            lambda w, g: w - learning_rate * g,
            policy_weights, gradients
        )

        # Evaluate updated policy
        new_seeds = run_season_jax(
            growth_step, neural_policy, seed_production,
            updated_weights, num_days=100
        )
        print(f"Seeds after update: {float(new_seeds):.3f}")
        print(f"Improvement: {float(new_seeds - result['seeds']):.3f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nThis demonstrates end-to-end differentiable composition:")
    print("  1. growth_step: Single day tree dynamics")
    print("  2. neural_policy: Allocation decisions")
    print("  3. seed_production: Fitness function")
    print("\nGradients flow through all three Tesseracts,")
    print("enabling optimization of the policy via gradient descent.")


if __name__ == "__main__":
    main()
