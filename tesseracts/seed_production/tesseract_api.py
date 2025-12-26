# Tesseract API module for seed_production
# Differentiable seed production from fruit integral (fitness function)

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from pydantic import BaseModel, Field

from tesseract_core.runtime import Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

#
# Schemata
#


class InputSchema(BaseModel):
    """Input schema for seed_production tesseract."""

    # Accumulated fruit over the season (integral)
    fruit_integral: Differentiable[Float32] = Field(
        description="Sum of fruit biomass over all days"
    )

    # Final energy level (gates seed viability)
    final_energy: Differentiable[Float32] = Field(
        description="Energy level at end of season"
    )

    # Configuration
    num_days: int = Field(default=100, description="Total season length")
    seed_energy_threshold: float = Field(
        default=0.5, description="Minimum energy to produce seeds"
    )
    seed_conversion: float = Field(
        default=10.0, description="Seeds per unit fruit biomass"
    )


class OutputSchema(BaseModel):
    """Output schema for seed_production tesseract."""

    seeds: Differentiable[Float32] = Field(description="Number of seeds produced")
    energy_gate: Differentiable[Float32] = Field(
        description="Energy viability gate [0, 1]"
    )
    normalized_fruit: Differentiable[Float32] = Field(
        description="Fruit integral normalized by season length"
    )


#
# Core computation
#


def seed_production(
    fruit_integral: jnp.ndarray,
    final_energy: jnp.ndarray,
    num_days: int,
    seed_energy_threshold: float,
    seed_conversion: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute seed production from integrated fruit.

    Seeds = conversion * (fruit_integral / num_days) * energy_gate

    The energy gate uses a sigmoid: σ(10 * (energy - threshold))
    This creates smooth differentiable gating while enforcing that
    depleted trees can't produce viable seeds.

    Args:
        fruit_integral: Sum of fruit biomass over season (fruit-days)
        final_energy: Final energy level
        num_days: Season length for normalization
        seed_energy_threshold: Energy level for 50% seed viability
        seed_conversion: Seeds per unit normalized fruit

    Returns:
        Tuple of (seeds, energy_gate, normalized_fruit)
    """
    # Normalize by season length to get "average fruit"
    normalized_fruit = fruit_integral / num_days

    # Energy gate: sigmoid with steepness 10
    energy_gate = sigmoid(10.0 * (final_energy - seed_energy_threshold))

    # Seeds = conversion * normalized_fruit * energy_gate
    seeds = seed_conversion * normalized_fruit * energy_gate

    return seeds, energy_gate, normalized_fruit


#
# Required endpoints
#


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    """JIT-compiled seed production."""
    fruit_integral = jnp.array(inputs["fruit_integral"])
    final_energy = jnp.array(inputs["final_energy"])
    num_days = inputs["num_days"]
    seed_energy_threshold = inputs["seed_energy_threshold"]
    seed_conversion = inputs["seed_conversion"]

    seeds, energy_gate, normalized_fruit = seed_production(
        fruit_integral=fruit_integral,
        final_energy=final_energy,
        num_days=num_days,
        seed_energy_threshold=seed_energy_threshold,
        seed_conversion=seed_conversion,
    )

    return {
        "seeds": seeds,
        "energy_gate": energy_gate,
        "normalized_fruit": normalized_fruit,
    }


def apply(inputs: InputSchema) -> OutputSchema:
    """Apply seed production calculation."""
    out = apply_jit(inputs.model_dump())
    return out


#
# JAX-handled AD endpoints
#


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    return jac_jit(inputs.model_dump(), tuple(jac_inputs), tuple(jac_outputs))


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return jvp_jit(
        inputs.model_dump(),
        tuple(jvp_inputs),
        tuple(jvp_outputs),
        tangent_vector,
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return vjp_jit(
        inputs.model_dump(),
        tuple(vjp_inputs),
        tuple(vjp_outputs),
        cotangent_vector,
    )


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    is_shapedtype_dict = lambda x: type(x) is dict and (x.keys() == {"shape", "dtype"})
    is_shapedtype_struct = lambda x: isinstance(x, jax.ShapeDtypeStruct)

    jaxified_inputs = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(**x) if is_shapedtype_dict(x) else x,
        abstract_inputs.model_dump(),
        is_leaf=is_shapedtype_dict,
    )
    dynamic_inputs, static_inputs = eqx.partition(
        jaxified_inputs, filter_spec=is_shapedtype_struct
    )

    def wrapped_apply(dynamic_inputs):
        inputs = eqx.combine(static_inputs, dynamic_inputs)
        return apply_jit(inputs)

    jax_shapes = jax.eval_shape(wrapped_apply, dynamic_inputs)
    return jax.tree.map(
        lambda x: (
            {"shape": x.shape, "dtype": str(x.dtype)} if is_shapedtype_struct(x) else x
        ),
        jax_shapes,
        is_leaf=is_shapedtype_struct,
    )


#
# Helper functions
#


@eqx.filter_jit
def jac_jit(
    inputs: dict,
    jac_inputs: tuple[str],
    jac_outputs: tuple[str],
):
    filtered_apply = filter_func(apply_jit, inputs, jac_outputs)
    return jax.jacrev(filtered_apply)(
        flatten_with_paths(inputs, include_paths=jac_inputs)
    )


@eqx.filter_jit
def jvp_jit(
    inputs: dict, jvp_inputs: tuple[str], jvp_outputs: tuple[str], tangent_vector: dict
):
    filtered_apply = filter_func(apply_jit, inputs, jvp_outputs)
    return jax.jvp(
        filtered_apply,
        [flatten_with_paths(inputs, include_paths=jvp_inputs)],
        [tangent_vector],
    )[1]


@eqx.filter_jit
def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str],
    vjp_outputs: tuple[str],
    cotangent_vector: dict,
):
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    return vjp_func(cotangent_vector)[0]
