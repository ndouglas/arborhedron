# Tesseract API module for neural_policy
# Differentiable neural network policy for tree growth allocation

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn import softmax
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

#
# Schemata
#


class TreeStateSchema(BaseModel):
    """Tree state for policy input."""

    energy: Differentiable[Float32] = Field(description="Energy store")
    water: Differentiable[Float32] = Field(description="Internal water store")
    nutrients: Differentiable[Float32] = Field(description="Nutrient store")
    roots: Differentiable[Float32] = Field(description="Root biomass")
    trunk: Differentiable[Float32] = Field(description="Trunk/wood biomass")
    shoots: Differentiable[Float32] = Field(description="Shoot biomass")
    leaves: Differentiable[Float32] = Field(description="Leaf biomass")
    flowers: Differentiable[Float32] = Field(description="Flower biomass")


class PolicyWeightsSchema(BaseModel):
    """Neural policy weights (2-layer MLP: 12 -> hidden -> 5)."""

    # Layer 1: 12 inputs -> hidden_size
    w1: Differentiable[Array[(None, None), Float32]] = Field(
        description="First layer weights [hidden_size, 12]"
    )
    b1: Differentiable[Array[(None,), Float32]] = Field(
        description="First layer biases [hidden_size]"
    )

    # Layer 2: hidden_size -> 5 outputs
    w2: Differentiable[Array[(None, None), Float32]] = Field(
        description="Second layer weights [5, hidden_size]"
    )
    b2: Differentiable[Array[(None,), Float32]] = Field(
        description="Second layer biases [5]"
    )


class AllocationSchema(BaseModel):
    """Resource allocation fractions (sum to 1)."""

    roots: Differentiable[Float32] = Field(description="Fraction to roots")
    trunk: Differentiable[Float32] = Field(description="Fraction to trunk")
    shoots: Differentiable[Float32] = Field(description="Fraction to shoots")
    leaves: Differentiable[Float32] = Field(description="Fraction to leaves")
    flowers: Differentiable[Float32] = Field(description="Fraction to flowers")


class InputSchema(BaseModel):
    """Input schema for neural_policy tesseract."""

    # Current tree state (8 values, excluding fruit and soil_water which policy doesn't see)
    state: TreeStateSchema = Field(description="Current tree state")

    # Environment
    light: Differentiable[Float32] = Field(description="Light availability [0, 1]")
    moisture: Differentiable[Float32] = Field(description="Soil moisture [0, 1]")
    wind: Differentiable[Float32] = Field(description="Wind speed [0, 1]")

    # Timing
    day: int = Field(description="Current day (0-indexed)")
    num_days: int = Field(default=100, description="Total season length")

    # Policy weights (differentiable for training)
    weights: PolicyWeightsSchema = Field(description="Neural policy weights")


class OutputSchema(BaseModel):
    """Output schema for neural_policy tesseract."""

    allocation: AllocationSchema = Field(description="Resource allocation fractions")
    logits: Differentiable[Array[(5,), Float32]] = Field(
        description="Raw logits before softmax"
    )


#
# Core policy logic
#


def make_features(state: dict, day: int, num_days: int, light, moisture, wind) -> jnp.ndarray:
    """
    Extract input features for neural policy.

    Features are lightly normalized to roughly [-1, 2] range.
    """
    # State features (divide by typical max values for rough normalization)
    state_features = jnp.array([
        state["energy"],           # [0, 1] typically
        state["water"],            # [0, 1] typically
        state["nutrients"],        # [0, 1] typically
        state["roots"] / 2.0,      # normalize to ~[0, 1.5]
        state["trunk"] / 2.0,
        state["shoots"] / 2.0,
        state["leaves"] / 2.0,
        state["flowers"] / 2.0,
    ])

    # Progress through season [0, 1]
    progress = jnp.array([day / num_days])

    # Environment features (already in [0, 1])
    env_features = jnp.array([light, moisture, wind])

    return jnp.concatenate([state_features, progress, env_features])


def mlp_forward(features: jnp.ndarray, w1, b1, w2, b2) -> jnp.ndarray:
    """
    Forward pass through 2-layer MLP.

    Args:
        features: Input features [12]
        w1, b1: First layer weights and biases
        w2, b2: Second layer weights and biases

    Returns:
        Logits [5]
    """
    # Layer 1: linear + tanh
    h = jnp.tanh(w1 @ features + b1)

    # Layer 2: linear (output logits)
    logits = w2 @ h + b2

    return logits


#
# Required endpoints
#


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    """JIT-compiled neural policy forward pass."""
    # Extract state
    state = inputs["state"]

    # Extract environment
    light = jnp.array(inputs["light"])
    moisture = jnp.array(inputs["moisture"])
    wind = jnp.array(inputs["wind"])
    day = inputs["day"]
    num_days = inputs["num_days"]

    # Extract weights
    weights = inputs["weights"]
    w1 = jnp.array(weights["w1"])
    b1 = jnp.array(weights["b1"])
    w2 = jnp.array(weights["w2"])
    b2 = jnp.array(weights["b2"])

    # Build features
    features = make_features(state, day, num_days, light, moisture, wind)

    # Forward pass
    logits = mlp_forward(features, w1, b1, w2, b2)

    # Softmax to allocation
    probs = softmax(logits)

    return {
        "allocation": {
            "roots": probs[0],
            "trunk": probs[1],
            "shoots": probs[2],
            "leaves": probs[3],
            "flowers": probs[4],
        },
        "logits": logits,
    }


def apply(inputs: InputSchema) -> OutputSchema:
    """Apply the neural policy."""
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


#
# Utility: Initialize random weights
#


def init_weights(key, hidden_size: int = 32) -> dict:
    """
    Initialize random policy weights.

    Args:
        key: JAX random key
        hidden_size: Number of hidden units

    Returns:
        Dictionary matching PolicyWeightsSchema
    """
    import jax.random as jr

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
