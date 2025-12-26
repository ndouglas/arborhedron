# Tesseract API module for growth_step
# Differentiable single-day tree growth simulation step

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

#
# Schemata
#


class TreeStateSchema(BaseModel):
    """Tree state with 10 compartments."""

    energy: Differentiable[Float32] = Field(description="Energy store")
    water: Differentiable[Float32] = Field(description="Internal water store")
    nutrients: Differentiable[Float32] = Field(description="Nutrient store")
    roots: Differentiable[Float32] = Field(description="Root biomass")
    trunk: Differentiable[Float32] = Field(description="Trunk/wood biomass")
    shoots: Differentiable[Float32] = Field(description="Shoot biomass")
    leaves: Differentiable[Float32] = Field(description="Leaf biomass")
    flowers: Differentiable[Float32] = Field(description="Flower biomass")
    fruit: Differentiable[Float32] = Field(description="Fruit biomass")
    soil_water: Differentiable[Float32] = Field(description="Soil water reservoir")


class AllocationSchema(BaseModel):
    """Resource allocation fractions (should sum to 1)."""

    roots: Differentiable[Float32] = Field(description="Fraction to roots")
    trunk: Differentiable[Float32] = Field(description="Fraction to trunk")
    shoots: Differentiable[Float32] = Field(description="Fraction to shoots")
    leaves: Differentiable[Float32] = Field(description="Fraction to leaves")
    flowers: Differentiable[Float32] = Field(description="Fraction to flowers")


class InputSchema(BaseModel):
    """Input schema for growth_step tesseract."""

    # Current tree state
    state: TreeStateSchema = Field(description="Current tree state")

    # Resource allocation
    allocation: AllocationSchema = Field(description="Resource allocation fractions")

    # Environment
    light: Differentiable[Float32] = Field(description="Light availability [0, 1]")
    moisture: Differentiable[Float32] = Field(description="Soil moisture [0, 1]")
    wind: Differentiable[Float32] = Field(description="Wind speed [0, 1]")

    # Timing
    day: int = Field(description="Current day (0-indexed)")
    num_days: int = Field(default=100, description="Total season length")


class OutputSchema(BaseModel):
    """Output schema for growth_step tesseract."""

    state: TreeStateSchema = Field(description="New tree state after one day")


#
# Core simulation logic (imported from local sim module copy)
#

from sim.config import Allocation, SimConfig, TreeState
from sim.dynamics import step as dynamics_step


def dict_to_tree_state(d: dict) -> TreeState:
    """Convert dict to TreeState."""
    state_d = d if "state" not in d else d["state"]
    return TreeState(
        energy=jnp.array(state_d["energy"]),
        water=jnp.array(state_d["water"]),
        nutrients=jnp.array(state_d["nutrients"]),
        roots=jnp.array(state_d["roots"]),
        trunk=jnp.array(state_d["trunk"]),
        shoots=jnp.array(state_d["shoots"]),
        leaves=jnp.array(state_d["leaves"]),
        flowers=jnp.array(state_d["flowers"]),
        fruit=jnp.array(state_d["fruit"]),
        soil_water=jnp.array(state_d["soil_water"]),
    )


def dict_to_allocation(d: dict) -> Allocation:
    """Convert dict to Allocation."""
    alloc_d = d if "allocation" not in d else d["allocation"]
    return Allocation(
        roots=jnp.array(alloc_d["roots"]),
        trunk=jnp.array(alloc_d["trunk"]),
        shoots=jnp.array(alloc_d["shoots"]),
        leaves=jnp.array(alloc_d["leaves"]),
        flowers=jnp.array(alloc_d["flowers"]),
    )


def tree_state_to_dict(state: TreeState) -> dict:
    """Convert TreeState to dict."""
    return {
        "energy": state.energy,
        "water": state.water,
        "nutrients": state.nutrients,
        "roots": state.roots,
        "trunk": state.trunk,
        "shoots": state.shoots,
        "leaves": state.leaves,
        "flowers": state.flowers,
        "fruit": state.fruit,
        "soil_water": state.soil_water,
    }


#
# Required endpoints
#


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    """JIT-compiled growth step."""
    # Extract inputs
    state = dict_to_tree_state(inputs["state"])
    allocation = dict_to_allocation(inputs["allocation"])
    light = jnp.array(inputs["light"])
    moisture = jnp.array(inputs["moisture"])
    wind = jnp.array(inputs["wind"])
    day = inputs["day"]

    # Use default config
    config = SimConfig()

    # Run one step of dynamics
    new_state = dynamics_step(
        state=state,
        allocation=allocation,
        light=light,
        moisture=moisture,
        wind=wind,
        config=config,
        day=day,
    )

    # Return as dict
    return {"state": tree_state_to_dict(new_state)}


def apply(inputs: InputSchema) -> OutputSchema:
    """Apply the growth step."""
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
