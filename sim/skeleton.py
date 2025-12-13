"""
Geometric skeleton representation for tree morphology.

This module implements a fixed-topology tree skeleton with continuous parameters.
Each segment has:
    - length: How far this branch extends
    - thickness: Wood investment (structural support)
    - alive: Soft gate [0,1] controlling branch existence

The topology is a binary tree of depth D, giving 2^D - 1 segments.
Segment 0 is the trunk; segments at depth D-1 are tips where leaves grow.

This representation enables:
    1. Spatial leaf placement (tips have positions in 2D/3D)
    2. Self-shading computation (overlapping leaf areas block light)
    3. Structural constraints (branches can't be longer than supported by trunk)
    4. Wind exposure (exposed branches take more damage)

All operations are differentiable for gradient-based optimization.
"""

from typing import NamedTuple
import jax.numpy as jnp
from jax import Array
from jax.nn import sigmoid


class SkeletonState(NamedTuple):
    """
    State of the tree skeleton.

    All arrays have shape [num_segments] where num_segments = 2^depth - 1.

    Attributes:
        length: Extension of each segment (0 = stub, max = fully grown)
        thickness: Wood investment in each segment (structural support)
        alive: Soft gate [0,1] for segment existence
        leaf_area: Leaf biomass at tips (only meaningful for tip segments)
        flower_area: Flower biomass at tips
    """
    length: Array       # [num_segments]
    thickness: Array    # [num_segments]
    alive: Array        # [num_segments] in [0,1]
    leaf_area: Array    # [num_segments] (tips only)
    flower_area: Array  # [num_segments] (tips only)

    @classmethod
    def initial(cls, depth: int = 4) -> "SkeletonState":
        """Create initial skeleton state (seedling)."""
        num_segments = 2**depth - 1
        num_tips = 2**(depth - 1)

        # Start with small trunk, everything else near-zero
        length = jnp.zeros(num_segments)
        length = length.at[0].set(0.1)  # Initial trunk

        thickness = jnp.zeros(num_segments)
        thickness = thickness.at[0].set(0.05)  # Initial trunk thickness

        # Trunk is alive, other segments start dormant
        alive = jnp.zeros(num_segments)
        alive = alive.at[0].set(1.0)  # Trunk alive

        # Small initial leaf area at first tier branches
        leaf_area = jnp.zeros(num_segments)
        flower_area = jnp.zeros(num_segments)

        return cls(
            length=length,
            thickness=thickness,
            alive=alive,
            leaf_area=leaf_area,
            flower_area=flower_area,
        )

    @property
    def num_segments(self) -> int:
        return len(self.length)

    @property
    def depth(self) -> int:
        """Tree depth (trunk = depth 0)."""
        return int(jnp.log2(self.num_segments + 1))

    def total_wood(self) -> Array:
        """Total wood biomass (length * thickness * alive)."""
        return jnp.sum(self.length * self.thickness * self.alive)

    def total_leaves(self) -> Array:
        """Total leaf area across all tips."""
        return jnp.sum(self.leaf_area * self.alive)

    def total_flowers(self) -> Array:
        """Total flower area across all tips."""
        return jnp.sum(self.flower_area * self.alive)


def get_tip_indices(depth: int) -> Array:
    """Get indices of tip segments (deepest level)."""
    first_tip = 2**(depth - 1) - 1
    num_tips = 2**(depth - 1)
    return jnp.arange(first_tip, first_tip + num_tips)


def get_parent(idx: int) -> int:
    """Get parent segment index (0 for trunk has no parent)."""
    if idx == 0:
        return -1
    return (idx - 1) // 2


def get_children(idx: int, num_segments: int) -> tuple[int, int]:
    """Get children segment indices (-1 if no children)."""
    left = 2 * idx + 1
    right = 2 * idx + 2
    if left >= num_segments:
        return -1, -1
    return left, right


def compute_segment_positions_2d(
    skeleton: SkeletonState,
    base_angle_spread: float = jnp.pi / 3,  # 60 degrees between branches
) -> tuple[Array, Array]:
    """
    Compute 2D positions of segment endpoints.

    Uses a simple recursive layout:
    - Trunk points upward from origin
    - Each branch splits at an angle from parent

    Returns:
        x_positions: [num_segments] x-coordinates of segment endpoints
        y_positions: [num_segments] y-coordinates of segment endpoints
    """
    num_segments = skeleton.num_segments
    depth = skeleton.depth

    # Base positions and angles
    x = jnp.zeros(num_segments)
    y = jnp.zeros(num_segments)

    # Trunk: vertical from (0,0)
    # Position represents the END of each segment
    x = x.at[0].set(0.0)
    y = y.at[0].set(skeleton.length[0] * skeleton.alive[0])

    # Compute positions level by level
    def get_base_position(idx):
        """Get the starting position of a segment (= end of parent)."""
        if idx == 0:
            return 0.0, 0.0
        parent = (idx - 1) // 2
        return x[parent], y[parent]

    def get_angle(idx, depth_level):
        """Get the growth angle for a segment."""
        if idx == 0:
            return jnp.pi / 2  # Trunk grows up

        # Alternate left/right from parent
        is_left = (idx % 2 == 1)
        parent_angle = get_angle((idx - 1) // 2, depth_level - 1)

        # Spread decreases with depth for natural look
        spread = base_angle_spread / (depth_level + 1)

        if is_left:
            return parent_angle + spread
        else:
            return parent_angle - spread

    # Build positions iteratively (can't use recursion in JAX)
    # For a depth-4 tree, we have levels 0, 1, 2, 3
    angles = jnp.zeros(num_segments)
    angles = angles.at[0].set(jnp.pi / 2)  # Trunk up

    for level in range(1, depth):
        start_idx = 2**level - 1
        end_idx = 2**(level + 1) - 1

        for idx in range(start_idx, min(end_idx, num_segments)):
            parent = (idx - 1) // 2
            parent_angle = float(angles[parent])

            is_left = (idx % 2 == 1)
            spread = base_angle_spread / (level + 1)

            if is_left:
                angles = angles.at[idx].set(parent_angle + spread)
            else:
                angles = angles.at[idx].set(parent_angle - spread)

            # Position = parent end + length * direction
            parent_x, parent_y = float(x[parent]), float(y[parent])
            length = skeleton.length[idx] * skeleton.alive[idx]

            x = x.at[idx].set(parent_x + length * jnp.cos(angles[idx]))
            y = y.at[idx].set(parent_y + length * jnp.sin(angles[idx]))

    return x, y


def compute_light_capture(
    skeleton: SkeletonState,
    light_direction: float = 0.0,  # 0 = directly overhead
    k_shade: float = 0.5,  # shading coefficient
) -> Array:
    """
    Compute effective light capture for each tip.

    Simple model: tips higher up capture more light.
    Self-shading: leaf area above reduces light to leaves below.

    Returns:
        light_capture: [num_segments] effective light per tip
    """
    x, y = compute_segment_positions_2d(skeleton)
    tip_indices = get_tip_indices(skeleton.depth)

    # For now, simple height-based model
    # Light capture proportional to height, reduced by self-shading
    light = jnp.zeros(skeleton.num_segments)

    for tip_idx in tip_indices:
        tip_y = y[int(tip_idx)]
        leaf_area = skeleton.leaf_area[int(tip_idx)] * skeleton.alive[int(tip_idx)]

        # Higher = more light (normalized)
        height_factor = sigmoid(tip_y - 0.5)  # Smooth threshold

        # Self-shading from other leaves above
        total_above = 0.0
        for other_tip in tip_indices:
            if int(other_tip) != int(tip_idx):
                other_y = y[int(other_tip)]
                other_leaf = skeleton.leaf_area[int(other_tip)] * skeleton.alive[int(other_tip)]
                # Only shade if above (with soft comparison)
                is_above = sigmoid(10 * (other_y - tip_y))
                total_above = total_above + is_above * other_leaf

        shade_factor = jnp.exp(-k_shade * total_above)

        # Light capture = leaf area * height factor * shade factor
        light = light.at[int(tip_idx)].set(leaf_area * height_factor * shade_factor)

    return light


def compute_wind_exposure(
    skeleton: SkeletonState,
    wind: float,
) -> Array:
    """
    Compute wind exposure for each segment.

    Higher segments and those with more leaf area are more exposed.
    Thickness provides protection.

    Returns:
        exposure: [num_segments] wind exposure factor for each segment
    """
    x, y = compute_segment_positions_2d(skeleton)

    # Exposure increases with height and leaf load
    height_factor = sigmoid(y - 0.3)  # Higher = more exposed

    # Leaf load increases exposure (catches wind)
    tip_indices = get_tip_indices(skeleton.depth)
    leaf_load = jnp.zeros(skeleton.num_segments)
    for tip_idx in tip_indices:
        leaf_load = leaf_load.at[int(tip_idx)].set(
            skeleton.leaf_area[int(tip_idx)] * skeleton.alive[int(tip_idx)]
        )

    # Thickness provides protection (thicker branches resist wind)
    protection = skeleton.thickness / (skeleton.thickness + 0.5)

    exposure = wind * height_factor * (1.0 + leaf_load) * (1.0 - 0.5 * protection)

    return exposure * skeleton.alive


def skeleton_to_scalar_state(skeleton: SkeletonState) -> dict:
    """
    Convert skeleton state to scalar compartments for compatibility.

    This allows using skeleton representation with existing dynamics.
    """
    return {
        'trunk': float(skeleton.thickness[0] * skeleton.length[0]),
        'shoots': float(skeleton.total_wood() - skeleton.thickness[0] * skeleton.length[0]),
        'leaves': float(skeleton.total_leaves()),
        'flowers': float(skeleton.total_flowers()),
    }
