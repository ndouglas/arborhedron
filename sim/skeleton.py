"""
Geometric skeleton representation for tree morphology.

This module implements a fixed-topology tree skeleton with continuous parameters.
Each segment has:
    - length: How far this branch extends
    - thickness: Wood investment (structural support)
    - alive: Soft gate [0,1] controlling branch existence
    - angle: Growth direction (relative to parent)

The topology is a binary tree of depth D, giving 2^D - 1 segments.
Segment 0 is the trunk; segments at depth D-1 are tips where leaves grow.

This representation enables:
    1. Spatial leaf placement (tips have positions in 2D/3D)
    2. Self-shading computation (Gaussian splat method)
    3. Structural constraints (branches can't be longer than supported by trunk)
    4. Wind exposure (orientation-based projected area)

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
        angle: Growth angle relative to parent (radians, + = left, - = right)
        leaf_area: Leaf biomass at tips (only meaningful for tip segments)
        flower_area: Flower biomass at tips
    """
    length: Array       # [num_segments]
    thickness: Array    # [num_segments]
    alive: Array        # [num_segments] in [0,1]
    angle: Array        # [num_segments] relative angle in radians
    leaf_area: Array    # [num_segments] (tips only)
    flower_area: Array  # [num_segments] (tips only)

    @classmethod
    def initial(cls, depth: int = 8) -> "SkeletonState":
        """Create initial skeleton state (seedling).

        Default depth=8 gives 255 segments and 128 tips for dense canopy.
        """
        num_segments = 2**depth - 1

        # Start with small trunk, everything else near-zero
        length = jnp.zeros(num_segments)
        length = length.at[0].set(0.1)  # Initial trunk

        thickness = jnp.zeros(num_segments)
        thickness = thickness.at[0].set(0.05)  # Initial trunk thickness

        # Trunk is alive, other segments start dormant
        alive = jnp.zeros(num_segments)
        alive = alive.at[0].set(1.0)  # Trunk alive

        # Default angles: trunk up, branches spread outward
        # Angle is relative to parent: + = left, - = right
        angle = jnp.zeros(num_segments)
        angle = angle.at[0].set(jnp.pi / 2)  # Trunk: straight up (absolute)

        # Set default branch spread angles
        for level in range(1, depth):
            start_idx = 2**level - 1
            end_idx = 2**(level + 1) - 1
            spread = jnp.pi / 4 / (level + 1)  # Decreasing spread with depth
            for idx in range(start_idx, min(end_idx, num_segments)):
                is_left = (idx % 2 == 1)
                angle = angle.at[idx].set(spread if is_left else -spread)

        # No initial leaves or flowers
        leaf_area = jnp.zeros(num_segments)
        flower_area = jnp.zeros(num_segments)

        return cls(
            length=length,
            thickness=thickness,
            alive=alive,
            angle=angle,
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
) -> tuple[Array, Array, Array]:
    """
    Compute 2D positions of segment endpoints using stored angles.

    Returns:
        x_positions: [num_segments] x-coordinates of segment endpoints
        y_positions: [num_segments] y-coordinates of segment endpoints
        absolute_angles: [num_segments] absolute angle of each segment (for wind calc)
    """
    num_segments = skeleton.num_segments
    depth = skeleton.depth

    x = jnp.zeros(num_segments)
    y = jnp.zeros(num_segments)
    absolute_angles = jnp.zeros(num_segments)

    # Trunk: position at (0, length), angle from skeleton.angle[0]
    trunk_angle = skeleton.angle[0]  # Should be ~pi/2 for vertical
    trunk_length = skeleton.length[0] * skeleton.alive[0]
    x = x.at[0].set(trunk_length * jnp.cos(trunk_angle))
    y = y.at[0].set(trunk_length * jnp.sin(trunk_angle))
    absolute_angles = absolute_angles.at[0].set(trunk_angle)

    # Build positions level by level
    for level in range(1, depth):
        start_idx = 2**level - 1
        end_idx = 2**(level + 1) - 1

        for idx in range(start_idx, min(end_idx, num_segments)):
            parent = (idx - 1) // 2

            # Absolute angle = parent's absolute angle + this segment's relative angle
            parent_abs_angle = absolute_angles[parent]
            rel_angle = skeleton.angle[idx]
            abs_angle = parent_abs_angle + rel_angle
            absolute_angles = absolute_angles.at[idx].set(abs_angle)

            # Position = parent end + length * direction
            parent_x, parent_y = x[parent], y[parent]
            length = skeleton.length[idx] * skeleton.alive[idx]

            x = x.at[idx].set(parent_x + length * jnp.cos(abs_angle))
            y = y.at[idx].set(parent_y + length * jnp.sin(abs_angle))

    return x, y, absolute_angles


def compute_light_capture(
    skeleton: SkeletonState,
    base_light: float = 1.0,  # Ambient light intensity
    shade_sigma: float = 0.3,  # Gaussian splat width for shading
    shade_beta: float = 2.0,  # Shading strength coefficient
) -> Array:
    """
    Compute effective light capture using Gaussian splat self-shading.

    Each leaf is a Gaussian "ink splat" on a 2D plane. Leaves shade each other
    based on overlap and relative height.

    Physics:
    - Light comes from above (y direction)
    - Each leaf casts a shadow proportional to its area
    - Shadow intensity decreases with horizontal distance (Gaussian falloff)
    - Lower leaves receive less light if shaded by leaves above

    Returns:
        light_capture: [num_segments] effective light per tip
    """
    x, y, _ = compute_segment_positions_2d(skeleton)
    tip_indices = get_tip_indices(skeleton.depth)

    light = jnp.zeros(skeleton.num_segments)

    for tip_idx in tip_indices:
        tip_idx = int(tip_idx)
        tip_x, tip_y = x[tip_idx], y[tip_idx]
        leaf_area = skeleton.leaf_area[tip_idx] * skeleton.alive[tip_idx]

        if leaf_area < 1e-6:
            continue

        # Compute shadow density at this leaf's position from all other leaves
        # Shadow comes from leaves that are ABOVE this one
        shadow_density = 0.0

        for other_tip in tip_indices:
            other_tip = int(other_tip)
            if other_tip == tip_idx:
                continue

            other_x, other_y = x[other_tip], y[other_tip]
            other_leaf = skeleton.leaf_area[other_tip] * skeleton.alive[other_tip]

            if other_leaf < 1e-6:
                continue

            # Only count leaves above (soft comparison for differentiability)
            height_diff = other_y - tip_y
            is_above = sigmoid(10.0 * height_diff)

            # Gaussian falloff based on horizontal distance
            dx = other_x - tip_x
            gaussian_weight = jnp.exp(-dx**2 / (2 * shade_sigma**2))

            # Shadow contribution = leaf area * above factor * proximity
            shadow_density = shadow_density + other_leaf * is_above * gaussian_weight

        # Light at this position = base light * exp(-beta * shadow_density)
        effective_light = base_light * jnp.exp(-shade_beta * shadow_density)

        # Light capture = leaf area * effective light
        light = light.at[tip_idx].set(leaf_area * effective_light)

    return light


def compute_wind_exposure(
    skeleton: SkeletonState,
    wind_speed: float,
    wind_direction: float = 0.0,  # Angle in radians (0 = from right, pi/2 = from above)
    height_exponent: float = 0.5,  # How exposure scales with height
) -> Array:
    """
    Compute wind exposure using orientation-based physics.

    Physics model:
    - Exposure depends on PROJECTED AREA perpendicular to wind
    - Segments oriented across wind catch more force
    - Higher segments experience more wind (boundary layer effect)
    - Leaf area downstream of segment adds to load

    Formula per segment:
        exposure = wind_speed * h^α * |sin(θ_segment - θ_wind)| * (1 + leaf_load) / protection

    Where:
        h = height (y position)
        α = height exponent (~0.5 for realistic boundary layer)
        θ_segment = segment orientation
        θ_wind = wind direction
        leaf_load = leaf area on this branch and downstream

    Returns:
        exposure: [num_segments] wind exposure factor for each segment
    """
    x, y, absolute_angles = compute_segment_positions_2d(skeleton)
    tip_indices = get_tip_indices(skeleton.depth)

    exposure = jnp.zeros(skeleton.num_segments)

    # Compute leaf load for each segment (its own leaves + all downstream)
    leaf_load = jnp.zeros(skeleton.num_segments)

    # First, set leaf area at tips
    for tip_idx in tip_indices:
        tip_idx = int(tip_idx)
        leaf_load = leaf_load.at[tip_idx].set(
            skeleton.leaf_area[tip_idx] * skeleton.alive[tip_idx]
        )

    # Propagate leaf load up the tree (children contribute to parent load)
    depth = skeleton.depth
    for level in range(depth - 1, 0, -1):
        start_idx = 2**level - 1
        end_idx = 2**(level + 1) - 1
        for idx in range(start_idx, min(end_idx, skeleton.num_segments)):
            parent = (idx - 1) // 2
            leaf_load = leaf_load.at[parent].add(leaf_load[idx] * skeleton.alive[idx])

    # Compute exposure for each segment
    for idx in range(skeleton.num_segments):
        alive = skeleton.alive[idx]
        if alive < 1e-6:
            continue

        # Height factor: h^α (with small offset to avoid zero)
        height = jnp.maximum(y[idx], 0.1)
        height_factor = height ** height_exponent

        # Orientation factor: |sin(θ_segment - θ_wind)|
        # Segments perpendicular to wind catch more force
        segment_angle = absolute_angles[idx]
        angle_diff = segment_angle - wind_direction
        orientation_factor = jnp.abs(jnp.sin(angle_diff))

        # Leaf load multiplier (branches with more downstream leaves are more exposed)
        load_factor = 1.0 + leaf_load[idx]

        # Thickness provides protection (thicker = more rigid, less deflection)
        thickness = skeleton.thickness[idx]
        protection = 1.0 / (1.0 + thickness)  # Thicker = less exposure

        # Total exposure
        segment_exposure = (
            wind_speed
            * height_factor
            * orientation_factor
            * load_factor
            * protection
        )

        exposure = exposure.at[idx].set(segment_exposure * alive)

    return exposure


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


# =============================================================================
# SOIL TILES - Spatial root allocation
# =============================================================================

class SoilTiles(NamedTuple):
    """
    Soil representation as discrete tiles with different properties.

    The ground is divided into tiles (e.g., 3 tiles: left, center, right).
    Each tile has different water and nutrient availability.
    Roots are allocated across tiles as fractions that sum to 1.

    This creates the "tree leans toward resources" dynamic:
    - Invest roots where water/nutrients are
    - Asymmetric investment → asymmetric form

    Attributes:
        water: [num_tiles] water availability per tile (0-1)
        nutrients: [num_tiles] nutrient availability per tile (0-1)
        root_allocation: [num_tiles] fraction of roots in each tile (sums to 1)
    """
    water: Array       # [num_tiles] water availability
    nutrients: Array   # [num_tiles] nutrient availability
    root_allocation: Array  # [num_tiles] root investment fractions

    @classmethod
    def default(cls, num_tiles: int = 3) -> "SoilTiles":
        """Create default soil with uniform properties."""
        return cls(
            water=jnp.ones(num_tiles) * 0.5,
            nutrients=jnp.ones(num_tiles) * 0.5,
            root_allocation=jnp.ones(num_tiles) / num_tiles,
        )

    @classmethod
    def asymmetric(cls, water_gradient: float = 0.3, nutrient_gradient: float = 0.2) -> "SoilTiles":
        """
        Create soil with gradients across tiles.

        Args:
            water_gradient: How much water varies (left = base - gradient, right = base + gradient)
            nutrient_gradient: How much nutrients vary

        Example: water_gradient=0.3 gives tiles with water [0.2, 0.5, 0.8]
        """
        base = 0.5
        return cls(
            water=jnp.array([base - water_gradient, base, base + water_gradient]),
            nutrients=jnp.array([base + nutrient_gradient, base, base - nutrient_gradient]),
            root_allocation=jnp.ones(3) / 3,
        )

    @property
    def num_tiles(self) -> int:
        return len(self.water)


def compute_root_uptake(
    tiles: SoilTiles,
    total_root_biomass: float,
    k_uptake: float = 0.5,
) -> tuple[float, float]:
    """
    Compute total water and nutrient uptake based on root allocation.

    Each tile contributes:
        uptake_i = root_fraction_i * root_biomass * resource_i / (resource_i + k)

    Total = sum over tiles.

    Returns:
        (water_uptake, nutrient_uptake)
    """
    # Michaelis-Menten saturation for each tile
    water_saturation = tiles.water / (tiles.water + k_uptake)
    nutrient_saturation = tiles.nutrients / (tiles.nutrients + k_uptake)

    # Weight by root allocation
    water_uptake = total_root_biomass * jnp.sum(tiles.root_allocation * water_saturation)
    nutrient_uptake = total_root_biomass * jnp.sum(tiles.root_allocation * nutrient_saturation)

    return float(water_uptake), float(nutrient_uptake)


def get_tile_positions(num_tiles: int = 3, width: float = 3.0) -> Array:
    """Get x-coordinates of tile centers."""
    tile_width = width / num_tiles
    return jnp.linspace(-width/2 + tile_width/2, width/2 - tile_width/2, num_tiles)


def compute_root_bias(tiles: SoilTiles) -> float:
    """
    Compute spatial bias of root allocation.

    Returns value in [-1, 1]:
        -1 = all roots on left
         0 = roots centered
        +1 = all roots on right

    This can be used to bias the tree's above-ground form.
    """
    positions = get_tile_positions(tiles.num_tiles)
    # Weighted average of positions
    center_of_mass = jnp.sum(tiles.root_allocation * positions)
    # Normalize to [-1, 1]
    max_position = positions[-1]
    return float(center_of_mass / max_position) if max_position > 0 else 0.0


# =============================================================================
# BILATERAL SYMMETRY HELPERS
# =============================================================================


def get_mirror_index(idx: int) -> int:
    """
    Get the mirror index for bilateral symmetry.

    In a binary tree:
    - Index 0 (trunk) mirrors to itself
    - Left child (odd index) mirrors to right sibling (even index)
    - Right child (even index) mirrors to left sibling (odd index)

    Example for depth=3:
              0           (trunk - self-mirror)
           /     \\
          1       2       (1 <-> 2)
         / \\    / \\
        3   4  5   6      (3 <-> 6, 4 <-> 5)

    Args:
        idx: Segment index

    Returns:
        Mirror segment index
    """
    if idx == 0:
        return 0

    # Find level and position within level
    level = int(jnp.floor(jnp.log2(idx + 1)))
    level_start = 2**level - 1
    pos_in_level = idx - level_start
    level_size = 2**level

    # Mirror position within level
    mirror_pos = level_size - 1 - pos_in_level
    return level_start + mirror_pos


def enforce_bilateral_symmetry(arr: Array, depth: int) -> Array:
    """
    Enforce bilateral symmetry on an array of segment values.

    For each pair of mirrored segments, set both to their average.
    This keeps the tree balanced while preserving overall magnitude.

    Args:
        arr: Array of shape [num_segments]
        depth: Tree depth

    Returns:
        Symmetrized array
    """
    num_segments = 2**depth - 1
    result = arr.copy() if hasattr(arr, 'copy') else jnp.array(arr)

    for idx in range(num_segments):
        mirror = get_mirror_index(idx)
        if mirror > idx:  # Only process each pair once
            avg = (arr[idx] + arr[mirror]) / 2
            result = result.at[idx].set(avg)
            result = result.at[mirror].set(avg)

    return result


def create_grown_skeleton(
    depth: int = 8,
    growth_stage: float = 1.0,
    prune_fraction: float = 0.3,
    jitter_strength: float = 0.1,
    symmetric: bool = True,
    seed: int = 42,
) -> SkeletonState:
    """
    Create a fully-grown skeleton with natural variation.

    This is the main function for generating tree forms for visualization.

    Args:
        depth: Tree depth (8 recommended for dense canopy)
        growth_stage: 0-1, how mature the tree is
        prune_fraction: Fraction of branches to prune (set alive=0)
        jitter_strength: Random variation in angles/lengths
        symmetric: Enforce bilateral symmetry
        seed: Random seed for reproducibility

    Returns:
        SkeletonState with grown tree
    """
    import numpy as np
    np.random.seed(seed)

    num_segments = 2**depth - 1
    tip_start = 2**(depth - 1) - 1

    # Initialize arrays
    length = np.zeros(num_segments)
    thickness = np.zeros(num_segments)
    alive = np.ones(num_segments)
    angle = np.zeros(num_segments)
    leaf_area = np.zeros(num_segments)
    flower_area = np.zeros(num_segments)

    # Trunk
    angle[0] = np.pi / 2  # Straight up
    length[0] = 0.5 * growth_stage
    thickness[0] = 0.15 * growth_stage

    # Build tree level by level
    for level in range(1, depth):
        level_start = 2**level - 1
        level_end = 2**(level + 1) - 1
        depth_factor = 1.0 - 0.12 * level  # Taper with depth

        for idx in range(level_start, min(level_end, num_segments)):
            # Base spread angle (left vs right)
            is_left = (idx % 2 == 1)
            base_spread = np.pi / 5 / (level + 1)
            angle[idx] = base_spread if is_left else -base_spread

            # Add jitter
            angle[idx] += np.random.uniform(-jitter_strength, jitter_strength)

            # Length and thickness decrease with depth
            length[idx] = 0.35 * depth_factor * growth_stage
            length[idx] *= (0.8 + 0.4 * np.random.random())  # Random variation

            thickness[idx] = 0.08 * (depth_factor ** 1.5) * growth_stage

            # Pruning: randomly kill some branches
            if np.random.random() < prune_fraction * (level / depth):
                alive[idx] = 0.0
            else:
                # Propagate death from parent
                parent = (idx - 1) // 2
                if alive[parent] < 0.5:
                    alive[idx] = 0.0

    # Leaves at tips (only on alive branches)
    for idx in range(tip_start, num_segments):
        if alive[idx] > 0.5:
            leaf_area[idx] = growth_stage * (0.3 + 0.7 * np.random.random())

            # Flowers appear late in growth
            if growth_stage > 0.7:
                flower_prob = (growth_stage - 0.7) / 0.3
                if np.random.random() < flower_prob * 0.4:
                    flower_area[idx] = 0.2 * np.random.random()

    # Enforce bilateral symmetry if requested
    if symmetric:
        length = enforce_bilateral_symmetry(jnp.array(length), depth)
        thickness = enforce_bilateral_symmetry(jnp.array(thickness), depth)
        alive = enforce_bilateral_symmetry(jnp.array(alive), depth)
        # Mirror angles: left = +angle, right = -angle (already done by construction)
        leaf_area = enforce_bilateral_symmetry(jnp.array(leaf_area), depth)
        flower_area = enforce_bilateral_symmetry(jnp.array(flower_area), depth)
    else:
        length = jnp.array(length)
        thickness = jnp.array(thickness)
        alive = jnp.array(alive)
        leaf_area = jnp.array(leaf_area)
        flower_area = jnp.array(flower_area)

    return SkeletonState(
        length=length,
        thickness=thickness,
        alive=alive,
        angle=jnp.array(angle),
        leaf_area=leaf_area,
        flower_area=flower_area,
    )
