"""
Stained-glass style visualization for tree skeletons.

This module implements the "poster" aesthetic:
- Dense Poisson-sampled canopy leaves
- Lead-came style branch ribbons
- Colored background panels
- Coleus-style variegated leaves

The goal is to transform sparse skeleton data into visually rich
stained-glass tree art.
"""

from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull, Delaunay, Voronoi
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint
from shapely.ops import unary_union
import jax.numpy as jnp

from sim.skeleton import (
    SkeletonState,
    compute_segment_positions_2d,
    get_tip_indices,
    get_parent,
)


# =============================================================================
# POISSON DISK SAMPLING
# =============================================================================


def poisson_disk_sample(
    points: np.ndarray,
    num_samples: int = 150,
    min_distance: float = 0.08,
    seed: int = 42,
) -> np.ndarray:
    """
    Sample points inside a convex hull using Poisson disk sampling.

    This gives evenly-distributed points without clumping.

    Args:
        points: Nx2 array of hull boundary points
        num_samples: Target number of samples
        min_distance: Minimum distance between samples
        seed: Random seed

    Returns:
        Mx2 array of sampled points inside the hull
    """
    np.random.seed(seed)

    if len(points) < 3:
        return np.array([]).reshape(0, 2)

    # Compute convex hull
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
    except Exception:
        return points  # Fallback to original points

    # Bounding box
    min_x, min_y = hull_points.min(axis=0)
    max_x, max_y = hull_points.max(axis=0)

    # Create Delaunay triangulation for point-in-hull test
    try:
        delaunay = Delaunay(hull_points)
    except Exception:
        return points

    samples = []
    attempts = 0
    max_attempts = num_samples * 30

    while len(samples) < num_samples and attempts < max_attempts:
        # Random candidate point
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        candidate = np.array([x, y])

        # Check if inside hull
        if delaunay.find_simplex(candidate) < 0:
            attempts += 1
            continue

        # Check minimum distance to existing samples
        if len(samples) > 0:
            distances = np.linalg.norm(np.array(samples) - candidate, axis=1)
            if np.min(distances) < min_distance:
                attempts += 1
                continue

        samples.append(candidate)
        attempts += 1

    return np.array(samples) if samples else np.array([]).reshape(0, 2)


def compute_canopy_hull(
    skeleton: SkeletonState,
    padding: float = 0.15,
) -> np.ndarray:
    """
    Compute the canopy boundary from tip positions.

    Args:
        skeleton: Tree skeleton
        padding: Extra padding around hull

    Returns:
        Nx2 array of hull boundary points
    """
    x, y, _ = compute_segment_positions_2d(skeleton)
    x, y = np.array(x), np.array(y)

    tip_indices = get_tip_indices(skeleton.depth)

    # Collect alive tip positions
    tip_points = []
    for tip_idx in tip_indices:
        tip_idx = int(tip_idx)
        if float(skeleton.alive[tip_idx]) > 0.3:
            tip_points.append([x[tip_idx], y[tip_idx]])

    if len(tip_points) < 3:
        return np.array([]).reshape(0, 2)

    tip_points = np.array(tip_points)

    # Add padding by scaling outward from centroid
    centroid = tip_points.mean(axis=0)
    padded = centroid + (tip_points - centroid) * (1 + padding)

    return padded


# =============================================================================
# MICRO-SHOOTS (connecting leaves to skeleton)
# =============================================================================


def get_branch_segments(skeleton: SkeletonState) -> list[tuple]:
    """
    Get list of branch segments as (start_x, start_y, end_x, end_y, idx) tuples.

    Only includes alive segments with visible length.
    """
    x, y, _ = compute_segment_positions_2d(skeleton)
    x, y = np.array(x), np.array(y)

    segments = []
    for idx in range(skeleton.num_segments):
        alive = float(skeleton.alive[idx])
        if alive < 0.1:
            continue

        if idx == 0:
            start_x, start_y = 0.0, 0.0
        else:
            parent = get_parent(idx)
            start_x, start_y = x[parent], y[parent]

        end_x, end_y = x[idx], y[idx]

        # Skip tiny segments
        seg_len = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        if seg_len < 0.01:
            continue

        segments.append((start_x, start_y, end_x, end_y, idx))

    return segments


def closest_point_on_segment(px: float, py: float,
                              x1: float, y1: float,
                              x2: float, y2: float) -> tuple[float, float]:
    """
    Find closest point on line segment (x1,y1)-(x2,y2) to point (px,py).
    """
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx*dx + dy*dy

    if seg_len_sq < 1e-10:
        return x1, y1

    # Project point onto line, clamped to [0,1]
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))

    return x1 + t * dx, y1 + t * dy


def find_closest_branch_point(px: float, py: float,
                               segments: list[tuple]) -> tuple[float, float, int]:
    """
    Find the closest point on any branch segment to a given point.

    Returns (closest_x, closest_y, segment_idx)
    """
    best_dist = float('inf')
    best_point = (px, py)
    best_idx = 0

    for (x1, y1, x2, y2, idx) in segments:
        cx, cy = closest_point_on_segment(px, py, x1, y1, x2, y2)
        dist = np.sqrt((cx - px)**2 + (cy - py)**2)

        if dist < best_dist:
            best_dist = dist
            best_point = (cx, cy)
            best_idx = idx

    return best_point[0], best_point[1], best_idx


def draw_micro_shoot(ax, leaf_x: float, leaf_y: float,
                     branch_x: float, branch_y: float,
                     wood_color: str = '#5D4037',
                     lead_color: str = '#1a1a1a'):
    """
    Draw a thin twig connecting a leaf to the branch skeleton.

    This makes leaves look "attached" rather than floating.
    """
    # Lead outline (thin black)
    ax.plot(
        [branch_x, leaf_x], [branch_y, leaf_y],
        color=lead_color,
        linewidth=2.0,
        solid_capstyle='round',
        zorder=5,  # Above background, below leaves
    )

    # Wood fill (brown)
    ax.plot(
        [branch_x, leaf_x], [branch_y, leaf_y],
        color=wood_color,
        linewidth=1.2,
        solid_capstyle='round',
        zorder=6,
    )


# =============================================================================
# ROSETTE FLOWERS (not circles!)
# =============================================================================


def draw_rosette_flower(ax, x: float, y: float, size: float,
                        num_petals: int = 8,
                        inner_ratio: float = 0.4,
                        color: str = '#FF6B6B',
                        center_color: str = '#FFE66D',
                        edge_color: str = '#C0392B',
                        rotation: float = 0.0,
                        zorder: int = 8):
    """
    Draw a rosette/shuriken flower instead of a circle.

    Creates alternating vertices at outer/inner radii to form petals.

    Args:
        ax: Matplotlib axes
        x, y: Center position
        size: Outer radius
        num_petals: Number of petals (8-12 recommended)
        inner_ratio: Inner radius as fraction of outer (0.3-0.5)
        color: Petal fill color
        center_color: Center disk color
        edge_color: Edge/outline color
        rotation: Rotation in radians
        zorder: Drawing order
    """
    # Create alternating radii for star/rosette shape
    angles = np.linspace(0, 2*np.pi, num_petals * 2, endpoint=False) + rotation
    radii = np.array([size if i % 2 == 0 else size * inner_ratio
                      for i in range(num_petals * 2)])

    # Compute vertices
    verts_x = x + radii * np.cos(angles)
    verts_y = y + radii * np.sin(angles)
    verts = list(zip(verts_x, verts_y))

    # Draw petals
    rosette = Polygon(
        verts,
        facecolor=color,
        edgecolor=edge_color,
        linewidth=1.0,
        zorder=zorder,
        alpha=0.9,
    )
    ax.add_patch(rosette)

    # Draw center disk
    center = mpatches.Circle(
        (x, y),
        radius=size * inner_ratio * 0.6,
        facecolor=center_color,
        edgecolor=edge_color,
        linewidth=0.5,
        zorder=zorder + 1,
    )
    ax.add_patch(center)


# =============================================================================
# VORONOI CANOPY MOSAIC
# =============================================================================


class CellType:
    """Cell types for Voronoi mosaic."""
    LEAF = 'leaf'
    FLOWER = 'flower'
    BACKGROUND = 'background'


def compute_canopy_boundary(
    skeleton: SkeletonState,
    padding: float = 0.15,
    smooth: bool = True,
) -> ShapelyPolygon | None:
    """
    Compute the canopy boundary as a Shapely polygon.

    Creates an arch/dome shape based on alive tip positions.

    Args:
        skeleton: Tree skeleton
        padding: Extra padding around tips
        smooth: If True, use convex hull; if False, use alpha shape

    Returns:
        Shapely Polygon representing canopy boundary, or None if too few tips
    """
    x, y, _ = compute_segment_positions_2d(skeleton)
    x, y = np.array(x), np.array(y)

    tip_indices = get_tip_indices(skeleton.depth)

    # Collect alive tip positions
    tip_points = []
    for tip_idx in tip_indices:
        tip_idx = int(tip_idx)
        if float(skeleton.alive[tip_idx]) > 0.3:
            tip_points.append([x[tip_idx], y[tip_idx]])

    if len(tip_points) < 3:
        return None

    tip_points = np.array(tip_points)

    # Add trunk base point to close the shape at the bottom
    trunk_base = np.array([[0.0, 0.0]])
    all_points = np.vstack([tip_points, trunk_base])

    # Compute convex hull
    try:
        hull = ConvexHull(all_points)
        hull_points = all_points[hull.vertices]
    except Exception:
        return None

    # Pad outward from centroid
    centroid = hull_points.mean(axis=0)
    padded = centroid + (hull_points - centroid) * (1 + padding)

    # Create Shapely polygon
    try:
        boundary = ShapelyPolygon(padded)
        if not boundary.is_valid:
            boundary = boundary.buffer(0)  # Fix invalid geometries
        return boundary
    except Exception:
        return None


def compute_bounded_voronoi(
    seed_points: np.ndarray,
    boundary: ShapelyPolygon,
    extend_factor: float = 2.0,
) -> list[tuple[np.ndarray, int]]:
    """
    Compute Voronoi cells clipped to a boundary polygon.

    Args:
        seed_points: Nx2 array of seed points
        boundary: Shapely Polygon to clip cells to
        extend_factor: How far to extend infinite Voronoi edges

    Returns:
        List of (vertices, seed_index) tuples for each valid cell
    """
    if len(seed_points) < 3:
        return []

    # Add boundary points to help with edge cases
    bounds = boundary.bounds  # (minx, miny, maxx, maxy)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2

    # Add far-away dummy points to bound infinite regions
    far = max(width, height) * extend_factor
    dummy_points = np.array([
        [cx - far, cy - far],
        [cx + far, cy - far],
        [cx - far, cy + far],
        [cx + far, cy + far],
    ])
    all_points = np.vstack([seed_points, dummy_points])

    # Compute Voronoi
    try:
        vor = Voronoi(all_points)
    except Exception:
        return []

    cells = []
    for i, region_idx in enumerate(vor.point_region):
        if i >= len(seed_points):  # Skip dummy points
            continue

        region = vor.regions[region_idx]
        if -1 in region or len(region) < 3:  # Skip infinite or degenerate regions
            continue

        # Get vertices
        verts = vor.vertices[region]

        # Create Shapely polygon for this cell
        try:
            cell_poly = ShapelyPolygon(verts)
            if not cell_poly.is_valid:
                cell_poly = cell_poly.buffer(0)

            # Clip to boundary
            clipped = cell_poly.intersection(boundary)

            if clipped.is_empty or clipped.area < 1e-6:
                continue

            # Handle MultiPolygon (take largest piece)
            if clipped.geom_type == 'MultiPolygon':
                clipped = max(clipped.geoms, key=lambda g: g.area)

            if clipped.geom_type != 'Polygon':
                continue

            # Extract coordinates
            coords = np.array(clipped.exterior.coords)
            cells.append((coords, i))

        except Exception:
            continue

    return cells


def assign_cell_types(
    cells: list[tuple[np.ndarray, int]],
    seed_points: np.ndarray,
    skeleton: SkeletonState,
    flower_fraction: float = 0.08,
    background_fraction: float = 0.15,
    seed: int = 42,
) -> list[tuple[np.ndarray, str, int]]:
    """
    Assign types (leaf, flower, background) to Voronoi cells.

    Args:
        cells: List of (vertices, seed_index) tuples
        seed_points: Original seed points
        skeleton: Tree skeleton (for flower area info)
        flower_fraction: Fraction of cells that should be flowers
        background_fraction: Fraction of cells that should be background
        seed: Random seed

    Returns:
        List of (vertices, cell_type, color_index) tuples
    """
    np.random.seed(seed)

    # Get tip positions and flower areas for reference
    x, y, _ = compute_segment_positions_2d(skeleton)
    x, y = np.array(x), np.array(y)
    tip_indices = get_tip_indices(skeleton.depth)

    tip_positions = []
    tip_flower_areas = []
    for tip_idx in tip_indices:
        tip_idx = int(tip_idx)
        if float(skeleton.alive[tip_idx]) > 0.3:
            tip_positions.append([x[tip_idx], y[tip_idx]])
            tip_flower_areas.append(float(skeleton.flower_area[tip_idx]))

    tip_positions = np.array(tip_positions) if tip_positions else np.array([]).reshape(0, 2)
    tip_flower_areas = np.array(tip_flower_areas) if tip_flower_areas else np.array([])

    # Assign types
    n_cells = len(cells)
    n_flowers = max(1, int(n_cells * flower_fraction))
    n_background = max(1, int(n_cells * background_fraction))

    # Shuffle indices for random assignment
    indices = np.arange(n_cells)
    np.random.shuffle(indices)

    # Cells closer to high-flower tips are more likely to be flowers
    flower_scores = np.zeros(n_cells)
    if len(tip_positions) > 0 and len(tip_flower_areas) > 0:
        for i, (verts, seed_idx) in enumerate(cells):
            centroid = verts.mean(axis=0)
            # Find closest tip
            dists = np.linalg.norm(tip_positions - centroid, axis=1)
            closest_idx = np.argmin(dists)
            # Score based on flower area of closest tip
            flower_scores[i] = tip_flower_areas[closest_idx] / (dists[closest_idx] + 0.1)

    # Sort by flower score for flower assignment
    flower_priority = np.argsort(-flower_scores)

    typed_cells = []
    flower_set = set(flower_priority[:n_flowers])
    background_set = set(indices[:n_background]) - flower_set

    for i, (verts, seed_idx) in enumerate(cells):
        if i in flower_set:
            cell_type = CellType.FLOWER
        elif i in background_set:
            cell_type = CellType.BACKGROUND
        else:
            cell_type = CellType.LEAF

        color_idx = np.random.randint(0, 100)
        typed_cells.append((verts, cell_type, color_idx))

    return typed_cells


def draw_voronoi_canopy(
    ax,
    typed_cells: list[tuple[np.ndarray, str, int]],
    leaf_colors: list[str] = None,
    flower_colors: list[str] = None,
    background_colors: list[str] = None,
    lead_color: str = '#1a1a1a',
    lead_width: float = 2.0,
    draw_veins: bool = True,
):
    """
    Draw Voronoi cells as stained-glass panes.

    Each cell is a colored polygon with thick lead-came outline.

    Args:
        ax: Matplotlib axes
        typed_cells: List of (vertices, cell_type, color_index) tuples
        leaf_colors: Colors for leaf cells
        flower_colors: Colors for flower cells
        background_colors: Colors for background cells
        lead_color: Color for cell outlines (lead came)
        lead_width: Width of outlines
        draw_veins: Draw center veins on leaf cells
    """
    if leaf_colors is None:
        leaf_colors = ['#C0392B', '#E74C3C', '#27AE60', '#2ECC71',
                       '#F39C12', '#E67E22', '#D35400', '#16A085']
    if flower_colors is None:
        flower_colors = ['#FF6B6B', '#FF8E8E', '#E74C3C', '#FF7675', '#D63031']
    if background_colors is None:
        background_colors = ['#FFF8DC', '#FFEFD5', '#FFE4B5', '#F5DEB3']

    for verts, cell_type, color_idx in typed_cells:
        # Select color based on type
        if cell_type == CellType.LEAF:
            color = leaf_colors[color_idx % len(leaf_colors)]
            zorder = 10
            alpha = 0.9
        elif cell_type == CellType.FLOWER:
            color = flower_colors[color_idx % len(flower_colors)]
            zorder = 9  # Slightly below leaves
            alpha = 0.95
        else:  # BACKGROUND
            color = background_colors[color_idx % len(background_colors)]
            zorder = 8
            alpha = 0.7

        # Draw cell as polygon
        cell_patch = Polygon(
            verts,
            facecolor=color,
            edgecolor=lead_color,
            linewidth=lead_width,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(cell_patch)

        # Add vein line for leaf cells
        if draw_veins and cell_type == CellType.LEAF:
            centroid = verts.mean(axis=0)
            # Find longest axis of cell for vein direction
            dists = np.linalg.norm(verts - centroid, axis=1)
            max_idx = np.argmax(dists)
            direction = verts[max_idx] - centroid
            direction = direction / (np.linalg.norm(direction) + 1e-6)

            # Draw short vein line
            vein_len = dists[max_idx] * 0.5
            ax.plot(
                [centroid[0] - direction[0] * vein_len * 0.3,
                 centroid[0] + direction[0] * vein_len * 0.5],
                [centroid[1] - direction[1] * vein_len * 0.3,
                 centroid[1] + direction[1] * vein_len * 0.5],
                color='#1a1a1a',
                linewidth=0.8,
                alpha=0.3,
                zorder=zorder + 1,
            )

        # Add center detail for flower cells
        if cell_type == CellType.FLOWER:
            centroid = verts.mean(axis=0)
            # Small yellow center
            center = mpatches.Circle(
                centroid,
                radius=0.015,
                facecolor='#FFE66D',
                edgecolor='#D35400',
                linewidth=0.5,
                zorder=zorder + 1,
            )
            ax.add_patch(center)


def render_stained_glass_voronoi(
    skeleton: SkeletonState,
    ax=None,
    figsize: tuple = (12, 14),
    cell_density: int = 120,
    min_cell_distance: float = 0.06,
    flower_fraction: float = 0.08,
    background_fraction: float = 0.12,
    leaf_palette: list[str] = None,
    show_background_panels: bool = True,
    show_ground: bool = True,
    title: str = "",
    seed: int = 42,
):
    """
    Render tree with Voronoi-filled canopy hull.

    WARNING: This fills the entire convex hull with cells, hiding the tree
    structure. Use render_stained_glass_natural() for a tree that looks
    like a tree.

    Args:
        skeleton: SkeletonState to render
        ax: Matplotlib axes (creates new if None)
        figsize: Figure size if creating new
        cell_density: Target number of cells in canopy
        min_cell_distance: Minimum distance between cell seeds
        flower_fraction: Fraction of cells that are flowers
        background_fraction: Fraction of cells that are background (gaps)
        leaf_palette: Colors for leaf cells
        show_background_panels: Draw radial background panels
        show_ground: Draw ground segments
        title: Plot title
        seed: Random seed

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    np.random.seed(seed)
    ax.set_facecolor('#FFF8DC')

    # Get positions
    x, y, _ = compute_segment_positions_2d(skeleton)
    x, y = np.array(x), np.array(y)

    # Set axis limits
    x_margin = 0.5
    y_margin = 0.3
    x_min, x_max = x.min() - x_margin, x.max() + x_margin
    y_max = y.max() + y_margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, y_max)

    # 1. Background panels
    if show_background_panels:
        draw_background_panels(ax, style='radial', num_panels=12)

    # 2. Ground
    if show_ground:
        draw_ground(ax)

    # 3. Branches (lead-came style)
    draw_branches_lead_came(ax, skeleton)

    # 4. Voronoi canopy
    boundary = compute_canopy_boundary(skeleton, padding=0.2)

    if boundary is not None:
        # Sample seed points inside boundary
        hull_points = np.array(boundary.exterior.coords)
        seed_points = poisson_disk_sample(
            hull_points,
            num_samples=cell_density,
            min_distance=min_cell_distance,
            seed=seed,
        )

        if len(seed_points) >= 3:
            # Compute Voronoi cells
            cells = compute_bounded_voronoi(seed_points, boundary)

            # Assign types
            typed_cells = assign_cell_types(
                cells, seed_points, skeleton,
                flower_fraction=flower_fraction,
                background_fraction=background_fraction,
                seed=seed,
            )

            # Draw canopy
            draw_voronoi_canopy(
                ax, typed_cells,
                leaf_colors=leaf_palette,
                draw_veins=True,
            )

    ax.set_aspect('equal')
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    return ax


# =============================================================================
# NATURAL TREE RENDERING (visible structure + tip leaves)
# =============================================================================


def generate_leaf_silhouette(
    tip_x: float,
    tip_y: float,
    length: float,
    width: float,
    angle: float,
    num_points: int = 12,
    asymmetry: float = 0.1,
    tip_sharpness: float = 0.3,
    base_width: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate a natural leaf silhouette (pointed tip, rounded middle, narrower base).

    The leaf shape is created using a parametric curve that mimics real leaf forms:
    - Pointed tip at the "front"
    - Widest in the upper-middle section
    - Narrower at the base (petiole attachment)

    Args:
        tip_x, tip_y: Position of leaf TIP (not center)
        length: Length from base to tip
        width: Maximum width of leaf
        angle: Direction the leaf points (tip direction) in radians
        num_points: Number of vertices per side
        asymmetry: Random left/right variation (0-0.3)
        tip_sharpness: How pointed the tip is (0.1-0.5, lower = sharper)
        base_width: Width at base relative to max width (0.2-0.5)
        seed: Random seed

    Returns:
        Nx2 array of polygon vertices forming leaf outline
    """
    np.random.seed(seed)

    # Parameter t goes from 0 (base) to 1 (tip)
    t = np.linspace(0, 1, num_points)

    # Width profile: starts narrow, widens, then narrows to tip
    # Use a modified sine curve that peaks around t=0.4-0.5
    # w(t) = sin(pi * t^0.7) gives a nice leaf shape
    width_profile = np.sin(np.pi * (t ** 0.7))

    # Modify base to be narrower
    base_taper = base_width + (1 - base_width) * t ** 0.5
    width_profile = width_profile * base_taper

    # Sharpen the tip
    tip_taper = 1 - (1 - tip_sharpness) * (t ** 2)
    width_profile = width_profile * tip_taper

    # Normalize so max is 1
    width_profile = width_profile / (width_profile.max() + 1e-6)

    # Add asymmetry (slight random variation)
    left_variation = 1 + np.random.uniform(-asymmetry, asymmetry, num_points)
    right_variation = 1 + np.random.uniform(-asymmetry, asymmetry, num_points)

    # Build left and right edges
    half_widths_left = width * 0.5 * width_profile * left_variation
    half_widths_right = width * 0.5 * width_profile * right_variation

    # Position along leaf axis (base to tip)
    along_axis = t * length

    # Convert to coordinates (leaf points in +angle direction)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # Perpendicular direction
    cos_perp, sin_perp = np.cos(angle + np.pi/2), np.sin(angle + np.pi/2)

    # Base position (tip minus length in angle direction)
    base_x = tip_x - length * cos_a
    base_y = tip_y - length * sin_a

    # Left edge (going from base to tip)
    left_x = base_x + along_axis * cos_a + half_widths_left * cos_perp
    left_y = base_y + along_axis * sin_a + half_widths_left * sin_perp

    # Right edge (going from tip back to base, for closed polygon)
    right_x = base_x + along_axis * cos_a - half_widths_right * cos_perp
    right_y = base_y + along_axis * sin_a - half_widths_right * sin_perp

    # Combine: left edge (base to tip) + right edge (tip to base)
    all_x = np.concatenate([left_x, right_x[::-1]])
    all_y = np.concatenate([left_y, right_y[::-1]])

    return np.column_stack([all_x, all_y])


def generate_leaf_polygon(
    center_x: float,
    center_y: float,
    size: float,
    angle: float,
    num_vertices: int = 7,
    irregularity: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate an irregular polygon leaf shape (LEGACY - use generate_leaf_silhouette).

    Creates organic-looking leaf shapes that aren't perfect ellipses.
    """
    np.random.seed(seed)
    base_angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
    angle_noise = np.random.uniform(-irregularity, irregularity, num_vertices)
    angles = base_angles + angle_noise

    radii = []
    for a in angles:
        rel_angle = a - angle
        r = size * (0.4 + 0.6 * np.abs(np.sin(rel_angle)))
        r *= (1 + np.random.uniform(-irregularity * 0.5, irregularity * 0.5))
        radii.append(r)

    radii = np.array(radii)
    verts_x = center_x + radii * np.cos(angles)
    verts_y = center_y + radii * np.sin(angles)

    return np.column_stack([verts_x, verts_y])


def draw_natural_leaf(
    ax,
    vertices: np.ndarray,
    fill_color: str,
    base_point: tuple[float, float],
    tip_point: tuple[float, float],
    edge_color: str = '#1a1a1a',
    edge_width: float = 1.5,
    draw_vein: bool = True,
    draw_variegation: bool = True,
    vein_color: str = '#1a1a1a',
    vein_alpha: float = 0.5,
    zorder: int = 10,
    seed: int = 0,
):
    """
    Draw a natural leaf with proper vein and optional variegation.

    Args:
        ax: Matplotlib axes
        vertices: Nx2 array of polygon vertices
        fill_color: Main leaf fill color
        base_point: (x, y) of leaf base (petiole attachment)
        tip_point: (x, y) of leaf tip
        edge_color: Edge/outline color (lead came)
        edge_width: Outline width
        draw_vein: Whether to draw center vein
        draw_variegation: Whether to draw inner color stripe
        vein_color: Vein line color
        vein_alpha: Vein transparency
        zorder: Drawing order
        seed: Random seed for variegation color
    """
    np.random.seed(seed)

    # Draw main leaf polygon
    leaf = Polygon(
        vertices,
        facecolor=fill_color,
        edgecolor=edge_color,
        linewidth=edge_width,
        zorder=zorder,
    )
    ax.add_patch(leaf)

    # Draw variegation (inner stripe) - coleus style
    if draw_variegation:
        # Inner colors for variegation
        inner_colors = ['#2ECC71', '#F1C40F', '#E74C3C', '#9B59B6',
                        '#3498DB', '#1ABC9C', '#E67E22']
        inner_color = inner_colors[seed % len(inner_colors)]

        # Create smaller inner leaf shape
        centroid = vertices.mean(axis=0)
        inner_verts = centroid + 0.4 * (vertices - centroid)

        inner_leaf = Polygon(
            inner_verts,
            facecolor=inner_color,
            edgecolor='none',
            alpha=0.6,
            zorder=zorder + 0.5,
        )
        ax.add_patch(inner_leaf)

    # Draw center vein (from base toward tip)
    if draw_vein:
        base_x, base_y = base_point
        tip_x, tip_y = tip_point

        # Vein goes from 20% from base to 85% toward tip
        vein_start_x = base_x + 0.2 * (tip_x - base_x)
        vein_start_y = base_y + 0.2 * (tip_y - base_y)
        vein_end_x = base_x + 0.85 * (tip_x - base_x)
        vein_end_y = base_y + 0.85 * (tip_y - base_y)

        ax.plot(
            [vein_start_x, vein_end_x],
            [vein_start_y, vein_end_y],
            color=vein_color,
            linewidth=1.0,
            alpha=vein_alpha,
            zorder=zorder + 1,
        )


def draw_polygon_leaf(
    ax,
    vertices: np.ndarray,
    fill_color: str,
    edge_color: str = '#1a1a1a',
    edge_width: float = 1.5,
    draw_vein: bool = True,
    vein_color: str = '#1a1a1a',
    vein_alpha: float = 0.4,
    zorder: int = 10,
):
    """
    Draw a polygon-shaped leaf (LEGACY - use draw_natural_leaf for silhouettes).
    """
    leaf = Polygon(
        vertices,
        facecolor=fill_color,
        edgecolor=edge_color,
        linewidth=edge_width,
        zorder=zorder,
    )
    ax.add_patch(leaf)

    if draw_vein:
        centroid = vertices.mean(axis=0)
        dists = np.linalg.norm(vertices - centroid, axis=1)
        sorted_idx = np.argsort(dists)
        far1 = vertices[sorted_idx[-1]]

        direction = far1 - centroid
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        vein_len = dists[sorted_idx[-1]] * 0.6

        ax.plot(
            [centroid[0] - direction[0] * vein_len * 0.2,
             centroid[0] + direction[0] * vein_len * 0.7],
            [centroid[1] - direction[1] * vein_len * 0.2,
             centroid[1] + direction[1] * vein_len * 0.7],
            color=vein_color,
            linewidth=0.8,
            alpha=vein_alpha,
            zorder=zorder + 1,
        )


def render_stained_glass_natural(
    skeleton: SkeletonState,
    ax=None,
    figsize: tuple = (12, 14),
    leaf_size: float = 0.12,
    leaf_colors: list[str] = None,
    flower_colors: list[str] = None,
    show_background: bool = True,
    show_ground: bool = True,
    title: str = "",
    seed: int = 42,
):
    """
    Render tree with visible branch structure and polygon leaves at tips.

    This approach maintains the tree's form:
    - Branches are clearly visible as the skeleton
    - Leaves are polygon shapes attached at tips
    - Flowers are small rosettes near tips
    - You can see through the canopy

    This looks like the inspirational stained-glass tree image.

    Args:
        skeleton: SkeletonState to render
        ax: Matplotlib axes (creates new if None)
        figsize: Figure size if creating new
        leaf_size: Base size for leaves
        leaf_colors: Color palette for leaves
        flower_colors: Color palette for flowers
        show_background: Draw background panels
        show_ground: Draw ground segments
        title: Plot title
        seed: Random seed

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    np.random.seed(seed)
    ax.set_facecolor('#FFF8DC')

    if leaf_colors is None:
        leaf_colors = ['#C0392B', '#E74C3C', '#27AE60', '#2ECC71',
                       '#F39C12', '#E67E22', '#D35400', '#16A085',
                       '#1ABC9C', '#9B59B6']
    if flower_colors is None:
        flower_colors = ['#FF6B6B', '#FF8E8E', '#E74C3C', '#FF7675']

    # Get positions
    x, y, _ = compute_segment_positions_2d(skeleton)
    x, y = np.array(x), np.array(y)

    # Set axis limits
    x_margin = 0.5
    y_margin = 0.3
    x_min, x_max = x.min() - x_margin, x.max() + x_margin
    y_max = y.max() + y_margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, y_max)

    # 1. Background panels
    if show_background:
        draw_background_panels(ax, style='radial', num_panels=12)

    # 2. Ground
    if show_ground:
        draw_ground(ax)

    # 3. Branches (lead-came style) - drawn FIRST so leaves appear on top
    draw_branches_lead_came(ax, skeleton)

    # 4. Leaves at tips - proper leaf silhouettes
    tip_indices = get_tip_indices(skeleton.depth)

    for i, tip_idx in enumerate(tip_indices):
        tip_idx = int(tip_idx)
        alive = float(skeleton.alive[tip_idx])
        leaf_area = float(skeleton.leaf_area[tip_idx])

        if alive < 0.3 or leaf_area < 0.05:
            continue

        # Get branch tip position
        branch_tip_x, branch_tip_y = x[tip_idx], y[tip_idx]

        # Get parent position for leaf direction
        parent_idx = (tip_idx - 1) // 2 if tip_idx > 0 else 0
        parent_x, parent_y = x[parent_idx], y[parent_idx]

        # Leaf points OUTWARD from branch (away from parent)
        branch_dir_x = branch_tip_x - parent_x
        branch_dir_y = branch_tip_y - parent_y
        branch_len = np.sqrt(branch_dir_x**2 + branch_dir_y**2) + 1e-6

        # Normalize branch direction
        dir_x = branch_dir_x / branch_len
        dir_y = branch_dir_y / branch_len

        # Leaf angle follows branch direction
        leaf_angle = np.arctan2(dir_y, dir_x)

        # Leaf dimensions based on leaf_area
        length = leaf_size * (0.8 + 0.6 * np.sqrt(leaf_area))
        width = length * (0.4 + 0.2 * np.random.random())  # Aspect ratio 2:1 to 3:1

        # Leaf tip position (offset from branch tip in leaf direction)
        leaf_tip_x = branch_tip_x + length * 0.6 * dir_x
        leaf_tip_y = branch_tip_y + length * 0.6 * dir_y

        # Generate leaf silhouette
        vertices = generate_leaf_silhouette(
            tip_x=leaf_tip_x,
            tip_y=leaf_tip_y,
            length=length,
            width=width,
            angle=leaf_angle,
            num_points=10,
            asymmetry=0.12,
            tip_sharpness=0.25,
            base_width=0.25,
            seed=i + seed,
        )

        # Compute base point for vein drawing
        base_x = leaf_tip_x - length * np.cos(leaf_angle)
        base_y = leaf_tip_y - length * np.sin(leaf_angle)

        # Pick color
        color = leaf_colors[i % len(leaf_colors)]

        draw_natural_leaf(
            ax, vertices,
            fill_color=color,
            base_point=(base_x, base_y),
            tip_point=(leaf_tip_x, leaf_tip_y),
            draw_vein=True,
            draw_variegation=True,
            zorder=10,
            seed=i + seed,
        )

    # 5. Flowers (small rosettes, sparse)
    flower_candidates = []
    for i, tip_idx in enumerate(tip_indices):
        tip_idx = int(tip_idx)
        flower_area = float(skeleton.flower_area[tip_idx])
        alive = float(skeleton.alive[tip_idx])

        if flower_area > 0.05 and alive > 0.3:
            flower_candidates.append((i, tip_idx, flower_area))

    # Take top flowers by area
    flower_candidates.sort(key=lambda f: f[2], reverse=True)
    max_flowers = min(len(flower_candidates), max(3, len(tip_indices) // 8))
    flower_candidates = flower_candidates[:max_flowers]

    for i, tip_idx, flower_area in flower_candidates:
        tip_x, tip_y = x[tip_idx], y[tip_idx]

        # Offset slightly from leaf
        offset = leaf_size * 0.3
        flower_x = tip_x + offset * np.cos(i * 1.3)
        flower_y = tip_y + offset * np.sin(i * 1.3) + offset * 0.5

        flower_size = 0.03 + 0.02 * np.sqrt(flower_area)

        draw_rosette_flower(
            ax, flower_x, flower_y,
            size=min(flower_size, 0.045),
            num_petals=np.random.choice([6, 8]),
            color=flower_colors[i % len(flower_colors)],
            zorder=11,
        )

    ax.set_aspect('equal')
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    return ax


# =============================================================================
# LEAF DRAWING
# =============================================================================


class LeafStyle(NamedTuple):
    """Style parameters for leaf rendering."""
    base_colors: list  # List of possible fill colors
    edge_color: str = '#1a1a1a'
    edge_width: float = 1.5
    vein_color: str = '#1a1a1a'
    vein_alpha: float = 0.4
    variegated: bool = True  # Coleus-style inner stripe


# Predefined palettes
AUTUMN_PALETTE = LeafStyle(
    base_colors=['#C0392B', '#E74C3C', '#27AE60', '#F39C12', '#E67E22', '#D35400'],
)

SPRING_PALETTE = LeafStyle(
    base_colors=['#27AE60', '#2ECC71', '#1ABC9C', '#16A085', '#82E0AA'],
)

SUNSET_PALETTE = LeafStyle(
    base_colors=['#E74C3C', '#C0392B', '#F39C12', '#D35400', '#922B21'],
)


def draw_leaf(
    ax,
    x: float,
    y: float,
    size: float,
    angle: float,
    style: LeafStyle,
    seed: int = 0,
):
    """
    Draw a single stylized leaf.

    Args:
        ax: Matplotlib axes
        x, y: Position
        size: Leaf size
        angle: Orientation in degrees
        style: LeafStyle parameters
        seed: For color selection
    """
    np.random.seed(seed)

    # Pick color from palette
    color = style.base_colors[seed % len(style.base_colors)]

    # Main leaf ellipse
    leaf = mpatches.Ellipse(
        (x, y),
        width=size * 0.45,
        height=size,
        angle=angle,
        facecolor=color,
        edgecolor=style.edge_color,
        linewidth=style.edge_width,
        zorder=10,
    )
    ax.add_patch(leaf)

    # Variegated inner stripe (Coleus effect)
    if style.variegated:
        # Slightly different color for inner stripe
        inner_colors = ['#2ECC71', '#F1C40F', '#E74C3C', '#9B59B6', '#3498DB']
        inner_color = inner_colors[seed % len(inner_colors)]

        inner = mpatches.Ellipse(
            (x, y),
            width=size * 0.15,
            height=size * 0.6,
            angle=angle,
            facecolor=inner_color,
            edgecolor='none',
            alpha=0.7,
            zorder=11,
        )
        ax.add_patch(inner)

    # Center vein
    vein_len = size * 0.4
    rad = np.radians(angle)
    dx = vein_len * np.sin(rad)
    dy = vein_len * np.cos(rad)
    ax.plot(
        [x - dx, x + dx],
        [y - dy, y + dy],
        color=style.vein_color,
        linewidth=0.8,
        alpha=style.vein_alpha,
        zorder=12,
    )


# =============================================================================
# BRANCH DRAWING (LEAD CAME STYLE)
# =============================================================================


def draw_branches_lead_came(
    ax,
    skeleton: SkeletonState,
    wood_color: str = '#8B4513',
    lead_color: str = '#1a1a1a',
    lead_width: float = 2.0,
):
    """
    Draw branches as lead-came ribbons (thick outline, colored fill).

    This gives the stained-glass "leaded" look.
    """
    x, y, _ = compute_segment_positions_2d(skeleton)
    x, y = np.array(x), np.array(y)

    num_segments = skeleton.num_segments

    # Draw each branch segment
    for idx in range(num_segments):
        alive = float(skeleton.alive[idx])
        if alive < 0.1:
            continue

        # Get start position
        if idx == 0:
            start_x, start_y = 0.0, 0.0
        else:
            parent = (idx - 1) // 2
            start_x, start_y = x[parent], y[parent]

        end_x, end_y = x[idx], y[idx]

        # Skip tiny segments
        seg_len = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        if seg_len < 0.01:
            continue

        # Line width based on thickness
        thickness = float(skeleton.thickness[idx])
        base_width = 1.5 + 10 * thickness * alive

        # Lead outline (black, wider)
        ax.plot(
            [start_x, end_x], [start_y, end_y],
            color=lead_color,
            linewidth=base_width + lead_width,
            solid_capstyle='round',
            zorder=3,
        )

        # Wood fill (brown, narrower)
        ax.plot(
            [start_x, end_x], [start_y, end_y],
            color=wood_color,
            linewidth=base_width,
            solid_capstyle='round',
            zorder=4,
        )


# =============================================================================
# BACKGROUND PANELS
# =============================================================================


def draw_background_panels(
    ax,
    panel_colors: list = None,
    style: str = 'radial',
    num_panels: int = 8,
):
    """
    Draw stained-glass background panels.

    Args:
        ax: Matplotlib axes
        panel_colors: List of colors for panels
        style: 'radial' or 'grid'
        num_panels: Number of panels
    """
    if panel_colors is None:
        panel_colors = ['#FFF8DC', '#FFE4B5', '#FFEFD5', '#F5DEB3', '#DEB887']

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    cx, cy = (xlim[0] + xlim[1]) / 2, 0  # Center at ground level

    if style == 'radial':
        # Radial divisions from center
        for i in range(num_panels):
            angle1 = i * np.pi / num_panels
            angle2 = (i + 1) * np.pi / num_panels

            # Create wedge
            r = 5  # Large enough to cover canvas
            verts = [
                (cx, cy),
                (cx + r * np.cos(angle1), cy + r * np.sin(angle1)),
                (cx + r * np.cos(angle2), cy + r * np.sin(angle2)),
            ]

            color = panel_colors[i % len(panel_colors)]
            wedge = Polygon(verts, facecolor=color, edgecolor='#1a1a1a',
                           linewidth=0.5, alpha=0.3, zorder=0)
            ax.add_patch(wedge)


def draw_ground(
    ax,
    colors: list = None,
    y_level: float = 0.0,
    depth: float = 0.5,
):
    """Draw stylized ground panels."""
    if colors is None:
        colors = ['#8BC34A', '#4A90A4', '#E07B54']

    xlim = ax.get_xlim()
    width = (xlim[1] - xlim[0]) / len(colors)

    for i, color in enumerate(colors):
        x_start = xlim[0] + i * width
        rect = mpatches.Rectangle(
            (x_start, y_level - depth),
            width, depth,
            facecolor=color,
            edgecolor='#1a1a1a',
            linewidth=1.5,
            alpha=0.7,
            zorder=1,
        )
        ax.add_patch(rect)


# =============================================================================
# MAIN RENDERING FUNCTION
# =============================================================================


def render_stained_glass_tree(
    skeleton: SkeletonState,
    ax=None,
    figsize: tuple = (12, 14),
    leaf_density: int = 150,
    leaf_size: float = 0.12,
    palette: LeafStyle = AUTUMN_PALETTE,
    show_background: bool = True,
    show_ground: bool = True,
    show_micro_shoots: bool = True,
    max_flower_ratio: float = 0.10,  # Max flowers as fraction of leaves
    flower_size_cap: float = 0.05,   # Maximum flower radius
    title: str = "",
    seed: int = 42,
):
    """
    Render tree skeleton in full stained-glass poster style.

    This is the main visualization function that combines:
    - Lead-came branch rendering
    - Poisson-sampled canopy leaves WITH micro-shoot attachments
    - Rosette-style flowers (not circles)
    - Stylized background panels
    - Ground segments

    Key aesthetic rules enforced:
    - Leaves are attached to skeleton via micro-shoots
    - Flowers are rosettes, not circles
    - Flowers are tucked INSIDE canopy (lower z-order than leaves)
    - Flower count is capped (5-10% of leaf count)

    Args:
        skeleton: SkeletonState to render
        ax: Matplotlib axes (creates new if None)
        figsize: Figure size if creating new
        leaf_density: Number of leaves to sample in canopy
        leaf_size: Base leaf size
        palette: LeafStyle for coloring
        show_background: Draw background panels
        show_ground: Draw ground segments
        show_micro_shoots: Draw twigs connecting leaves to branches
        max_flower_ratio: Maximum flowers as fraction of leaf count
        flower_size_cap: Maximum flower radius
        title: Plot title
        seed: Random seed for reproducibility

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    np.random.seed(seed)
    ax.set_facecolor('#FFF8DC')  # Warm cream background

    # Get tip positions for canopy
    x, y, _ = compute_segment_positions_2d(skeleton)
    x, y = np.array(x), np.array(y)

    # Pre-compute branch segments for micro-shoot attachment
    branch_segments = get_branch_segments(skeleton)

    # Set axis limits based on tree extent
    x_margin = 0.5
    y_margin = 0.3
    x_min, x_max = x.min() - x_margin, x.max() + x_margin
    y_max = y.max() + y_margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, y_max)

    # 1. Background panels (lowest z-order)
    if show_background:
        draw_background_panels(ax, style='radial', num_panels=12)

    # 2. Ground
    if show_ground:
        draw_ground(ax)

    # 3. Branches (lead-came style)
    draw_branches_lead_came(ax, skeleton)

    # 4. Canopy leaves (Poisson sampled) WITH micro-shoots
    hull_points = compute_canopy_hull(skeleton, padding=0.2)
    leaf_count = 0

    if len(hull_points) >= 3:
        leaf_points = poisson_disk_sample(
            hull_points,
            num_samples=leaf_density,
            min_distance=leaf_size * 0.8,
            seed=seed,
        )
        leaf_count = len(leaf_points)

        # Draw micro-shoots FIRST (behind leaves)
        if show_micro_shoots and len(branch_segments) > 0:
            for lx, ly in leaf_points:
                bx, by, _ = find_closest_branch_point(lx, ly, branch_segments)
                draw_micro_shoot(ax, lx, ly, bx, by)

        # Draw leaves (on top of micro-shoots)
        for i, (lx, ly) in enumerate(leaf_points):
            # Angle pointing outward from trunk center
            angle = np.degrees(np.arctan2(ly, lx + 0.001)) - 90

            # Size variation
            size = leaf_size * (0.7 + 0.6 * np.random.random())

            draw_leaf(ax, lx, ly, size, angle, palette, seed=i + seed)

    # 5. Tip leaves (at actual branch tips, slightly larger)
    tip_indices = get_tip_indices(skeleton.depth)
    for i, tip_idx in enumerate(tip_indices):
        tip_idx = int(tip_idx)
        alive = float(skeleton.alive[tip_idx])
        leaf_area = float(skeleton.leaf_area[tip_idx])

        if alive < 0.3 or leaf_area < 0.1:
            continue

        tip_x, tip_y = x[tip_idx], y[tip_idx]
        angle = np.degrees(np.arctan2(tip_y, tip_x + 0.001)) - 90
        size = leaf_size * 1.2 * np.sqrt(leaf_area)

        draw_leaf(ax, tip_x, tip_y, size, angle, palette, seed=i + 1000 + seed)

    # 6. Flowers (rosettes, TUCKED INSIDE canopy, count-capped)
    # Collect flower candidates
    flower_candidates = []
    for i, tip_idx in enumerate(tip_indices):
        tip_idx = int(tip_idx)
        flower_area = float(skeleton.flower_area[tip_idx])
        alive = float(skeleton.alive[tip_idx])

        if flower_area < 0.05 or alive < 0.3:
            continue

        tip_x, tip_y = x[tip_idx], y[tip_idx]
        flower_candidates.append((tip_x, tip_y, flower_area, i))

    # Cap flower count to max_flower_ratio * leaf_count
    max_flowers = max(2, int(leaf_count * max_flower_ratio))
    # Sort by flower_area (descending) and take top N
    flower_candidates.sort(key=lambda f: f[2], reverse=True)
    flower_candidates = flower_candidates[:max_flowers]

    # Draw flowers (z-order 8-9, BELOW leaves which are at 10-12)
    flower_colors = ['#FF6B6B', '#FF8E8E', '#E74C3C', '#FF7675', '#D63031']
    for tip_x, tip_y, flower_area, i in flower_candidates:
        # Small offset to tuck into canopy gap
        offset = 0.02
        flower_x = tip_x + offset * np.cos(i * 0.7)
        flower_y = tip_y + offset * np.sin(i * 0.7)

        # Size: capped and scaled down
        raw_size = 0.03 + 0.04 * np.sqrt(flower_area)
        flower_size = min(raw_size, flower_size_cap)

        draw_rosette_flower(
            ax, flower_x, flower_y,
            size=flower_size,
            num_petals=np.random.choice([6, 8, 10]),
            inner_ratio=0.35 + 0.1 * np.random.random(),
            color=flower_colors[i % len(flower_colors)],
            center_color='#FFE66D',
            edge_color='#922B21',
            rotation=np.random.random() * np.pi / 4,
            zorder=8,  # BELOW leaves (10-12)
        )

    ax.set_aspect('equal')
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    return ax


def render_comparison(
    skeletons: list,
    titles: list,
    figsize: tuple = (18, 8),
    **kwargs,
):
    """
    Render multiple skeletons side by side for comparison.

    Args:
        skeletons: List of SkeletonState objects
        titles: List of titles for each
        figsize: Overall figure size
        **kwargs: Passed to render_stained_glass_tree

    Returns:
        Figure and axes
    """
    n = len(skeletons)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for ax, skeleton, title in zip(axes, skeletons, titles):
        render_stained_glass_tree(skeleton, ax=ax, title=title, **kwargs)

    plt.tight_layout()
    return fig, axes
