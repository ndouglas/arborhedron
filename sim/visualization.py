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
from scipy.spatial import ConvexHull, Delaunay
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
