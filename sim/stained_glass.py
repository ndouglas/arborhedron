"""
Stained-glass tree renderer with L-system branching and leaf panels.

Combines:
- L-system recursive tree generation
- Equal-area vein panel leaf geometry
- Optional fruits and blossoms
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle, Ellipse
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import math


# =============================================================================
# VECTOR UTILITIES
# =============================================================================

def vec(x: float, y: float) -> np.ndarray:
    """Create a 2D vector."""
    return np.array([x, y], dtype=float)


def vec_lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two vectors."""
    return a + t * (b - a)


def vec_mag(v: np.ndarray) -> float:
    """Magnitude of a vector."""
    return np.sqrt(v[0]**2 + v[1]**2)


def vec_normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    m = vec_mag(v)
    if m < 1e-9:
        return vec(0, 0)
    return v / m


def vec_rotate(v: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector by angle (radians)."""
    c, s = math.cos(angle), math.sin(angle)
    return vec(v[0] * c - v[1] * s, v[0] * s + v[1] * c)


# =============================================================================
# NOISE FUNCTION
# =============================================================================

def simple_noise(seed: float, t: float) -> float:
    """
    Simple deterministic noise function.
    Returns value in [0, 1].
    """
    x = seed + t * 3.0
    n = (
        math.sin(x * 12.9898) * 43758.5453 +
        math.sin(x * 78.233) * 12345.6789 +
        math.sin(x * 45.164) * 98765.4321
    )
    return (math.sin(n) + 1) / 2


# =============================================================================
# QUADRATIC BEZIER
# =============================================================================

def quad_bezier_pts(
    a: np.ndarray,
    cpt: np.ndarray,
    b: np.ndarray,
    n: int
) -> list[np.ndarray]:
    """Sample n+1 points along a quadratic Bézier curve."""
    pts = []
    for i in range(n + 1):
        t = i / n
        p0 = vec_lerp(a, cpt, t)
        p1 = vec_lerp(cpt, b, t)
        pts.append(vec_lerp(p0, p1, t))
    return pts


# =============================================================================
# LEAF GEOMETRY
# =============================================================================

@dataclass
class LeafGeom:
    """
    Leaf geometry with vein panels for stained-glass rendering.
    """
    x: float
    y: float
    length: float = 200.0
    max_width: float = 70.0
    sharpness: float = 1.6
    rotation: float = 0.0
    jitter: float = 0.0
    jitter_seed: float = 0.0
    outline_steps: int = 80

    def width_at(self, t: float) -> float:
        """Width profile at parameter t ∈ [0, 1]."""
        return self.max_width * (math.sin(math.pi * t) ** self.sharpness)

    def wobble_at(self, t: float) -> float:
        """Jitter offset at parameter t."""
        if self.jitter <= 0:
            return 0.0
        n = simple_noise(self.jitter_seed, t)
        return (n - 0.5) * self.jitter

    def half_width_at(self, t: float) -> float:
        """Half-width with jitter at parameter t."""
        return max(0, self.width_at(t) + self.wobble_at(t))

    def y_at(self, t: float) -> float:
        """Y position in local coords at parameter t."""
        return -t * self.length

    def local_mid(self, t: float) -> np.ndarray:
        return vec(0, self.y_at(t))

    def local_right(self, t: float) -> np.ndarray:
        return vec(+self.half_width_at(t), self.y_at(t))

    def local_left(self, t: float) -> np.ndarray:
        return vec(-self.half_width_at(t), self.y_at(t))

    def to_world(self, v: np.ndarray) -> np.ndarray:
        """Transform local coordinates to world coordinates."""
        c, s = math.cos(self.rotation), math.sin(self.rotation)
        return vec(
            self.x + (v[0] * c - v[1] * s),
            self.y + (v[0] * s + v[1] * c)
        )

    def sample_t(self, t_a: float, t_b: float, n: int,
                 fn: Callable[[float], np.ndarray]) -> list[np.ndarray]:
        pts = []
        for i in range(n + 1):
            t = t_a + (t_b - t_a) * i / n
            pts.append(fn(t))
        return pts

    def equal_area_breaks(self, n: int, samples: int = 2000) -> list[float]:
        """Compute n+1 breakpoints that divide leaf into n equal-area panels."""
        cum = [0.0] * (samples + 1)
        total = 0.0
        prev_t, prev_w = 0.0, self.half_width_at(0)

        for i in range(1, samples + 1):
            t = i / samples
            w = self.half_width_at(t)
            total += 0.5 * (prev_w + w) * (t - prev_t)
            cum[i] = total
            prev_t, prev_w = t, w

        def inv(target: float) -> float:
            lo, hi = 0, samples
            while lo < hi:
                mid = (lo + hi) // 2
                if cum[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            if lo <= 0:
                return 0.0
            if lo >= samples:
                return 1.0
            t0, t1 = (lo - 1) / samples, lo / samples
            a0, a1 = cum[lo - 1], cum[lo]
            alpha = (target - a0) / max(1e-9, a1 - a0)
            return t0 + (t1 - t0) * max(0, min(1, alpha))

        return [inv((total * k) / n) for k in range(n + 1)]

    def compute_vein_layout(self, n_veins: int, angle_deg: float,
                            eps: float = 1e-4) -> dict:
        all_breaks = self.equal_area_breaks(n_veins + 1)
        mid_ts = all_breaks[1:-1]

        theta = math.radians(max(45, min(89.5, angle_deg)))
        tan_theta = math.tan(theta)

        margin_ts = []
        for t0 in mid_ts:
            w0 = self.half_width_at(t0)
            dt = w0 / (self.length * tan_theta)
            margin_ts.append(max(0, min(1, t0 + dt)))

        for i in range(1, len(margin_ts)):
            margin_ts[i] = max(margin_ts[i], margin_ts[i - 1] + eps)
            margin_ts[i] = min(margin_ts[i], 1.0)

        return {
            'mid_all': [0.0] + mid_ts + [1.0],
            'mar_all': [0.0] + margin_ts + [1.0]
        }

    def vein_polyline_local(self, a: np.ndarray, b: np.ndarray,
                            curve: float = 0.0, samples: int = 10) -> list[np.ndarray]:
        if curve <= 0:
            return [a.copy(), b.copy()]

        v = b - a
        mid = a + v * 0.55
        perp = vec(-v[1], v[0])
        m = vec_mag(perp)
        if m > 1e-9:
            perp = perp / m
        if perp[1] > 0:
            perp = -perp

        amp = vec_mag(v) * curve
        cpt = mid + perp * amp
        return quad_bezier_pts(a, cpt, b, samples)

    def build_vein_boundaries(self, n_veins: int, angle_deg: float = 75.0,
                              curve: float = 0.0, vein_samples: int = 10) -> dict:
        layout = self.compute_vein_layout(n_veins, angle_deg)
        mid_all, mar_all = layout['mid_all'], layout['mar_all']

        right, left = [], []
        for k in range(len(mid_all)):
            tm, te = mid_all[k], mar_all[k]
            a = self.local_mid(tm)
            right.append(self.vein_polyline_local(a, self.local_right(te), curve, vein_samples))
            left.append(self.vein_polyline_local(a, self.local_left(te), curve, vein_samples))

        return {**layout, 'right': right, 'left': left}

    def get_outline_polygon(self) -> list[np.ndarray]:
        right, left = [], []
        for i in range(self.outline_steps + 1):
            t = i / self.outline_steps
            right.append(self.to_world(self.local_right(t)))
            left.append(self.to_world(self.local_left(t)))
        outline = list(reversed(right)) + left[1:]
        return outline

    def get_midrib_points(self) -> list[np.ndarray]:
        return [self.to_world(self.local_mid(i / self.outline_steps))
                for i in range(self.outline_steps + 1)]

    def get_vein_lines(self, n_veins: int, angle_deg: float = 75.0,
                       curve: float = 0.0, vein_samples: int = 10):
        vb = self.build_vein_boundaries(n_veins, angle_deg, curve, vein_samples)
        right_veins = [[self.to_world(p) for p in vb['right'][k]]
                       for k in range(1, len(vb['mid_all']) - 1)]
        left_veins = [[self.to_world(p) for p in vb['left'][k]]
                      for k in range(1, len(vb['mid_all']) - 1)]
        return right_veins, left_veins

    def get_vein_panels(self, n_veins: int, angle_deg: float = 75.0,
                        curve: float = 0.0, edge_samples: int = 12,
                        vein_samples: int = 10) -> dict:
        vb = self.build_vein_boundaries(n_veins, angle_deg, curve, vein_samples)
        mid_all, mar_all = vb['mid_all'], vb['mar_all']

        right_panels, left_panels = [], []
        for k in range(len(mid_all) - 1):
            tm0, tm1 = mid_all[k], mid_all[k + 1]
            te0, te1 = mar_all[k], mar_all[k + 1]

            m0, m1 = self.local_mid(tm0), self.local_mid(tm1)
            v0r, v1r = vb['right'][k], vb['right'][k + 1]
            v0l, v1l = vb['left'][k], vb['left'][k + 1]

            margin_r = self.sample_t(te1, te0, edge_samples, self.local_right)
            margin_l = self.sample_t(te1, te0, edge_samples, self.local_left)

            poly_r = [m0, m1] + v1r[1:] + margin_r[1:] + list(reversed(v0r[:-1]))
            poly_l = [m0, m1] + v1l[1:] + margin_l[1:] + list(reversed(v0l[:-1]))

            right_panels.append([self.to_world(p) for p in poly_r])
            left_panels.append([self.to_world(p) for p in poly_l])

        return {'right': right_panels, 'left': left_panels}


def make_leaf(x: float, y: float, length: float = 200.0, max_width: float = 70.0,
              sharpness: float = 1.6, rotation: float = 0.0,
              jitter: float = 0.0, jitter_seed: float = 0.0) -> LeafGeom:
    return LeafGeom(x=x, y=y, length=length, max_width=max_width,
                    sharpness=sharpness, rotation=rotation,
                    jitter=jitter, jitter_seed=jitter_seed)


# =============================================================================
# TREE PARAMETERS
# =============================================================================

@dataclass
class TreeParams:
    """Parameters for L-system tree generation."""
    angle: float = 28.0           # Branch angle in degrees
    scale: float = 0.68           # Length scale per level
    depth: int = 5                # Recursion depth
    trunk_length: float = 120.0   # Initial trunk length
    thickness: float = 6.0        # Base line thickness
    randomness: float = 0.10      # Random variation (0-1)
    early_stop: float = 0.15      # Early termination chance
    extra_growth: float = 0.20    # Extra growth chance
    show_roots: bool = True
    show_leaves: bool = True
    show_blossoms: bool = False
    blossom_type: str = 'cherry'  # cherry, apple, orange, berries, flowers
    blossom_density: float = 0.35


# =============================================================================
# TREE SKELETON (L-SYSTEM GENERATION)
# =============================================================================

@dataclass
class Branch:
    """A branch segment in the tree."""
    start: np.ndarray
    end: np.ndarray
    thickness: float
    depth: int
    is_root: bool = False
    is_terminal: bool = False


@dataclass
class TreeSkeleton:
    """Complete tree structure."""
    branches: list[Branch] = field(default_factory=list)
    leaf_positions: list[tuple[np.ndarray, float]] = field(default_factory=list)  # (pos, angle)
    blossom_positions: list[np.ndarray] = field(default_factory=list)


def generate_tree_skeleton(
    params: TreeParams,
    origin: np.ndarray,
    seed: int = 42
) -> TreeSkeleton:
    """
    Generate tree skeleton using L-system style recursion.

    Returns TreeSkeleton with branches, leaf positions, and blossom positions.
    """
    np.random.seed(seed)
    skeleton = TreeSkeleton()

    def draw_branch(pos: np.ndarray, direction: np.ndarray, length: float,
                    depth: int, thick: float, is_root: bool, max_depth: int):
        if depth <= 0:
            return

        variation = params.randomness
        this_len = length * np.random.uniform(1 - variation, 1 + variation)
        this_angle = params.angle * np.random.uniform(1 - variation, 1 + variation)

        # Calculate end point
        end = pos + direction * this_len

        # Add branch
        branch = Branch(
            start=pos.copy(),
            end=end.copy(),
            thickness=thick,
            depth=depth,
            is_root=is_root
        )
        skeleton.branches.append(branch)

        current_generation = max_depth - depth

        # Early termination check
        early_terminate = False
        if not is_root and current_generation >= 2 and depth > 1:
            if np.random.random() < params.early_stop:
                early_terminate = True
                branch.is_terminal = True

        # Add leaves at terminals
        if (depth <= 2 or early_terminate) and params.show_leaves and not is_root:
            # Add leaf positions on both sides
            base_angle = math.atan2(direction[1], direction[0])
            for side in [-1, 1]:
                leaf_angle = base_angle + side * math.radians(this_angle + np.random.uniform(-15, 15))
                skeleton.leaf_positions.append((end.copy(), leaf_angle))

        # Add blossom positions
        if not is_root:
            if depth <= 2 or early_terminate:
                skeleton.blossom_positions.append(end.copy())
            elif depth <= max_depth - 1 and np.random.random() < 0.4:
                skeleton.blossom_positions.append(end.copy())

        if early_terminate:
            return

        if depth > 1:
            new_len = this_len * params.scale
            new_thick = thick * 0.7

            # Extra growth bonuses
            right_bonus = 1 if (not is_root and current_generation >= 1 and
                               np.random.random() < params.extra_growth) else 0
            left_bonus = 1 if (not is_root and current_generation >= 1 and
                              np.random.random() < params.extra_growth) else 0

            # Right branch
            right_dir = vec_rotate(direction, math.radians(this_angle))
            draw_branch(end, right_dir, new_len, depth - 1 + right_bonus,
                       new_thick, is_root, max_depth)

            # Left branch
            left_dir = vec_rotate(direction, math.radians(-this_angle))
            draw_branch(end, left_dir, new_len, depth - 1 + left_bonus,
                       new_thick, is_root, max_depth)

            # Sometimes add middle branch
            if np.random.random() > 0.6 and depth > 2:
                mid_dir = vec_rotate(direction, math.radians(np.random.uniform(-10, 10)))
                draw_branch(end, mid_dir, new_len * 0.8, depth - 2,
                           new_thick * 0.8, is_root, max_depth)

    # Generate trunk and branches (going up, so direction is (0, -1) in screen coords)
    up_direction = vec(0, -1)
    draw_branch(origin, up_direction, params.trunk_length,
               params.depth, params.thickness, False, params.depth)

    # Generate roots (going down)
    if params.show_roots:
        down_direction = vec(0, 1)
        draw_branch(origin, down_direction, params.trunk_length * 0.8,
                   params.depth - 1, params.thickness, True, params.depth - 1)

    return skeleton


# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

@dataclass
class TreeStyle:
    """Style configuration for tree rendering."""
    # Lead lines
    lead_color: str = '#1e1914'
    branch_color: str = '#8c5537'
    root_color: str = '#64463c'

    # Leaf colors (autumn palette)
    leaf_colors: tuple = (
        '#b42840', '#dc3c32', '#c85028',
        '#e67828', '#289050', '#3ca05a',
        '#1e6446', '#643264', '#508c8c'
    )

    # Panel color variations (lighter/darker for left/right)
    panel_darken: float = 0.85

    # Vein parameters
    n_veins: int = 4
    vein_angle: float = 55.0
    vein_curve: float = 0.0

    # Background
    background_top: str = '#ffc850'
    background_bottom: str = '#f5b464'


# =============================================================================
# BLOSSOM/FRUIT RENDERING
# =============================================================================

def draw_cherry_blossom(ax: plt.Axes, pos: np.ndarray, size: float, rng: np.random.Generator):
    """Draw cherry blossom at position."""
    petal_count = 5
    for i in range(petal_count):
        angle = (2 * math.pi / petal_count) * i
        px = pos[0] + math.cos(angle) * size * 0.4
        py = pos[1] + math.sin(angle) * size * 0.4

        # Pink petal
        pink = rng.uniform(0.7, 0.9)
        color = (1.0, pink, min(1.0, pink + 0.05))
        ellipse = Ellipse((px, py), size * 0.5, size * 0.7, angle=math.degrees(angle),
                         facecolor=color, edgecolor='#1e1914', linewidth=1, zorder=15)
        ax.add_patch(ellipse)

    # Yellow center
    center = Circle(pos, size * 0.17, facecolor='#ffdc64', edgecolor='#1e1914', linewidth=0.8, zorder=16)
    ax.add_patch(center)


def draw_apple(ax: plt.Axes, pos: np.ndarray, size: float, rng: np.random.Generator):
    """Draw apple at position."""
    # Red or green
    if rng.random() > 0.3:
        color = (rng.uniform(0.7, 0.85), rng.uniform(0.1, 0.25), rng.uniform(0.15, 0.3))
    else:
        color = (rng.uniform(0.45, 0.7), rng.uniform(0.7, 0.85), rng.uniform(0.2, 0.4))

    apple = Circle(pos, size * 0.5, facecolor=color, edgecolor='#1e1914', linewidth=1.5, zorder=15)
    ax.add_patch(apple)

    # Stem
    ax.plot([pos[0], pos[0] + rng.uniform(-1, 1)],
            [pos[1] - size * 0.4, pos[1] - size * 0.6],
            color='#503c28', linewidth=1.5, solid_capstyle='round', zorder=16)


def draw_orange(ax: plt.Axes, pos: np.ndarray, size: float, rng: np.random.Generator):
    """Draw orange at position."""
    color = (rng.uniform(0.94, 1.0), rng.uniform(0.55, 0.7), rng.uniform(0.1, 0.25))
    orange = Circle(pos, size * 0.55, facecolor=color, edgecolor='#1e1914', linewidth=1.5, zorder=15)
    ax.add_patch(orange)


def draw_berries(ax: plt.Axes, pos: np.ndarray, size: float, rng: np.random.Generator):
    """Draw berry cluster at position."""
    berry_colors = [
        (0.24, 0.16, 0.35), (0.4, 0.16, 0.32),
        (0.16, 0.2, 0.4), (0.6, 0.16, 0.24)
    ]
    color = berry_colors[rng.integers(len(berry_colors))]

    cluster_size = rng.integers(3, 6)
    for i in range(cluster_size):
        angle = (2 * math.pi / cluster_size) * i + rng.uniform(-0.3, 0.3)
        dist = rng.uniform(size * 0.15, size * 0.4)
        bx = pos[0] + math.cos(angle) * dist
        by = pos[1] + math.sin(angle) * dist
        berry = Circle((bx, by), size * 0.2, facecolor=color,
                       edgecolor='#1e1914', linewidth=1, zorder=15)
        ax.add_patch(berry)


def draw_wildflower(ax: plt.Axes, pos: np.ndarray, size: float, rng: np.random.Generator):
    """Draw wildflower at position."""
    flower_colors = [
        (1.0, 0.4, 0.6), (0.6, 0.4, 0.8), (0.4, 0.6, 1.0),
        (1.0, 0.8, 0.4), (1.0, 0.6, 0.4), (1.0, 1.0, 0.8)
    ]
    color = flower_colors[rng.integers(len(flower_colors))]
    petal_count = rng.integers(5, 9)

    for i in range(petal_count):
        angle = (2 * math.pi / petal_count) * i
        px = pos[0] + math.cos(angle) * size * 0.35
        py = pos[1] + math.sin(angle) * size * 0.35
        ellipse = Ellipse((px, py), size * 0.3, size * 0.55, angle=math.degrees(angle),
                         facecolor=color, edgecolor='#1e1914', linewidth=1, zorder=15)
        ax.add_patch(ellipse)

    # Center
    center_colors = [(1.0, 0.86, 0.3), (0.3, 0.24, 0.16), (1.0, 0.7, 0.4)]
    center = Circle(pos, size * 0.2, facecolor=center_colors[rng.integers(len(center_colors))],
                   edgecolor='#1e1914', linewidth=0.8, zorder=16)
    ax.add_patch(center)


BLOSSOM_DRAWERS = {
    'cherry': draw_cherry_blossom,
    'apple': draw_apple,
    'orange': draw_orange,
    'berries': draw_berries,
    'flowers': draw_wildflower,
}


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_tree(
    params: TreeParams = None,
    style: TreeStyle = None,
    seed: int = 42,
    canvas_size: tuple = (550, 550),
    figsize: tuple = (8, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Render a stained-glass style tree.

    Args:
        params: Tree generation parameters
        style: Visual style configuration
        seed: Random seed for reproducibility
        canvas_size: Virtual canvas size (width, height)
        figsize: Figure size in inches

    Returns:
        (figure, axes) tuple
    """
    if params is None:
        params = TreeParams()
    if style is None:
        style = TreeStyle()

    rng = np.random.default_rng(seed)
    width, height = canvas_size

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Flip Y for screen coords
    ax.set_aspect('equal')
    ax.axis('off')

    # Background gradient
    for y in range(height):
        inter = y / height
        r = int(255 * (1 - inter) + 245 * inter) / 255
        g = int(200 * (1 - inter) + 180 * inter) / 255
        b = int(80 * (1 - inter) + 100 * inter) / 255
        ax.axhline(y=y, color=(r, g, b), linewidth=1)

    # Generate tree skeleton
    ground_y = height * 0.65
    origin = vec(width / 2, ground_y)
    skeleton = generate_tree_skeleton(params, origin, seed)

    # Draw branches
    for branch in skeleton.branches:
        # Outline
        ax.plot([branch.start[0], branch.end[0]],
                [branch.start[1], branch.end[1]],
                color=style.lead_color, linewidth=branch.thickness + 2,
                solid_capstyle='round')
        # Fill
        fill_color = style.root_color if branch.is_root else style.branch_color
        ax.plot([branch.start[0], branch.end[0]],
                [branch.start[1], branch.end[1]],
                color=fill_color, linewidth=branch.thickness,
                solid_capstyle='round')

    # Collect all leaf data first, then draw in proper order
    leaf_data = []
    if params.show_leaves:
        leaf_colors = style.leaf_colors
        for i, (pos, angle) in enumerate(skeleton.leaf_positions):
            # Leaf parameters - bigger leaves for visibility
            leaf_size = rng.uniform(22, 38)
            leaf_width = leaf_size * rng.uniform(0.38, 0.48)
            color_idx = rng.integers(len(leaf_colors))
            base_color = leaf_colors[color_idx]

            # Create leaf geometry
            leaf = make_leaf(
                x=pos[0], y=pos[1],
                length=leaf_size,
                max_width=leaf_width,
                sharpness=1.4,
                rotation=angle + math.pi/2,  # Point outward
                jitter=rng.uniform(0, 0.2),
                jitter_seed=seed + i
            )

            # Get panels
            panels = leaf.get_vein_panels(
                style.n_veins,
                angle_deg=style.vein_angle,
                curve=style.vein_curve,
                edge_samples=8,
                vein_samples=8
            )

            # Parse base color
            if base_color.startswith('#'):
                r = int(base_color[1:3], 16) / 255
                g = int(base_color[3:5], 16) / 255
                b = int(base_color[5:7], 16) / 255
            else:
                r, g, b = 0.5, 0.5, 0.5

            leaf_data.append({
                'leaf': leaf,
                'panels': panels,
                'rgb': (r, g, b)
            })

    # Draw each leaf as a complete unit (solid base + panels + outlines)
    for leaf_idx, ld in enumerate(leaf_data):
        r, g, b = ld['rgb']
        panels = ld['panels']
        leaf = ld['leaf']

        # Base zorder for this leaf (later leaves on top)
        base_z = 10 + leaf_idx * 0.01

        # First: draw solid fill for entire leaf shape (covers any gaps)
        outline = leaf.get_outline_polygon()
        outline_pts = np.array([[p[0], p[1]] for p in outline])
        base_fill = MplPolygon(outline_pts, facecolor=(r, g, b), edgecolor='none',
                               zorder=base_z, alpha=1.0)
        ax.add_patch(base_fill)

        # Second: draw panels on top of the base fill
        for j, panel in enumerate(panels['right']):
            shade = 0.95 + 0.05 * (j / max(1, len(panels['right'])))
            color = (min(1, r * shade), min(1, g * shade), min(1, b * shade))
            pts = np.array([[p[0], p[1]] for p in panel])
            patch = MplPolygon(pts, facecolor=color, edgecolor='none', zorder=base_z + 0.001, alpha=1.0)
            ax.add_patch(patch)

        for j, panel in enumerate(panels['left']):
            shade = style.panel_darken + 0.05 * (j / max(1, len(panels['left'])))
            color = (min(1, r * shade), min(1, g * shade), min(1, b * shade))
            pts = np.array([[p[0], p[1]] for p in panel])
            patch = MplPolygon(pts, facecolor=color, edgecolor='none', zorder=base_z + 0.001, alpha=1.0)
            ax.add_patch(patch)

        # Third: draw outline and veins for this leaf
        outline_z = base_z + 0.002

        # Draw outline
        outline_xs = [p[0] for p in outline] + [outline[0][0]]
        outline_ys = [p[1] for p in outline] + [outline[0][1]]
        ax.plot(outline_xs, outline_ys, color=style.lead_color, linewidth=1.5,
                solid_capstyle='round', solid_joinstyle='round', zorder=outline_z)

        # Draw midrib
        midrib = leaf.get_midrib_points()
        mx = [p[0] for p in midrib]
        my = [p[1] for p in midrib]
        ax.plot(mx, my, color=style.lead_color, linewidth=0.8,
                solid_capstyle='round', zorder=outline_z)

        # Draw veins
        right_veins, left_veins = leaf.get_vein_lines(
            style.n_veins, style.vein_angle, style.vein_curve, 8
        )
        for vein in right_veins + left_veins:
            vx = [p[0] for p in vein]
            vy = [p[1] for p in vein]
            ax.plot(vx, vy, color=style.lead_color, linewidth=0.6,
                    solid_capstyle='round', zorder=outline_z)

    # Draw blossoms/fruit
    if params.show_blossoms:
        drawer = BLOSSOM_DRAWERS.get(params.blossom_type, draw_cherry_blossom)
        for pos in skeleton.blossom_positions:
            if rng.random() < params.blossom_density:
                size = rng.uniform(10, 18)
                # Offset slightly
                offset_pos = pos + vec(rng.uniform(-5, 5), rng.uniform(-5, 5))
                drawer(ax, offset_pos, size, rng)

    return fig, ax


def save_tree(
    filepath: str,
    params: TreeParams = None,
    style: TreeStyle = None,
    seed: int = 42,
    dpi: int = 150,
    canvas_size: tuple = (550, 550),
    figsize: tuple = (8, 8),
):
    """Render and save a tree to file."""
    fig, ax = render_tree(params, style, seed, canvas_size, figsize)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved to {filepath}")


# =============================================================================
# STRESS-MORPHOLOGY INTEGRATION
# =============================================================================

@dataclass
class StressVisuals:
    """
    Computed stress indicators for visualization.

    These values are normalized to [0, 1] where:
    - 0 = no stress (healthy conditions)
    - 1 = maximum stress

    Derived from simulation Trajectory and ClimateConfig.
    """
    # Environmental stress levels (from climate)
    drought_stress: float = 0.0      # Low moisture availability
    light_stress: float = 0.0        # Low light (shade)
    wind_stress: float = 0.0         # High wind exposure
    nutrient_stress: float = 0.0     # Low soil nutrients (inferred from growth)

    # Tree condition indicators (from final state)
    vigor: float = 1.0               # Overall health (0=dying, 1=thriving)
    leaf_health: float = 1.0         # Leaf condition (chlorosis, wilting)
    structural_investment: float = 0.5  # Trunk/root ratio (wind adaptation)
    reproductive_success: float = 0.0   # Fruit/flower production

    # Growth pattern modifiers
    etiolation: float = 0.0          # Elongated growth from low light
    compactness: float = 0.5         # Dense vs sparse canopy
    asymmetry: float = 0.0           # Wind-induced lean/asymmetry


def compute_stress_visuals(
    trajectory: 'Trajectory',
    climate: 'ClimateConfig',
) -> StressVisuals:
    """
    Analyze simulation results to compute stress visualization parameters.

    Args:
        trajectory: Complete simulation trajectory
        climate: Climate configuration used in simulation

    Returns:
        StressVisuals with computed stress indicators
    """
    import numpy as np

    # Get final state
    final_state = trajectory.states[-1]

    # Compute average environmental conditions
    avg_moisture = np.mean(trajectory.moisture_history)
    avg_light = np.mean(trajectory.light_history)
    avg_wind = np.mean(trajectory.wind_history)

    # Environmental stress levels
    # Drought: low moisture is stressful
    drought_stress = float(np.clip(1.0 - avg_moisture / 0.6, 0, 1))

    # Light stress: low light causes etiolation
    light_stress = float(np.clip(1.0 - avg_light / 0.7, 0, 1))

    # Wind stress: high wind is stressful
    wind_stress = float(np.clip((avg_wind - 0.2) / 0.5, 0, 1))

    # Compute vigor from biomass and energy
    total_biomass = float(final_state.total_biomass())
    final_energy = float(final_state.energy)

    # Expected healthy biomass after 100 days is roughly 2-4 units
    vigor = float(np.clip(total_biomass / 3.0, 0, 1))

    # Leaf health: ratio of leaves to expected, modified by water stress
    leaf_mass = float(final_state.leaves)
    expected_leaves = total_biomass * 0.3  # Healthy trees ~30% leaves
    leaf_health = float(np.clip(leaf_mass / max(0.1, expected_leaves), 0, 1))

    # Nutrient stress: inferred from low leaf mass relative to structure
    # If lots of wood but few leaves, likely nutrient limited
    trunk_mass = float(final_state.trunk)
    if trunk_mass > 0.1:
        leaf_trunk_ratio = leaf_mass / trunk_mass
        nutrient_stress = float(np.clip(1.0 - leaf_trunk_ratio / 2.0, 0, 1))
    else:
        nutrient_stress = 0.0

    # Structural investment: trunk + roots relative to total
    root_mass = float(final_state.roots)
    structural_investment = float(np.clip(
        (trunk_mass + root_mass) / max(0.1, total_biomass), 0, 1
    ))

    # Reproductive success
    fruit_mass = float(final_state.fruit)
    flower_mass = float(final_state.flowers)
    reproductive_success = float(np.clip(
        (fruit_mass + flower_mass * 0.5) / max(0.1, total_biomass), 0, 1
    ))

    # Etiolation from low light
    etiolation = light_stress * 0.8

    # Compactness: inverse of wind stress (windy = compact)
    compactness = float(np.clip(0.3 + wind_stress * 0.5 + drought_stress * 0.2, 0, 1))

    # Asymmetry from wind
    asymmetry = wind_stress * 0.6

    return StressVisuals(
        drought_stress=drought_stress,
        light_stress=light_stress,
        wind_stress=wind_stress,
        nutrient_stress=nutrient_stress,
        vigor=vigor,
        leaf_health=leaf_health,
        structural_investment=structural_investment,
        reproductive_success=reproductive_success,
        etiolation=etiolation,
        compactness=compactness,
        asymmetry=asymmetry,
    )


def stress_to_params(stress: StressVisuals, base_params: TreeParams = None) -> TreeParams:
    """
    Map stress indicators to tree generation parameters.

    Args:
        stress: Computed stress visualization indicators
        base_params: Base parameters to modify (uses defaults if None)

    Returns:
        TreeParams adjusted for stress conditions
    """
    if base_params is None:
        base_params = TreeParams()

    # Vigor affects overall tree size
    trunk_scale = 0.6 + 0.6 * stress.vigor  # 60-120% of base
    depth_mod = -1 if stress.vigor < 0.3 else 0  # Stunted if very low vigor

    # Drought: sparser canopy, more early termination
    early_stop = base_params.early_stop + stress.drought_stress * 0.25

    # Wind: more compact, less randomness, asymmetric
    angle_mod = -5 * stress.wind_stress  # Tighter angles in wind
    randomness_mod = -0.05 * stress.wind_stress

    # Low light: etiolation (longer, thinner branches)
    scale_mod = stress.etiolation * 0.1  # Longer internodes
    thickness_mod = -stress.etiolation * 2  # Thinner branches

    # Reproductive success affects blossoms
    show_blossoms = stress.reproductive_success > 0.1
    blossom_density = stress.reproductive_success * 0.8

    return TreeParams(
        angle=base_params.angle + angle_mod,
        scale=min(0.85, base_params.scale + scale_mod),
        depth=max(2, base_params.depth + depth_mod),
        trunk_length=base_params.trunk_length * trunk_scale,
        thickness=max(2, base_params.thickness + thickness_mod),
        randomness=max(0, base_params.randomness + randomness_mod),
        early_stop=min(0.5, early_stop),
        extra_growth=base_params.extra_growth * stress.vigor,
        show_roots=base_params.show_roots,
        show_leaves=base_params.show_leaves,
        show_blossoms=show_blossoms,
        blossom_type=base_params.blossom_type,
        blossom_density=blossom_density,
    )


def stress_to_style(stress: StressVisuals, base_style: TreeStyle = None) -> TreeStyle:
    """
    Map stress indicators to visual style parameters.

    Args:
        stress: Computed stress visualization indicators
        base_style: Base style to modify (uses defaults if None)

    Returns:
        TreeStyle adjusted for stress conditions
    """
    if base_style is None:
        base_style = TreeStyle()

    # Color palettes for different stress conditions
    # Healthy: rich greens, autumn reds/oranges
    healthy_colors = (
        '#b42840', '#dc3c32', '#c85028',
        '#e67828', '#289050', '#3ca05a',
        '#1e6446', '#643264', '#508c8c'
    )

    # Drought-stressed: yellows, browns, muted
    drought_colors = (
        '#c4a030', '#d4b040', '#b89030',
        '#a08028', '#7a9048', '#8a9850',
        '#687838', '#786850', '#708068'
    )

    # Nutrient-deficient: pale, chlorotic yellows
    chlorotic_colors = (
        '#c8c878', '#d0d080', '#b8b868',
        '#c0c070', '#98b878', '#a8c088',
        '#88a868', '#a8a880', '#90a890'
    )

    # Low light: dark greens, elongated look
    shade_colors = (
        '#1a4030', '#205038', '#184028',
        '#286048', '#185830', '#206838',
        '#144820', '#304048', '#285050'
    )

    # Blend colors based on stress levels
    def blend_color(hex1: str, hex2: str, t: float) -> str:
        """Blend two hex colors by factor t (0=hex1, 1=hex2)."""
        r1, g1, b1 = int(hex1[1:3], 16), int(hex1[3:5], 16), int(hex1[5:7], 16)
        r2, g2, b2 = int(hex2[1:3], 16), int(hex2[3:5], 16), int(hex2[5:7], 16)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f'#{r:02x}{g:02x}{b:02x}'

    # Determine dominant stress and blend colors
    max_stress = max(stress.drought_stress, stress.nutrient_stress, stress.light_stress)

    if max_stress < 0.2:
        # Healthy tree
        leaf_colors = healthy_colors
    elif stress.drought_stress >= stress.nutrient_stress and stress.drought_stress >= stress.light_stress:
        # Drought dominant
        t = min(1.0, stress.drought_stress)
        leaf_colors = tuple(
            blend_color(healthy_colors[i], drought_colors[i], t)
            for i in range(len(healthy_colors))
        )
    elif stress.nutrient_stress >= stress.light_stress:
        # Nutrient deficiency dominant
        t = min(1.0, stress.nutrient_stress)
        leaf_colors = tuple(
            blend_color(healthy_colors[i], chlorotic_colors[i], t)
            for i in range(len(healthy_colors))
        )
    else:
        # Shade dominant
        t = min(1.0, stress.light_stress)
        leaf_colors = tuple(
            blend_color(healthy_colors[i], shade_colors[i], t)
            for i in range(len(healthy_colors))
        )

    # Leaf health affects panel darkness
    panel_darken = 0.75 + 0.15 * stress.leaf_health  # Sicker = darker panels

    # Fewer veins on stressed leaves
    n_veins = max(2, int(4 * stress.leaf_health))

    return TreeStyle(
        lead_color=base_style.lead_color,
        branch_color=base_style.branch_color,
        root_color=base_style.root_color,
        leaf_colors=leaf_colors,
        panel_darken=panel_darken,
        n_veins=n_veins,
        vein_angle=base_style.vein_angle,
        vein_curve=base_style.vein_curve,
        background_top=base_style.background_top,
        background_bottom=base_style.background_bottom,
    )


def render_stressed_tree(
    trajectory: 'Trajectory',
    climate: 'ClimateConfig',
    seed: int = 42,
    base_params: TreeParams = None,
    base_style: TreeStyle = None,
    canvas_size: tuple = (550, 550),
    figsize: tuple = (8, 8),
) -> tuple[plt.Figure, plt.Axes, StressVisuals]:
    """
    Render a tree whose appearance reflects simulation stress history.

    Args:
        trajectory: Simulation trajectory with stress history
        climate: Climate configuration used in simulation
        seed: Random seed for rendering
        base_params: Base tree parameters (stress modifies these)
        base_style: Base visual style (stress modifies these)
        canvas_size: Canvas dimensions
        figsize: Figure size

    Returns:
        (figure, axes, stress_visuals) tuple
    """
    # Compute stress indicators from simulation
    stress = compute_stress_visuals(trajectory, climate)

    # Map stress to visual parameters
    params = stress_to_params(stress, base_params)
    style = stress_to_style(stress, base_style)

    # Render the tree
    fig, ax = render_tree(params, style, seed, canvas_size, figsize)

    return fig, ax, stress


def save_stressed_tree(
    filepath: str,
    trajectory: 'Trajectory',
    climate: 'ClimateConfig',
    seed: int = 42,
    dpi: int = 150,
    base_params: TreeParams = None,
    base_style: TreeStyle = None,
    canvas_size: tuple = (550, 550),
    figsize: tuple = (8, 8),
) -> StressVisuals:
    """
    Render and save a stress-responsive tree visualization.

    Returns the computed StressVisuals for inspection.
    """
    fig, ax, stress = render_stressed_tree(
        trajectory, climate, seed, base_params, base_style, canvas_size, figsize
    )
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved to {filepath}")
    return stress


# =============================================================================
# CONVENIENCE WRAPPERS
# =============================================================================

def render_stained_glass(seed: int = 42, **kwargs):
    """Backward-compatible wrapper."""
    return render_tree(seed=seed, **kwargs)


def save_stained_glass(filepath: str, seed: int = 42, dpi: int = 150, **kwargs):
    """Backward-compatible wrapper."""
    save_tree(filepath, seed=seed, dpi=dpi, **kwargs)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Leaf geometry
    'LeafGeom',
    'make_leaf',
    # Tree structure
    'TreeParams',
    'TreeStyle',
    'TreeSkeleton',
    'Branch',
    'generate_tree_skeleton',
    # Rendering
    'render_tree',
    'save_tree',
    'render_stained_glass',
    'save_stained_glass',
    # Stress integration
    'StressVisuals',
    'compute_stress_visuals',
    'stress_to_params',
    'stress_to_style',
    'render_stressed_tree',
    'save_stressed_tree',
]
