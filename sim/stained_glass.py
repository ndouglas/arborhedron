"""
Stained-glass leaf geometry renderer.

Port of p5.js leaf geometry code to Python/matplotlib.
The tree is conceptualized as a leaf, with equal-area vein panels
forming natural stained-glass segments.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path
from matplotlib.patches import PathPatch
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


# =============================================================================
# NOISE FUNCTION
# =============================================================================

def simple_noise(seed: float, t: float) -> float:
    """
    Simple deterministic noise function.

    Uses sine-based pseudo-noise for reproducibility.
    Returns value in [0, 1].
    """
    x = seed + t * 3.0
    # Multiple sine waves at different frequencies for organic feel
    n = (
        math.sin(x * 12.9898) * 43758.5453 +
        math.sin(x * 78.233) * 12345.6789 +
        math.sin(x * 45.164) * 98765.4321
    )
    return (math.sin(n) + 1) / 2  # Normalize to [0, 1]


# =============================================================================
# COLOR UTILITIES
# =============================================================================

def rgb_to_hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Convert RGB [0-1] to HSV [0-1]."""
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    diff = max_c - min_c

    # Value
    v = max_c

    # Saturation
    if max_c < 1e-9:
        s = 0.0
    else:
        s = diff / max_c

    # Hue
    if diff < 1e-9:
        h = 0.0
    elif max_c == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_c == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360

    return (h / 360, s, v)  # Normalize hue to [0-1]


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """Convert HSV [0-1] to RGB [0-1]."""
    h = h * 360  # Convert to degrees
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    return (r + m, g + m, b + m)


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert hex color to RGB [0-1]."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_hex(r: float, g: float, b: float) -> str:
    """Convert RGB [0-1] to hex color."""
    return '#{:02x}{:02x}{:02x}'.format(
        int(min(255, max(0, r * 255))),
        int(min(255, max(0, g * 255))),
        int(min(255, max(0, b * 255)))
    )


def apply_color_stress(
    hex_color: str,
    hue_shift: float = 0.0,
    saturation_mult: float = 1.0,
    value_mult: float = 1.0
) -> str:
    """
    Apply stress-based color transformation.

    Args:
        hex_color: Original color in hex format
        hue_shift: Shift hue by this amount [0-1] (positive = toward yellow/red)
        saturation_mult: Multiply saturation by this factor
        value_mult: Multiply value/brightness by this factor

    Returns:
        Transformed color in hex format
    """
    r, g, b = hex_to_rgb(hex_color)
    h, s, v = rgb_to_hsv(r, g, b)

    # Apply transformations
    h = (h + hue_shift) % 1.0
    s = min(1.0, max(0.0, s * saturation_mult))
    v = min(1.0, max(0.0, v * value_mult))

    r, g, b = hsv_to_rgb(h, s, v)
    return rgb_to_hex(r, g, b)


# =============================================================================
# STRESS PROFILE
# =============================================================================

@dataclass
class StressProfile:
    """
    Composable environmental stress profile for plant visualization.

    Stress profiles can be combined using multiplication (*) operator.
    Multiplier fields (1.0 = baseline) compose multiplicatively.
    Shift/additive fields (0.0 = baseline) compose additively.

    Example:
        # Single stress
        stress = StressProfile.drought(0.7)

        # Composed stress - nitrogen-deficient drought
        stress = StressProfile.drought(0.6) * StressProfile.nitrogen_deficient(0.5)
    """

    # === Multiplier fields (1.0 = healthy baseline) ===
    # These compose multiplicatively

    leaf_size: float = 1.0          # Leaf length/width scale
    leaf_count: float = 1.0         # Number of leaves multiplier
    leaf_sharpness: float = 1.0     # Leaf shape (>1 = narrower, pointier)
    leaf_saturation: float = 1.0    # Color saturation (<1 = pale/chlorotic)
    leaf_value: float = 1.0         # Color brightness (<1 = browning)

    blossom_size: float = 1.0       # Blossom radius scale
    blossom_count: float = 1.0      # Number of blossoms multiplier
    blossom_saturation: float = 1.0 # Petal color saturation

    stem_width: float = 1.0         # Stem thickness
    stem_length: float = 1.0        # Stem length/internode spacing

    # === Additive/shift fields (0.0 = healthy baseline) ===
    # These compose additively

    leaf_hue_shift: float = 0.0     # Hue shift [0-1] (+ = yellow/red, - = blue)
    leaf_edge_damage: float = 0.0   # Brown edge necrosis [0-1]
    leaf_curl: float = 0.0          # Drought curl effect [0-1]
    leaf_vein_prominence: float = 0.0  # More visible veins under stress [0-1]

    blossom_hue_shift: float = 0.0  # Petal hue shift

    def __mul__(self, other: 'StressProfile') -> 'StressProfile':
        """
        Compose two stress profiles.

        Multiplier fields are multiplied together.
        Additive fields are summed (with clamping where appropriate).
        """
        return StressProfile(
            # Multiplicative fields
            leaf_size=self.leaf_size * other.leaf_size,
            leaf_count=self.leaf_count * other.leaf_count,
            leaf_sharpness=self.leaf_sharpness * other.leaf_sharpness,
            leaf_saturation=self.leaf_saturation * other.leaf_saturation,
            leaf_value=self.leaf_value * other.leaf_value,
            blossom_size=self.blossom_size * other.blossom_size,
            blossom_count=self.blossom_count * other.blossom_count,
            blossom_saturation=self.blossom_saturation * other.blossom_saturation,
            stem_width=self.stem_width * other.stem_width,
            stem_length=self.stem_length * other.stem_length,
            # Additive fields (clamped to reasonable ranges)
            leaf_hue_shift=max(-0.5, min(0.5, self.leaf_hue_shift + other.leaf_hue_shift)),
            leaf_edge_damage=min(1.0, self.leaf_edge_damage + other.leaf_edge_damage),
            leaf_curl=min(1.0, self.leaf_curl + other.leaf_curl),
            leaf_vein_prominence=min(1.0, self.leaf_vein_prominence + other.leaf_vein_prominence),
            blossom_hue_shift=max(-0.5, min(0.5, self.blossom_hue_shift + other.blossom_hue_shift)),
        )

    def __rmul__(self, other: 'StressProfile') -> 'StressProfile':
        """Support reverse multiplication."""
        return self.__mul__(other)

    @classmethod
    def healthy(cls) -> 'StressProfile':
        """Healthy baseline - no stress."""
        return cls()

    @classmethod
    def drought(cls, severity: float = 1.0) -> 'StressProfile':
        """
        Drought stress profile.

        Effects: smaller/fewer leaves, narrower shape, yellowing,
        curling, stress-induced flowering, thinner/shorter stems.
        """
        s = max(0.0, min(1.0, severity))
        return cls(
            leaf_size=1.0 - 0.4 * s,        # Up to 40% smaller
            leaf_count=1.0 - 0.3 * s,       # Fewer leaves (abscission)
            leaf_sharpness=1.0 + 0.5 * s,   # Narrower, more pointed
            leaf_saturation=1.0 - 0.3 * s,  # Less vibrant
            leaf_value=1.0 - 0.15 * s,      # Slight browning
            blossom_count=1.0 + 0.4 * s,    # Stress-induced flowering
            blossom_size=1.0 - 0.2 * s,     # Smaller flowers
            stem_width=1.0 - 0.25 * s,      # Thinner stems
            stem_length=1.0 - 0.3 * s,      # Shorter internodes
            leaf_hue_shift=0.08 * s,        # Yellowing
            leaf_curl=0.5 * s,              # Leaf curling
            leaf_edge_damage=0.2 * s,       # Some edge browning
        )

    @classmethod
    def flood(cls, severity: float = 1.0) -> 'StressProfile':
        """
        Flood/waterlogging stress profile.

        Effects: yellowing (chlorosis), wilting appearance,
        swollen stem base, reduced flowering.
        """
        s = max(0.0, min(1.0, severity))
        return cls(
            leaf_size=1.0 - 0.15 * s,       # Slightly smaller
            leaf_count=1.0 - 0.25 * s,      # Some leaf drop
            leaf_saturation=1.0 - 0.45 * s, # Chlorotic/pale
            leaf_value=1.0 - 0.1 * s,       # Slight darkening
            blossom_count=1.0 - 0.4 * s,    # Reduced flowering
            blossom_saturation=1.0 - 0.3 * s,
            stem_width=1.0 + 0.15 * s,      # Swollen (adventitious roots)
            leaf_hue_shift=0.1 * s,         # Yellow chlorosis
            leaf_vein_prominence=0.3 * s,   # Veins more visible
        )

    @classmethod
    def nitrogen_deficient(cls, severity: float = 1.0) -> 'StressProfile':
        """
        Nitrogen deficiency stress profile.

        Effects: pale/yellow leaves (older first), smaller leaves,
        stunted growth, delayed flowering.
        """
        s = max(0.0, min(1.0, severity))
        return cls(
            leaf_size=1.0 - 0.35 * s,       # Smaller leaves
            leaf_count=1.0 - 0.2 * s,       # Fewer leaves
            leaf_saturation=1.0 - 0.5 * s,  # Very pale
            stem_length=1.0 - 0.3 * s,      # Stunted
            stem_width=1.0 - 0.15 * s,      # Thinner
            blossom_count=1.0 - 0.35 * s,   # Delayed/fewer flowers
            leaf_hue_shift=0.12 * s,        # Strong yellowing
        )

    @classmethod
    def nitrogen_excess(cls, severity: float = 1.0) -> 'StressProfile':
        """
        Nitrogen excess stress profile.

        Effects: large dark green leaves, abundant foliage,
        weak elongated stems, delayed flowering.
        """
        s = max(0.0, min(1.0, severity))
        return cls(
            leaf_size=1.0 + 0.3 * s,        # Larger leaves
            leaf_count=1.0 + 0.4 * s,       # More leaves
            leaf_saturation=1.0 + 0.2 * s,  # Darker green
            stem_length=1.0 + 0.25 * s,     # Elongated (weak)
            stem_width=1.0 - 0.1 * s,       # Weaker stems
            blossom_count=1.0 - 0.5 * s,    # Much delayed flowering
            blossom_size=1.0 - 0.15 * s,    # Smaller when they appear
            leaf_hue_shift=-0.03 * s,       # Slightly bluer/darker green
        )

    @classmethod
    def phosphorus_deficient(cls, severity: float = 1.0) -> 'StressProfile':
        """
        Phosphorus deficiency stress profile.

        Effects: purple/red tints, small leaves, stunted growth,
        delayed maturity.
        """
        s = max(0.0, min(1.0, severity))
        return cls(
            leaf_size=1.0 - 0.3 * s,        # Smaller
            leaf_count=1.0 - 0.15 * s,      # Slightly fewer
            stem_length=1.0 - 0.35 * s,     # Very stunted
            blossom_count=1.0 - 0.4 * s,    # Delayed
            leaf_hue_shift=-0.15 * s,       # Purple/red shift (toward blue/magenta)
            leaf_saturation=1.0 + 0.1 * s,  # Colors can be intense
        )

    @classmethod
    def potassium_deficient(cls, severity: float = 1.0) -> 'StressProfile':
        """
        Potassium deficiency stress profile.

        Effects: brown leaf edges (marginal necrosis), weak stems,
        poor flower/fruit quality.
        """
        s = max(0.0, min(1.0, severity))
        return cls(
            leaf_size=1.0 - 0.1 * s,        # Slightly smaller
            stem_width=1.0 - 0.2 * s,       # Weak stems
            blossom_size=1.0 - 0.25 * s,    # Poor quality
            blossom_saturation=1.0 - 0.2 * s,
            leaf_edge_damage=0.7 * s,       # Strong marginal necrosis
            leaf_hue_shift=0.05 * s,        # Slight yellowing
        )

    @classmethod
    def low_light(cls, severity: float = 1.0) -> 'StressProfile':
        """
        Low light (etiolation) stress profile.

        Effects: elongated stems, large thin pale leaves, few flowers.
        """
        s = max(0.0, min(1.0, severity))
        return cls(
            leaf_size=1.0 + 0.2 * s,        # Larger (to capture light)
            leaf_sharpness=1.0 - 0.3 * s,   # Broader, thinner
            leaf_saturation=1.0 - 0.35 * s, # Pale
            stem_length=1.0 + 0.5 * s,      # Very elongated (etiolated)
            stem_width=1.0 - 0.2 * s,       # Thinner stems
            blossom_count=1.0 - 0.6 * s,    # Few flowers
            leaf_hue_shift=0.05 * s,        # Slight yellowing
        )

    @classmethod
    def high_light(cls, severity: float = 1.0) -> 'StressProfile':
        """
        High light/sun scorch stress profile.

        Effects: bleached/yellow spots, smaller tougher leaves.
        """
        s = max(0.0, min(1.0, severity))
        return cls(
            leaf_size=1.0 - 0.2 * s,        # Smaller, tougher
            leaf_sharpness=1.0 + 0.2 * s,   # More compact shape
            leaf_saturation=1.0 - 0.4 * s,  # Bleached
            leaf_value=1.0 + 0.1 * s,       # Lighter/bleached
            leaf_hue_shift=0.1 * s,         # Yellowing
            leaf_edge_damage=0.3 * s,       # Scorch damage
        )

    @classmethod
    def wind_stress(cls, severity: float = 1.0) -> 'StressProfile':
        """
        Wind stress profile.

        Effects: smaller tougher leaves, thicker stems,
        asymmetric growth (not modeled here), fewer flowers.
        """
        s = max(0.0, min(1.0, severity))
        return cls(
            leaf_size=1.0 - 0.25 * s,       # Smaller, tougher
            leaf_sharpness=1.0 + 0.3 * s,   # Narrower
            stem_width=1.0 + 0.3 * s,       # Thicker, stronger
            stem_length=1.0 - 0.15 * s,     # Shorter internodes
            blossom_count=1.0 - 0.2 * s,    # Some reduction
        )

    @classmethod
    def cold_stress(cls, severity: float = 1.0) -> 'StressProfile':
        """
        Cold stress profile.

        Effects: purple/red coloration, stunted growth, curling.
        """
        s = max(0.0, min(1.0, severity))
        return cls(
            leaf_size=1.0 - 0.2 * s,
            stem_length=1.0 - 0.25 * s,     # Stunted
            leaf_hue_shift=-0.1 * s,        # Purple/red shift
            leaf_curl=0.3 * s,              # Curling
            blossom_count=1.0 - 0.3 * s,
        )

    @classmethod
    def heat_stress(cls, severity: float = 1.0) -> 'StressProfile':
        """
        Heat stress profile.

        Effects: wilting, curling, yellowing, reduced flowering.
        """
        s = max(0.0, min(1.0, severity))
        return cls(
            leaf_saturation=1.0 - 0.25 * s,
            leaf_curl=0.6 * s,              # Strong curling/wilting
            leaf_hue_shift=0.08 * s,        # Yellowing
            blossom_count=1.0 - 0.4 * s,    # Flower drop
            leaf_edge_damage=0.25 * s,      # Some scorching
        )


# =============================================================================
# QUADRATIC BEZIER
# =============================================================================

def quad_bezier_pts(
    a: np.ndarray,
    cpt: np.ndarray,
    b: np.ndarray,
    n: int
) -> list[np.ndarray]:
    """
    Sample n+1 points along a quadratic Bézier curve.

    Args:
        a: Start point
        cpt: Control point
        b: End point
        n: Number of segments (n+1 points returned)
    """
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

    The leaf is defined in local coordinates:
    - Origin at base (attachment point)
    - Y-axis points toward tip (negative Y in screen coords)
    - X-axis is perpendicular (width direction)

    Attributes:
        x, y: World position of leaf base
        length: Length from base to tip
        max_width: Maximum half-width of leaf
        sharpness: Controls pointedness (higher = sharper tip)
        rotation: Rotation angle in radians
        jitter: Amount of edge wobble (0 = smooth)
        jitter_seed: Seed for reproducible jitter
        outline_steps: Number of points for outline sampling
    """
    x: float
    y: float
    length: float = 200.0
    max_width: float = 70.0
    sharpness: float = 1.6
    rotation: float = 0.0
    jitter: float = 0.0
    jitter_seed: float = 0.0
    outline_steps: int = 120

    # Cached layout data
    _vein_cache: dict = field(default_factory=dict, repr=False)

    def width_at(self, t: float) -> float:
        """Width profile at parameter t ∈ [0, 1]."""
        # Use abs() to avoid complex numbers when t is slightly outside [0, 1]
        return self.max_width * (abs(math.sin(math.pi * t)) ** self.sharpness)

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
        """Midrib point in local coords."""
        return vec(0, self.y_at(t))

    def local_right(self, t: float) -> np.ndarray:
        """Right edge point in local coords."""
        return vec(+self.half_width_at(t), self.y_at(t))

    def local_left(self, t: float) -> np.ndarray:
        """Left edge point in local coords."""
        return vec(-self.half_width_at(t), self.y_at(t))

    def to_world(self, v: np.ndarray) -> np.ndarray:
        """Transform local coordinates to world coordinates."""
        c, s = math.cos(self.rotation), math.sin(self.rotation)
        return vec(
            self.x + (v[0] * c - v[1] * s),
            self.y + (v[0] * s + v[1] * c)
        )

    def sample_t(
        self,
        t_a: float,
        t_b: float,
        n: int,
        fn: Callable[[float], np.ndarray]
    ) -> list[np.ndarray]:
        """Sample n+1 points from fn between t_a and t_b."""
        pts = []
        for i in range(n + 1):
            t = t_a + (t_b - t_a) * i / n
            pts.append(fn(t))
        return pts

    # =========================================================================
    # EQUAL-AREA BREAKPOINTS
    # =========================================================================

    def equal_area_breaks(self, n: int, samples: int = 2000) -> list[float]:
        """
        Compute n+1 breakpoints that divide the leaf into n equal-area panels.

        Uses the integral of the half-width function.
        """
        # Build cumulative area array
        cum = [0.0] * (samples + 1)
        total = 0.0

        prev_t = 0.0
        prev_w = self.half_width_at(0)

        for i in range(1, samples + 1):
            t = i / samples
            w = self.half_width_at(t)
            # Trapezoidal integration
            total += 0.5 * (prev_w + w) * (t - prev_t)
            cum[i] = total
            prev_t = t
            prev_w = w

        # Inverse function: find t where cumulative area = target
        def inv(target: float) -> float:
            # Binary search
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

            # Linear interpolation for precision
            t0 = (lo - 1) / samples
            t1 = lo / samples
            a0 = cum[lo - 1]
            a1 = cum[lo]
            alpha = (target - a0) / max(1e-9, a1 - a0)
            return t0 + (t1 - t0) * max(0, min(1, alpha))

        # Compute breakpoints
        breaks = []
        for k in range(n + 1):
            breaks.append(inv((total * k) / n))

        return breaks

    # =========================================================================
    # VEIN LAYOUT
    # =========================================================================

    def compute_vein_layout(
        self,
        n_veins: int,
        angle_deg: float,
        eps: float = 1e-4
    ) -> dict:
        """
        Compute midrib anchors and margin hit points for veins.

        Args:
            n_veins: Number of interior veins
            angle_deg: Angle of veins from midrib (degrees)
            eps: Minimum spacing between margin points

        Returns:
            Dict with 'mid_all' and 'mar_all' lists of t-values
        """
        # n_veins veins => n_veins+1 panels => need n_veins interior anchors
        all_breaks = self.equal_area_breaks(n_veins + 1)
        mid_ts = all_breaks[1:-1]  # Interior breakpoints only

        theta = math.radians(max(45, min(89.5, angle_deg)))
        tan_theta = math.tan(theta)

        # Compute where each vein hits the margin
        margin_ts = []
        for t0 in mid_ts:
            w0 = self.half_width_at(t0)
            dt = w0 / (self.length * tan_theta)
            margin_ts.append(max(0, min(1, t0 + dt)))

        # Keep monotone so panels don't fold
        for i in range(1, len(margin_ts)):
            margin_ts[i] = max(margin_ts[i], margin_ts[i - 1] + eps)
            margin_ts[i] = min(margin_ts[i], 1.0)

        return {
            'mid_all': [0.0] + mid_ts + [1.0],
            'mar_all': [0.0] + margin_ts + [1.0]
        }

    # =========================================================================
    # VEIN POLYLINES
    # =========================================================================

    def vein_polyline_local(
        self,
        a: np.ndarray,
        b: np.ndarray,
        curve: float = 0.0,
        samples: int = 10
    ) -> list[np.ndarray]:
        """
        Build a polyline from midrib anchor to margin hit.

        Args:
            a: Start point (on midrib)
            b: End point (on margin)
            curve: Curvature amount (0 = straight)
            samples: Number of segments for curved vein
        """
        if curve <= 0:
            return [a.copy(), b.copy()]

        v = b - a
        mid = a + v * 0.55

        # Perpendicular vector
        perp = vec(-v[1], v[0])
        m = vec_mag(perp)
        if m > 1e-9:
            perp = perp / m

        # Bow "up" in local space (negative y)
        if perp[1] > 0:
            perp = -perp

        amp = vec_mag(v) * curve
        cpt = mid + perp * amp

        return quad_bezier_pts(a, cpt, b, samples)

    def build_vein_boundaries(
        self,
        n_veins: int,
        angle_deg: float = 75.0,
        curve: float = 0.0,
        vein_samples: int = 10
    ) -> dict:
        """
        Build vein boundary polylines for both sides.

        Returns dict with layout info and 'right'/'left' polyline lists.
        """
        layout = self.compute_vein_layout(n_veins, angle_deg)
        mid_all = layout['mid_all']
        mar_all = layout['mar_all']

        right = []
        left = []

        for k in range(len(mid_all)):
            tm = mid_all[k]
            te = mar_all[k]
            a = self.local_mid(tm)

            right.append(self.vein_polyline_local(
                a, self.local_right(te), curve, vein_samples
            ))
            left.append(self.vein_polyline_local(
                a, self.local_left(te), curve, vein_samples
            ))

        return {
            **layout,
            'right': right,
            'left': left
        }

    # =========================================================================
    # OUTLINE AND VEIN POINTS
    # =========================================================================

    def get_outline_points(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Get right and left edge points in world coordinates."""
        right = []
        left = []
        for i in range(self.outline_steps + 1):
            t = i / self.outline_steps
            right.append(self.to_world(self.local_right(t)))
            left.append(self.to_world(self.local_left(t)))
        return right, left

    def get_outline_polygon(self) -> list[np.ndarray]:
        """Get complete outline as a closed polygon in world coords."""
        right, left = self.get_outline_points()
        # Start at tip (t=1), go down right side, up left side
        outline = []
        for pt in reversed(right):
            outline.append(pt)
        for pt in left[1:]:  # Skip first (same as last of right reversed)
            outline.append(pt)
        return outline

    def get_midrib_points(self) -> list[np.ndarray]:
        """Get midrib line points in world coordinates."""
        pts = []
        for i in range(self.outline_steps + 1):
            t = i / self.outline_steps
            pts.append(self.to_world(self.local_mid(t)))
        return pts

    def mid_point_world(self, t: float) -> np.ndarray:
        """Get a single midrib point at parameter t in world coordinates."""
        return self.to_world(self.local_mid(t))

    def get_vein_lines(
        self,
        n_veins: int,
        angle_deg: float = 75.0,
        curve: float = 0.0,
        vein_samples: int = 10
    ) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
        """
        Get interior vein lines in world coordinates.

        Returns (right_veins, left_veins) where each is a list of polylines.
        """
        vb = self.build_vein_boundaries(n_veins, angle_deg, curve, vein_samples)

        right_veins = []
        left_veins = []

        # Interior veins only (skip boundary 0 and last)
        for k in range(1, len(vb['mid_all']) - 1):
            right_veins.append([self.to_world(p) for p in vb['right'][k]])
            left_veins.append([self.to_world(p) for p in vb['left'][k]])

        return right_veins, left_veins

    # =========================================================================
    # VEIN PANELS (POLYGONS)
    # =========================================================================

    def get_vein_panels(
        self,
        n_veins: int,
        angle_deg: float = 75.0,
        curve: float = 0.0,
        edge_samples: int = 12,
        vein_samples: int = 10
    ) -> dict:
        """
        Get vein panel polygons in world coordinates.

        Each panel is bounded by:
        - A segment of the midrib
        - Two veins (inner and outer boundaries)
        - A segment of the leaf margin

        Returns dict with 'right' and 'left' lists of polygon point lists,
        plus 'breaks' with the t-value breakpoints.
        """
        vb = self.build_vein_boundaries(n_veins, angle_deg, curve, vein_samples)
        mid_all = vb['mid_all']
        mar_all = vb['mar_all']

        right_panels = []
        left_panels = []

        for k in range(len(mid_all) - 1):
            tm0, tm1 = mid_all[k], mid_all[k + 1]
            te0, te1 = mar_all[k], mar_all[k + 1]

            m0 = self.local_mid(tm0)
            m1 = self.local_mid(tm1)

            v0r = vb['right'][k]
            v1r = vb['right'][k + 1]
            v0l = vb['left'][k]
            v1l = vb['left'][k + 1]

            # Margin segment between outer ends of adjacent veins
            margin_r = self.sample_t(te1, te0, edge_samples, self.local_right)
            margin_l = self.sample_t(te1, te0, edge_samples, self.local_left)

            # Panel polygon order:
            # midrib segment -> vein1 outward -> margin -> vein0 inward
            poly_r_local = (
                [m0, m1] +
                v1r[1:] +
                margin_r[1:] +
                list(reversed(v0r[:-1]))
            )

            poly_l_local = (
                [m0, m1] +
                v1l[1:] +
                margin_l[1:] +
                list(reversed(v0l[:-1]))
            )

            right_panels.append([self.to_world(p) for p in poly_r_local])
            left_panels.append([self.to_world(p) for p in poly_l_local])

        return {
            'right': right_panels,
            'left': left_panels,
            'breaks': {
                'mid': mid_all,
                'margin': mar_all
            }
        }


# =============================================================================
# RENDERING
# =============================================================================

@dataclass
class LeafStyle:
    """Style configuration for leaf rendering."""
    # Lead line colors and widths
    lead_color: str = '#141414'
    outline_width: float = 3.0
    midrib_width: float = 2.0
    vein_width: float = 1.5

    # Panel colors (right and left alternating)
    panel_colors_right: tuple = (
        '#5a9e5a', '#6ab06a', '#7ac27a', '#8ad48a', '#9ae69a'
    )
    panel_colors_left: tuple = (
        '#4a8e4a', '#5aa05a', '#6ab26a', '#7ac47a', '#8ad68a'
    )

    # Background
    background_color: str = '#f8f8f8'


def render_leaf(
    leaf: LeafGeom,
    n_veins: int = 5,
    angle_deg: float = 55.0,
    curve: float = 0.0,
    style: LeafStyle = None,
    ax: plt.Axes = None,
    figsize: tuple = (10, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Render a leaf with stained-glass vein panels.

    Args:
        leaf: LeafGeom object to render
        n_veins: Number of interior veins
        angle_deg: Vein angle from midrib
        curve: Vein curvature (0 = straight)
        style: LeafStyle configuration
        ax: Existing axes to draw on (creates new figure if None)
        figsize: Figure size if creating new figure

    Returns:
        (figure, axes) tuple
    """
    if style is None:
        style = LeafStyle()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(style.background_color)
    else:
        fig = ax.figure

    # Get panel polygons
    panels = leaf.get_vein_panels(
        n_veins,
        angle_deg=angle_deg,
        curve=curve,
        edge_samples=12,
        vein_samples=12
    )

    # --- Draw panels (fills first) ---
    for i, panel in enumerate(panels['right']):
        color = style.panel_colors_right[i % len(style.panel_colors_right)]
        pts = np.array([[p[0], p[1]] for p in panel])
        patch = MplPolygon(pts, facecolor=color, edgecolor='none')
        ax.add_patch(patch)

    for i, panel in enumerate(panels['left']):
        color = style.panel_colors_left[i % len(style.panel_colors_left)]
        pts = np.array([[p[0], p[1]] for p in panel])
        patch = MplPolygon(pts, facecolor=color, edgecolor='none')
        ax.add_patch(patch)

    # --- Draw strokes on top (lead lines) ---

    # Outline
    outline = leaf.get_outline_polygon()
    xs = [p[0] for p in outline] + [outline[0][0]]
    ys = [p[1] for p in outline] + [outline[0][1]]
    ax.plot(xs, ys, color=style.lead_color, linewidth=style.outline_width,
            solid_capstyle='round', solid_joinstyle='round')

    # Midrib
    midrib = leaf.get_midrib_points()
    xs = [p[0] for p in midrib]
    ys = [p[1] for p in midrib]
    ax.plot(xs, ys, color=style.lead_color, linewidth=style.midrib_width,
            solid_capstyle='round')

    # Veins
    right_veins, left_veins = leaf.get_vein_lines(
        n_veins, angle_deg=angle_deg, curve=curve, vein_samples=12
    )

    for vein in right_veins:
        xs = [p[0] for p in vein]
        ys = [p[1] for p in vein]
        ax.plot(xs, ys, color=style.lead_color, linewidth=style.vein_width,
                solid_capstyle='round')

    for vein in left_veins:
        xs = [p[0] for p in vein]
        ys = [p[1] for p in vein]
        ax.plot(xs, ys, color=style.lead_color, linewidth=style.vein_width,
                solid_capstyle='round')

    ax.set_aspect('equal')
    ax.axis('off')

    return fig, ax


def make_leaf(
    x: float,
    y: float,
    length: float = 200.0,
    max_width: float = 70.0,
    sharpness: float = 1.6,
    rotation: float = 0.0,
    jitter: float = 0.0,
    jitter_seed: float = 0.0,
    outline_steps: int = 120
) -> LeafGeom:
    """
    Convenience function to create a LeafGeom.

    Args:
        x, y: Base position in world coordinates
        length: Length from base to tip
        max_width: Maximum half-width
        sharpness: Pointedness (higher = sharper)
        rotation: Rotation in radians
        jitter: Edge wobble amount
        jitter_seed: Seed for reproducible jitter
        outline_steps: Sampling resolution
    """
    return LeafGeom(
        x=x, y=y,
        length=length,
        max_width=max_width,
        sharpness=sharpness,
        rotation=rotation,
        jitter=jitter,
        jitter_seed=jitter_seed,
        outline_steps=outline_steps
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def render_stained_glass(
    seed: int = 42,
    figsize: tuple = (10, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Render a stained-glass leaf.

    Args:
        seed: Random seed for jitter
        figsize: Figure size

    Returns:
        (figure, axes) tuple
    """
    # Create leaf geometry (tree-as-leaf concept)
    leaf = make_leaf(
        x=figsize[0] * 50 * 0.5,  # Center horizontally
        y=figsize[1] * 50 * 0.75,  # Lower third
        length=260,
        max_width=50,
        sharpness=1.2,
        rotation=-0.2,
        jitter=0.6,
        jitter_seed=seed
    )

    fig, ax = render_leaf(
        leaf,
        n_veins=5,
        angle_deg=55,
        curve=0.0,
        figsize=figsize
    )

    # Set axis limits
    margin = 50
    ax.set_xlim(0, figsize[0] * 50 + margin)
    ax.set_ylim(figsize[1] * 50 + margin, -margin)

    return fig, ax


def save_stained_glass(
    filepath: str,
    seed: int = 42,
    dpi: int = 150,
    figsize: tuple = (10, 8),
):
    """
    Render and save a stained-glass leaf.

    Args:
        filepath: Output path
        seed: Random seed
        dpi: Output resolution
        figsize: Figure size
    """
    fig, ax = render_stained_glass(seed=seed, figsize=figsize)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved to {filepath}")


# =============================================================================
# BLOSSOM GEOMETRY (RADIAL PETALS)
# =============================================================================

@dataclass
class BlossomGeom:
    """
    Blossom geometry with radial petals for stained-glass rendering.

    Each petal is like a leaf but arranged radially around a center point.
    The width profile maps to angular offset within each petal's wedge.

    Attributes:
        x, y: World position of blossom center
        petals: Number of petals
        base_radius: Distance from center to petal base
        petal_length: Length of each petal
        petal_max_width: Maximum width of petal (maps to angular span)
        sharpness: Controls petal pointedness
        spread: How much of wedge the petal fills (0-1, 1=exactly meets neighbors)
        rotation: Base rotation angle in radians
        jitter: Amount of edge wobble
        jitter_seed: Seed for reproducible jitter
        petal_steps: Sampling resolution for outlines
    """
    x: float
    y: float
    petals: int = 8
    base_radius: float = 18.0
    petal_length: float = 95.0
    petal_max_width: float = 55.0
    sharpness: float = 1.4
    spread: float = 0.98
    rotation: float = 0.0
    jitter: float = 0.0
    jitter_seed: float = 0.0
    petal_steps: int = 90

    def __post_init__(self):
        self._step = 2 * math.pi / self.petals
        self._half_span = (self._step * self.spread) * 0.5

    def width_at(self, t: float) -> float:
        """Width profile at parameter t ∈ [0, 1]."""
        # Use abs() to avoid complex numbers when t is slightly outside [0, 1]
        return self.petal_max_width * (abs(math.sin(math.pi * t)) ** self.sharpness)

    def wobble_at(self, k: int, t: float) -> float:
        """Jitter offset for petal k at parameter t."""
        if self.jitter <= 0:
            return 0.0
        # Vary per-petal and along t
        n = simple_noise(self.jitter_seed + k * 19.73, t * 3.1)
        return (n - 0.5) * self.jitter

    def half_width_at(self, k: int, t: float) -> float:
        """Half-width with jitter for petal k at t."""
        return max(0, self.width_at(t) + self.wobble_at(k, t))

    def r_at(self, t: float) -> float:
        """Radial distance at parameter t."""
        return self.base_radius + t * self.petal_length

    def ang_offset_at(self, k: int, t: float) -> float:
        """Angular offset for petal k at t (maps width to angle)."""
        hw = self.half_width_at(k, t)
        return self._half_span * (hw / max(1e-9, self.petal_max_width))

    def to_world_polar(self, ang: float, r: float) -> np.ndarray:
        """Convert polar (angle, radius) to world coordinates."""
        return vec(
            self.x + math.cos(ang) * r,
            self.y + math.sin(ang) * r
        )

    def edge_pt(self, k: int, t: float, side: int) -> np.ndarray:
        """
        Edge point of petal k at parameter t.

        Args:
            k: Petal index
            t: Parameter along petal (0=base, 1=tip)
            side: +1 for right edge, -1 for left edge
        """
        a0 = self.rotation + k * self._step
        a = a0 + side * self.ang_offset_at(k, t)
        return self.to_world_polar(a, self.r_at(t))

    def mid_pt(self, k: int, t: float) -> np.ndarray:
        """Midrib point of petal k at parameter t."""
        a0 = self.rotation + k * self._step
        return self.to_world_polar(a0, self.r_at(t))

    def sample_t(
        self,
        t_a: float,
        t_b: float,
        n: int,
        fn: Callable[[float], np.ndarray]
    ) -> list[np.ndarray]:
        """Sample n+1 points from fn between t_a and t_b."""
        pts = []
        for i in range(n + 1):
            t = t_a + (t_b - t_a) * i / n
            pts.append(fn(t))
        return pts

    # =========================================================================
    # EQUAL-AREA BREAKPOINTS
    # =========================================================================

    def equal_area_breaks(self, n: int, samples: int = 1200) -> list[float]:
        """
        Compute n+1 breakpoints that divide petals into n equal-area bands.

        Averages across petals so bands line up radially.
        """
        cum = [0.0] * (samples + 1)
        total = 0.0

        # Average half-width across all petals
        def hw_avg(t: float) -> float:
            s = sum(self.half_width_at(k, t) for k in range(self.petals))
            return s / self.petals

        prev_t = 0.0
        prev_w = hw_avg(0)

        for i in range(1, samples + 1):
            t = i / samples
            w = hw_avg(t)
            total += 0.5 * (prev_w + w) * (t - prev_t)
            cum[i] = total
            prev_t = t
            prev_w = w

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

            t0 = (lo - 1) / samples
            t1 = lo / samples
            a0 = cum[lo - 1]
            a1 = cum[lo]
            alpha = (target - a0) / max(1e-9, a1 - a0)
            return t0 + (t1 - t0) * max(0, min(1, alpha))

        breaks = []
        for k in range(n + 1):
            breaks.append(inv((total * k) / n))

        return breaks

    # =========================================================================
    # PETAL PANELS
    # =========================================================================

    def get_petal_panels(
        self,
        n_bands: int = 4,
        edge_samples: int = 18
    ) -> dict:
        """
        Get petal panel polygons in world coordinates.

        Each petal is divided into n_bands radial strips.

        Returns dict with 'petals' (list of list of polygon point lists)
        and 'breaks' (the t-value breakpoints).
        """
        breaks = self.equal_area_breaks(n_bands)

        by_petal = []
        for k in range(self.petals):
            bands = []
            for b in range(len(breaks) - 1):
                t0, t1 = breaks[b], breaks[b + 1]

                # Right edge from t0->t1
                right_seg = self.sample_t(
                    t0, t1, edge_samples,
                    lambda t, k=k: self.edge_pt(k, t, +1)
                )
                # Left edge from t1->t0 (reverse direction)
                left_seg = self.sample_t(
                    t1, t0, edge_samples,
                    lambda t, k=k: self.edge_pt(k, t, -1)
                )

                # Combine into closed polygon
                poly = right_seg + left_seg
                bands.append(poly)

            by_petal.append(bands)

        return {'petals': by_petal, 'breaks': breaks}

    # =========================================================================
    # OUTLINE AND LINE GEOMETRY
    # =========================================================================

    def get_petal_outline(self, k: int) -> list[np.ndarray]:
        """Get outline polygon for petal k in world coordinates."""
        right = []
        left = []

        for i in range(self.petal_steps + 1):
            t = i / self.petal_steps
            right.append(self.edge_pt(k, t, +1))
            left.append(self.edge_pt(k, t, -1))

        # Go up right edge, then back down left edge
        outline = right + list(reversed(left))
        return outline

    def get_all_outlines(self) -> list[list[np.ndarray]]:
        """Get outline polygons for all petals."""
        return [self.get_petal_outline(k) for k in range(self.petals)]

    def get_midrib_points(self, k: int) -> list[np.ndarray]:
        """Get midrib line points for petal k."""
        pts = []
        for i in range(self.petal_steps + 1):
            t = i / self.petal_steps
            pts.append(self.mid_pt(k, t))
        return pts

    def get_all_midribs(self) -> list[list[np.ndarray]]:
        """Get midrib lines for all petals."""
        return [self.get_midrib_points(k) for k in range(self.petals)]

    def get_cross_veins(
        self,
        n_bands: int = 4,
        edge_samples: int = 18
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Get cross-vein lines (at band boundaries).

        Returns list of (left_point, right_point) tuples.
        """
        breaks = self.equal_area_breaks(n_bands)
        veins = []

        for k in range(self.petals):
            # Skip base (0) and tip (last)
            for i in range(1, len(breaks) - 1):
                t = breaks[i]
                left = self.edge_pt(k, t, -1)
                right = self.edge_pt(k, t, +1)
                veins.append((left, right))

        return veins

    def get_center_disk(self, radius: float = None) -> list[np.ndarray]:
        """Get center disk polygon."""
        if radius is None:
            radius = self.base_radius * 0.92

        pts = []
        n = 40
        for i in range(n + 1):
            a = self.rotation + (2 * math.pi * i) / n
            pts.append(self.to_world_polar(a, radius))

        return pts


# =============================================================================
# BLOSSOM RENDERING
# =============================================================================

@dataclass
class BlossomStyle:
    """Style configuration for blossom rendering."""
    # Lead line colors and widths
    lead_color: str = '#141414'
    outline_width: float = 3.0
    midrib_width: float = 1.6
    cross_vein_width: float = 1.2

    # Panel colors (gradient from base to tip)
    panel_colors: tuple = (
        '#E8B4D4', '#D896C8', '#C878BC', '#B85AB0', '#A83CA4'
    )

    # Center disk
    center_color: str = '#EBD278'

    # Background
    background_color: str = '#f8f8f8'


def render_blossom(
    blossom: BlossomGeom,
    n_bands: int = 4,
    style: BlossomStyle = None,
    ax: plt.Axes = None,
    figsize: tuple = (10, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Render a blossom with stained-glass petal panels.

    Args:
        blossom: BlossomGeom object to render
        n_bands: Number of radial bands per petal
        style: BlossomStyle configuration
        ax: Existing axes to draw on (creates new figure if None)
        figsize: Figure size if creating new figure

    Returns:
        (figure, axes) tuple
    """
    if style is None:
        style = BlossomStyle()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(style.background_color)
    else:
        fig = ax.figure

    # Get panel polygons
    panels = blossom.get_petal_panels(n_bands, edge_samples=14)

    # --- Draw panels (fills first) ---
    for k, bands in enumerate(panels['petals']):
        for b, band in enumerate(bands):
            # Gentle variation per petal and band
            base_idx = (k + b) % len(style.panel_colors)
            color = style.panel_colors[base_idx]

            pts = np.array([[p[0], p[1]] for p in band])
            patch = MplPolygon(pts, facecolor=color, edgecolor='none')
            ax.add_patch(patch)

    # --- Center disk ---
    center = blossom.get_center_disk(blossom.base_radius * 0.85)
    pts = np.array([[p[0], p[1]] for p in center])
    patch = MplPolygon(
        pts, facecolor=style.center_color, edgecolor='none'
    )
    ax.add_patch(patch)

    # --- Lead lines on top ---

    # Petal outlines
    for outline in blossom.get_all_outlines():
        xs = [p[0] for p in outline] + [outline[0][0]]
        ys = [p[1] for p in outline] + [outline[0][1]]
        ax.plot(xs, ys, color=style.lead_color, linewidth=style.outline_width,
                solid_capstyle='round', solid_joinstyle='round')

    # Midribs
    for midrib in blossom.get_all_midribs():
        xs = [p[0] for p in midrib]
        ys = [p[1] for p in midrib]
        ax.plot(xs, ys, color=style.lead_color, linewidth=style.midrib_width,
                solid_capstyle='round')

    # Cross veins
    cross_veins = blossom.get_cross_veins(n_bands)
    for left, right in cross_veins:
        ax.plot([left[0], right[0]], [left[1], right[1]],
                color=style.lead_color, linewidth=style.cross_vein_width,
                solid_capstyle='round')

    ax.set_aspect('equal')
    ax.axis('off')

    return fig, ax


def make_blossom(
    x: float,
    y: float,
    petals: int = 8,
    base_radius: float = 18.0,
    petal_length: float = 95.0,
    petal_max_width: float = 55.0,
    sharpness: float = 1.4,
    spread: float = 0.98,
    rotation: float = 0.0,
    jitter: float = 0.0,
    jitter_seed: float = 0.0,
    petal_steps: int = 90
) -> BlossomGeom:
    """
    Convenience function to create a BlossomGeom.

    Args:
        x, y: Center position in world coordinates
        petals: Number of petals
        base_radius: Distance from center to petal base
        petal_length: Length of each petal
        petal_max_width: Maximum petal width (maps to angular span)
        sharpness: Petal pointedness (higher = sharper)
        spread: How much of wedge petal fills (0-1)
        rotation: Base rotation in radians
        jitter: Edge wobble amount
        jitter_seed: Seed for reproducible jitter
        petal_steps: Sampling resolution
    """
    return BlossomGeom(
        x=x, y=y,
        petals=petals,
        base_radius=base_radius,
        petal_length=petal_length,
        petal_max_width=petal_max_width,
        sharpness=sharpness,
        spread=spread,
        rotation=rotation,
        jitter=jitter,
        jitter_seed=jitter_seed,
        petal_steps=petal_steps
    )


# =============================================================================
# SHOOT GEOMETRY (CURVED STEM)
# =============================================================================

def cubic_bezier_point(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    t: float
) -> np.ndarray:
    """Evaluate cubic Bézier curve at parameter t."""
    u = 1 - t
    return (
        p0 * (u * u * u) +
        p1 * (3 * u * u * t) +
        p2 * (3 * u * t * t) +
        p3 * (t * t * t)
    )


def cubic_bezier_tangent(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    t: float
) -> np.ndarray:
    """Evaluate cubic Bézier tangent at parameter t."""
    u = 1 - t
    return (
        (p1 - p0) * (3 * u * u) +
        (p2 - p1) * (6 * u * t) +
        (p3 - p2) * (3 * t * t)
    )


@dataclass
class ShootGeom:
    """
    Curved shoot (stem) geometry using cubic Bézier curve.

    The shoot is a ribbon with variable width along its length.

    Attributes:
        p0, p1, p2, p3: Bézier control points
        steps: Sampling resolution
        w0, w1: Width at start and end
        w_jitter: Width jitter amount
        w_jitter_freq: Width jitter frequency
        jitter_seed: Seed for reproducible jitter
    """
    p0: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    p3: np.ndarray
    steps: int = 80
    w0: float = 18.0
    w1: float = 7.0
    w_jitter: float = 1.1
    w_jitter_freq: float = 2.1
    jitter_seed: float = 0.0

    # Cached curve data
    _pts: list = field(default_factory=list, repr=False)
    _tans: list = field(default_factory=list, repr=False)
    _norms: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._compute_curve()

    def _compute_curve(self):
        """Compute curve points, tangents, and normals."""
        self._pts = []
        self._tans = []
        self._norms = []

        for i in range(self.steps + 1):
            t = i / self.steps
            p = cubic_bezier_point(self.p0, self.p1, self.p2, self.p3, t)
            v = cubic_bezier_tangent(self.p0, self.p1, self.p2, self.p3, t)

            # Normalize tangent
            mag = vec_mag(v)
            if mag < 1e-9:
                tan = vec(0, -1)
            else:
                tan = v / mag

            # Left normal (perpendicular to tangent)
            norm = vec(-tan[1], tan[0])

            self._pts.append(p)
            self._tans.append(tan)
            self._norms.append(norm)

    @property
    def pts(self) -> list[np.ndarray]:
        return self._pts

    @property
    def tans(self) -> list[np.ndarray]:
        return self._tans

    @property
    def norms(self) -> list[np.ndarray]:
        return self._norms

    def width_at(self, t: float) -> float:
        """Get ribbon width at parameter t."""
        base = self.w0 + t * (self.w1 - self.w0)
        if self.w_jitter > 0:
            jitter = (simple_noise(self.jitter_seed + 1000, t * self.w_jitter_freq) - 0.5) * 2 * self.w_jitter
        else:
            jitter = 0
        return max(2, base + jitter)

    def point_at(self, t: float) -> np.ndarray:
        """Get point on curve at parameter t."""
        return cubic_bezier_point(self.p0, self.p1, self.p2, self.p3, t)

    def tangent_at(self, t: float) -> np.ndarray:
        """Get tangent at parameter t."""
        v = cubic_bezier_tangent(self.p0, self.p1, self.p2, self.p3, t)
        mag = vec_mag(v)
        if mag < 1e-9:
            return vec(0, -1)
        return v / mag

    def normal_at(self, t: float) -> np.ndarray:
        """Get left normal at parameter t."""
        tan = self.tangent_at(t)
        return vec(-tan[1], tan[0])

    def get_ribbon_polygon(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        Get ribbon as polygon.

        Returns (polygon, left_edge, right_edge) where polygon is closed
        and edges are for stroke drawing.
        """
        left = []
        right = []

        for i in range(len(self._pts)):
            t = i / self.steps
            w = self.width_at(t) * 0.5
            p = self._pts[i]
            n = self._norms[i]

            left.append(p + n * w)
            right.append(p - n * w)

        # Closed polygon: left edge forward, right edge backward
        poly = left + list(reversed(right))

        return poly, left, right


def make_shoot(
    p0: tuple,
    p1: tuple,
    p2: tuple,
    p3: tuple,
    steps: int = 80,
    w0: float = 18.0,
    w1: float = 7.0,
    w_jitter: float = 1.1,
    w_jitter_freq: float = 2.1,
    jitter_seed: float = 0.0
) -> ShootGeom:
    """Convenience function to create a ShootGeom."""
    return ShootGeom(
        p0=vec(*p0),
        p1=vec(*p1),
        p2=vec(*p2),
        p3=vec(*p3),
        steps=steps,
        w0=w0,
        w1=w1,
        w_jitter=w_jitter,
        w_jitter_freq=w_jitter_freq,
        jitter_seed=jitter_seed
    )


# =============================================================================
# COLLISION DETECTION
# =============================================================================

@dataclass
class CollisionCircle:
    """A circle used for collision detection."""
    center: np.ndarray
    radius: float


def circles_overlap(a: CollisionCircle, b: CollisionCircle) -> bool:
    """Check if two collision circles overlap."""
    dist = vec_mag(a.center - b.center)
    return dist < (a.radius + b.radius)


def circles_any_overlap(
    circles_a: list[CollisionCircle],
    circles_b: list[CollisionCircle]
) -> bool:
    """Check if any circle from A overlaps any circle from B."""
    for a in circles_a:
        for b in circles_b:
            if circles_overlap(a, b):
                return True
    return False


def circles_inside_bounds(
    circles: list[CollisionCircle],
    width: float,
    height: float,
    margin: float = 6.0
) -> bool:
    """Check if all circles are inside canvas bounds."""
    for c in circles:
        if (c.center[0] - c.radius < margin or
            c.center[1] - c.radius < margin or
            c.center[0] + c.radius > width - margin or
            c.center[1] + c.radius > height - margin):
            return False
    return True


def leaf_collider_circles(leaf: LeafGeom, pad: float = 10.0) -> list[CollisionCircle]:
    """
    Generate collision circles for a leaf.

    Uses capsule-like approach: circles along midrib with radii from half_width_at.
    """
    ts = [0.06, 0.18, 0.45, 0.72, 0.90]
    circles = []

    for t in ts:
        circles.append(CollisionCircle(
            center=leaf.mid_point_world(t),
            radius=leaf.half_width_at(t) + pad * (0.6 if t < 0.1 else 1.0)
        ))

    return circles


def blossom_collider_circles(
    center: np.ndarray,
    base_radius: float,
    petal_length: float,
    pad: float = 14.0
) -> list[CollisionCircle]:
    """
    Generate collision circle for a blossom.

    Blossoms are roughly circular, so one circle is sufficient.
    """
    return [CollisionCircle(
        center=center.copy(),
        radius=(base_radius + petal_length * 0.92) + pad
    )]


# =============================================================================
# PLACEMENT SOLVER
# =============================================================================

@dataclass
class PlacedElement:
    """A placed element (leaf or blossom) with collision info."""
    kind: str  # 'leaf' or 'blossom'
    data: dict
    circles: list[CollisionCircle]


def leaf_placements_along_shoot(shoot: ShootGeom, n: int) -> list[dict]:
    """Generate evenly-spaced anchor targets for leaves along shoot."""
    t0, t1 = 0.18, 0.92
    targets = []
    for i in range(n):
        u = (i + 0.5) / n
        t = t0 + u * (t1 - t0)
        targets.append({'t': t, 'idx': int(t * shoot.steps)})
    return targets


def blossom_placements_along_shoot(shoot: ShootGeom, n: int) -> list[dict]:
    """Generate anchor targets for blossoms along shoot (biased toward tip)."""
    t0, t1 = 0.52, 0.92
    targets = []
    for i in range(n):
        u = (i + 0.5) / n
        tt = u ** 0.65  # Bias toward tip
        t = t0 + tt * (t1 - t0)
        targets.append({'t': t, 'idx': int(t * shoot.steps)})
    return targets


def try_place_leaf(
    i: int,
    anchor: dict,
    shoot: ShootGeom,
    placed: list[PlacedElement],
    config: dict,
    canvas_size: tuple,
    rng: np.random.Generator
) -> PlacedElement | None:
    """
    Try to place a leaf at the given anchor with collision avoidance.

    Args:
        i: Leaf index
        anchor: Target anchor with 't' parameter
        shoot: The shoot geometry
        placed: Already placed elements
        config: Leaf configuration dict
        canvas_size: (width, height) tuple
        rng: Random number generator

    Returns:
        PlacedElement if successful, None otherwise
    """
    cfg = config
    t_base = anchor['t']
    width, height = canvas_size

    for attempt in range(80):
        # Jitter the t parameter
        tj = max(0.12, min(0.96, t_base + rng.normal() * 0.02))
        idx = int(tj * shoot.steps)
        idx = max(0, min(idx, len(shoot.pts) - 1))

        p = shoot.pts[idx]
        tan = shoot.tans[idx]
        n = shoot.norms[idx]

        # Alternate sides
        side = 1 if (i % 2 == 0) else -1
        if rng.random() < 0.16:
            side *= -1

        stem_half_w = shoot.width_at(tj) * 0.5
        attach_gap = cfg.get('attach_gap', 5)
        attach = p + n * side * (stem_half_w + attach_gap)

        # Direction: outward + slight forward bias
        outward = cfg.get('outward', 0.96)
        forward = cfg.get('forward', 0.26)
        direction = n * side * outward - tan * forward
        dir_mag = vec_mag(direction)
        if dir_mag > 1e-9:
            direction = direction / dir_mag

        rot = math.atan2(direction[0], -direction[1])
        rot += (rng.random() - 0.5) * 2 * cfg.get('rotation_jitter', 0.22)

        # Random leaf parameters
        length_range = cfg.get('length', [50, 100])
        width_range = cfg.get('max_width', [14, 22])
        sharp_range = cfg.get('sharpness', [1.1, 1.5])

        L = rng.uniform(length_range[0], length_range[1])
        W = rng.uniform(width_range[0], width_range[1])
        sharp = rng.uniform(sharp_range[0], sharp_range[1])

        leaf = LeafGeom(
            x=attach[0],
            y=attach[1],
            length=L,
            max_width=W,
            sharpness=sharp,
            rotation=rot,
            jitter=cfg.get('jitter', 0.75),
            jitter_seed=cfg.get('seed', 0) + 200 + i * 31 + attempt * 7,
            outline_steps=cfg.get('outline_steps', 140)
        )

        circles = leaf_collider_circles(leaf, cfg.get('collide_pad', 10))

        # Check bounds
        if not circles_inside_bounds(circles, width, height, 10):
            continue

        # Check overlaps
        ok = True
        for other in placed:
            if circles_any_overlap(circles, other.circles):
                ok = False
                break
        if not ok:
            continue

        return PlacedElement(
            kind='leaf',
            data={'leaf': leaf, 'i': i},
            circles=circles
        )

    return None


def try_place_blossom(
    i: int,
    anchor: dict,
    shoot: ShootGeom,
    placed: list[PlacedElement],
    config: dict,
    canvas_size: tuple,
    rng: np.random.Generator
) -> PlacedElement | None:
    """
    Try to place a blossom at the given anchor with collision avoidance.

    Args:
        i: Blossom index
        anchor: Target anchor with 't' parameter
        shoot: The shoot geometry
        placed: Already placed elements
        config: Blossom configuration dict
        canvas_size: (width, height) tuple
        rng: Random number generator

    Returns:
        PlacedElement if successful, None otherwise
    """
    cfg = config
    t_base = anchor['t']
    width, height = canvas_size

    for attempt in range(90):
        # Jitter the t parameter
        tj = max(0.35, min(0.98, t_base + rng.normal() * 0.02))
        idx = int(tj * shoot.steps)
        idx = max(0, min(idx, len(shoot.pts) - 1))

        p = shoot.pts[idx]
        tan = shoot.tans[idx]
        n = shoot.norms[idx]

        # Alternate sides
        side = -1 if (i % 2 == 0) else 1
        if rng.random() < 0.22:
            side *= -1

        stem_half_w = shoot.width_at(tj) * 0.5

        # Branch direction: outward + slightly up
        bdir = n * side * 1.0 - tan * 0.25
        bdir_mag = vec_mag(bdir)
        if bdir_mag > 1e-9:
            bdir = bdir / bdir_mag

        branch_len_range = cfg.get('branch_len', [30, 60])
        blen = rng.uniform(branch_len_range[0], branch_len_range[1])

        branch_base = p + n * side * (stem_half_w - 1)
        branch_tip = branch_base + bdir * blen

        # Random blossom parameters
        petals_range = cfg.get('petals', [9, 13])
        base_r_range = cfg.get('base_radius', [12, 18])
        petal_len_range = cfg.get('petal_length', [30, 50])
        petal_w_range = cfg.get('petal_max_width', [28, 42])
        sharp_range = cfg.get('sharpness', [1.2, 1.45])
        spread_range = cfg.get('spread', [0.96, 1.0])

        petals = int(rng.uniform(petals_range[0], petals_range[1] + 0.999))
        base_radius = rng.uniform(base_r_range[0], base_r_range[1])
        petal_length = rng.uniform(petal_len_range[0], petal_len_range[1])
        petal_max_width = rng.uniform(petal_w_range[0], petal_w_range[1])

        circles = blossom_collider_circles(
            branch_tip, base_radius, petal_length,
            cfg.get('collide_pad', 14)
        )

        # Check bounds
        if not circles_inside_bounds(circles, width, height, 10):
            continue

        # Check overlaps
        ok = True
        for other in placed:
            if circles_any_overlap(circles, other.circles):
                ok = False
                break
        if not ok:
            continue

        blossom = BlossomGeom(
            x=branch_tip[0],
            y=branch_tip[1],
            petals=petals,
            base_radius=base_radius,
            petal_length=petal_length,
            petal_max_width=petal_max_width,
            sharpness=rng.uniform(sharp_range[0], sharp_range[1]),
            spread=rng.uniform(spread_range[0], spread_range[1]),
            rotation=rng.random() * 2 * math.pi,
            jitter=cfg.get('jitter', 1.0),
            jitter_seed=cfg.get('seed', 0) + 9001 + i * 101 + attempt * 13,
            petal_steps=cfg.get('petal_steps', 80)
        )

        return PlacedElement(
            kind='blossom',
            data={
                'blossom': blossom,
                'branch_base': branch_base,
                'branch_tip': branch_tip,
                'i': i
            },
            circles=circles
        )

    return None


def place_elements_no_overlap(
    shoot: ShootGeom,
    n_leaves: int,
    n_blossoms: int,
    leaf_config: dict,
    blossom_config: dict,
    canvas_size: tuple,
    seed: int = 42
) -> list[PlacedElement]:
    """
    Place leaves and blossoms along shoot without overlap.

    Args:
        shoot: The shoot geometry
        n_leaves: Number of leaves to place
        n_blossoms: Number of blossoms to place
        leaf_config: Leaf configuration dict
        blossom_config: Blossom configuration dict
        canvas_size: (width, height) tuple
        seed: Random seed

    Returns:
        List of placed elements
    """
    rng = np.random.default_rng(seed)
    placed = []

    # Get anchor targets
    leaf_targets = leaf_placements_along_shoot(shoot, n_leaves)
    blossom_targets = blossom_placements_along_shoot(shoot, n_blossoms)

    # Place leaves first (larger, pack better along stem)
    for i, anchor in enumerate(leaf_targets):
        element = try_place_leaf(i, anchor, shoot, placed, leaf_config, canvas_size, rng)
        if element:
            placed.append(element)

    # Then place blossoms
    for i, anchor in enumerate(blossom_targets):
        element = try_place_blossom(i, anchor, shoot, placed, blossom_config, canvas_size, rng)
        if element:
            placed.append(element)

    return placed


# =============================================================================
# SHOOT COMPOSITION RENDERING
# =============================================================================

@dataclass
class ShootStyle:
    """Style configuration for shoot composition."""
    # Shoot ribbon
    shoot_fill: str = '#8CBE87'
    shoot_stroke: str = '#141414'
    shoot_stroke_width: float = 3.0

    # Branch ribbon (for blossoms)
    branch_fill: str = '#8CBE87'
    branch_stroke: str = '#141414'
    branch_stroke_width: float = 2.6
    branch_w0: float = 9.0
    branch_w1: float = 6.0

    # Leaf panel colors (gradient by band)
    leaf_colors: tuple = (
        '#5AA05A', '#6AB46A', '#7AC87A', '#8ADC8A', '#9AF09A'
    )

    # Blossom panel colors
    blossom_colors: tuple = (
        '#E8B4D4', '#D896C8', '#C878BC', '#B85AB0', '#A83CA4'
    )
    blossom_center: str = '#EBD278'

    # Lead lines
    lead_color: str = '#141414'
    outline_width: float = 3.0
    midrib_width: float = 2.0
    vein_width: float = 1.5
    cross_vein_width: float = 1.2
    centerline_width: float = 1.2

    # Background
    background_color: str = '#f8f8f8'

    def with_stress(self, stress: 'StressProfile') -> 'ShootStyle':
        """
        Create a new ShootStyle with stress-based color transformations applied.

        Args:
            stress: StressProfile to apply

        Returns:
            New ShootStyle with transformed colors
        """
        # Apply leaf color transformations
        stressed_leaf_colors = tuple(
            apply_color_stress(
                c,
                hue_shift=stress.leaf_hue_shift,
                saturation_mult=stress.leaf_saturation,
                value_mult=stress.leaf_value
            )
            for c in self.leaf_colors
        )

        # Apply blossom color transformations
        stressed_blossom_colors = tuple(
            apply_color_stress(
                c,
                hue_shift=stress.blossom_hue_shift,
                saturation_mult=stress.blossom_saturation,
                value_mult=1.0
            )
            for c in self.blossom_colors
        )

        # Apply stem color transformations (less affected, but some yellowing under stress)
        stressed_shoot_fill = apply_color_stress(
            self.shoot_fill,
            hue_shift=stress.leaf_hue_shift * 0.3,  # Stems yellow less than leaves
            saturation_mult=stress.leaf_saturation * 0.5 + 0.5,  # Less affected
            value_mult=stress.leaf_value
        )
        stressed_branch_fill = apply_color_stress(
            self.branch_fill,
            hue_shift=stress.leaf_hue_shift * 0.3,
            saturation_mult=stress.leaf_saturation * 0.5 + 0.5,
            value_mult=stress.leaf_value
        )

        return ShootStyle(
            shoot_fill=stressed_shoot_fill,
            shoot_stroke=self.shoot_stroke,
            shoot_stroke_width=self.shoot_stroke_width,
            branch_fill=stressed_branch_fill,
            branch_stroke=self.branch_stroke,
            branch_stroke_width=self.branch_stroke_width,
            branch_w0=self.branch_w0,
            branch_w1=self.branch_w1,
            leaf_colors=stressed_leaf_colors,
            blossom_colors=stressed_blossom_colors,
            blossom_center=self.blossom_center,
            lead_color=self.lead_color,
            outline_width=self.outline_width,
            midrib_width=self.midrib_width,
            vein_width=self.vein_width,
            cross_vein_width=self.cross_vein_width,
            centerline_width=self.centerline_width,
            background_color=self.background_color,
        )


def draw_branch_ribbon(
    ax: plt.Axes,
    base: np.ndarray,
    tip: np.ndarray,
    w0: float,
    w1: float,
    fill_color: str,
    stroke_color: str = None,
    stroke_width: float = 2.6
):
    """Draw a tapered branch ribbon."""
    v = tip - base
    m = vec_mag(v)
    if m < 1e-6:
        return

    t = v / m
    n = vec(-t[1], t[0])

    a1 = base + n * w0 * 0.5
    a2 = base - n * w0 * 0.5
    b1 = tip + n * w1 * 0.5
    b2 = tip - n * w1 * 0.5

    poly = np.array([a1, b1, b2, a2])
    patch = MplPolygon(poly, facecolor=fill_color, edgecolor='none')
    ax.add_patch(patch)

    if stroke_color:
        ax.plot([a1[0], b1[0]], [a1[1], b1[1]],
                color=stroke_color, linewidth=stroke_width,
                solid_capstyle='round')
        ax.plot([a2[0], b2[0]], [a2[1], b2[1]],
                color=stroke_color, linewidth=stroke_width,
                solid_capstyle='round')


def render_shoot_composition(
    shoot: ShootGeom,
    placed: list[PlacedElement],
    style: ShootStyle = None,
    leaf_config: dict = None,
    blossom_config: dict = None,
    ax: plt.Axes = None,
    figsize: tuple = (10, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Render a complete shoot composition with leaves and blossoms.

    Drawing order:
    1. Shoot ribbon fill
    2. Element fills (leaves, branches, blossoms)
    3. Element lead lines
    4. Shoot lead lines

    Args:
        shoot: ShootGeom object
        placed: List of PlacedElement objects
        style: ShootStyle configuration
        leaf_config: Leaf venation config
        blossom_config: Blossom band config
        ax: Existing axes (creates new figure if None)
        figsize: Figure size if creating new figure

    Returns:
        (figure, axes) tuple
    """
    if style is None:
        style = ShootStyle()
    if leaf_config is None:
        leaf_config = {'vein_count': 5, 'vein_angle': 55, 'vein_curve': 0.0}
    if blossom_config is None:
        blossom_config = {'bands': 4, 'center_disk': True, 'center_r': 14}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(style.background_color)
    else:
        fig = ax.figure

    # 1. SHOOT RIBBON FILL
    poly, left_edge, right_edge = shoot.get_ribbon_polygon()
    pts = np.array([[p[0], p[1]] for p in poly])
    patch = MplPolygon(pts, facecolor=style.shoot_fill, edgecolor='none')
    ax.add_patch(patch)

    # 2. ELEMENT FILLS
    for elem in placed:
        if elem.kind == 'leaf':
            leaf = elem.data['leaf']
            panels = leaf.get_vein_panels(
                leaf_config.get('vein_count', 5),
                angle_deg=leaf_config.get('vein_angle', 55),
                curve=leaf_config.get('vein_curve', 0.0),
                edge_samples=leaf_config.get('edge_samples', 12),
                vein_samples=leaf_config.get('vein_samples', 12)
            )

            for k, panel in enumerate(panels['right']):
                color = style.leaf_colors[k % len(style.leaf_colors)]
                pts = np.array([[p[0], p[1]] for p in panel])
                patch = MplPolygon(pts, facecolor=color, edgecolor='none')
                ax.add_patch(patch)

            for k, panel in enumerate(panels['left']):
                color = style.leaf_colors[k % len(style.leaf_colors)]
                pts = np.array([[p[0], p[1]] for p in panel])
                patch = MplPolygon(pts, facecolor=color, edgecolor='none')
                ax.add_patch(patch)

        else:  # blossom
            data = elem.data
            blossom = data['blossom']

            # Branch fill
            draw_branch_ribbon(
                ax, data['branch_base'], data['branch_tip'],
                style.branch_w0, style.branch_w1,
                style.branch_fill, stroke_color=None
            )

            # Blossom panels
            panels = blossom.get_petal_panels(
                blossom_config.get('bands', 4),
                edge_samples=blossom_config.get('edge_samples', 14)
            )

            for k, bands in enumerate(panels['petals']):
                for b, band in enumerate(bands):
                    color = style.blossom_colors[(k + b) % len(style.blossom_colors)]
                    pts = np.array([[p[0], p[1]] for p in band])
                    patch = MplPolygon(pts, facecolor=color, edgecolor='none')
                    ax.add_patch(patch)

            # Center disk
            if blossom_config.get('center_disk', True):
                center = blossom.get_center_disk(blossom_config.get('center_r', 14))
                pts = np.array([[p[0], p[1]] for p in center])
                patch = MplPolygon(pts, facecolor=style.blossom_center, edgecolor='none')
                ax.add_patch(patch)

    # 3. ELEMENT LEAD LINES
    for elem in placed:
        if elem.kind == 'leaf':
            leaf = elem.data['leaf']

            # Outline
            outline = leaf.get_outline_polygon()
            xs = [p[0] for p in outline] + [outline[0][0]]
            ys = [p[1] for p in outline] + [outline[0][1]]
            ax.plot(xs, ys, color=style.lead_color, linewidth=style.outline_width,
                    solid_capstyle='round', solid_joinstyle='round')

            # Midrib
            midrib = leaf.get_midrib_points()
            xs = [p[0] for p in midrib]
            ys = [p[1] for p in midrib]
            ax.plot(xs, ys, color=style.lead_color, linewidth=style.midrib_width,
                    solid_capstyle='round')

            # Veins
            right_veins, left_veins = leaf.get_vein_lines(
                leaf_config.get('vein_count', 5),
                angle_deg=leaf_config.get('vein_angle', 55),
                curve=leaf_config.get('vein_curve', 0.0),
                vein_samples=leaf_config.get('vein_samples', 12)
            )

            for vein in right_veins + left_veins:
                xs = [p[0] for p in vein]
                ys = [p[1] for p in vein]
                ax.plot(xs, ys, color=style.lead_color, linewidth=style.vein_width,
                        solid_capstyle='round')

        else:  # blossom
            data = elem.data
            blossom = data['blossom']

            # Branch strokes - stop at blossom edge, not center
            branch_base = data['branch_base']
            branch_tip = data['branch_tip']
            # Shorten to stop at blossom base_radius
            branch_vec = branch_tip - branch_base
            branch_len = np.linalg.norm(branch_vec)
            if branch_len > blossom.base_radius:
                # Stop at the edge of the blossom
                shortened_tip = branch_tip - (branch_vec / branch_len) * blossom.base_radius
            else:
                shortened_tip = branch_base  # Branch too short, skip stroke

            draw_branch_ribbon(
                ax, branch_base, shortened_tip,
                style.branch_w0, style.branch_w1,
                fill_color='none',
                stroke_color=style.lead_color,
                stroke_width=style.branch_stroke_width
            )

            # Petal outlines
            for outline in blossom.get_all_outlines():
                xs = [p[0] for p in outline] + [outline[0][0]]
                ys = [p[1] for p in outline] + [outline[0][1]]
                ax.plot(xs, ys, color=style.lead_color, linewidth=style.outline_width,
                        solid_capstyle='round', solid_joinstyle='round')

            # Midribs
            for midrib in blossom.get_all_midribs():
                xs = [p[0] for p in midrib]
                ys = [p[1] for p in midrib]
                ax.plot(xs, ys, color=style.lead_color, linewidth=style.midrib_width * 0.8,
                        solid_capstyle='round')

            # Cross veins
            cross_veins = blossom.get_cross_veins(blossom_config.get('bands', 4))
            for left, right in cross_veins:
                ax.plot([left[0], right[0]], [left[1], right[1]],
                        color=style.lead_color, linewidth=style.cross_vein_width,
                        solid_capstyle='round')

    # 4. SHOOT LEAD LINES
    # Ribbon edges
    for edge in [left_edge, right_edge]:
        xs = [p[0] for p in edge]
        ys = [p[1] for p in edge]
        ax.plot(xs, ys, color=style.shoot_stroke, linewidth=style.shoot_stroke_width,
                solid_capstyle='round')

    # Centerline
    xs = [p[0] for p in shoot.pts]
    ys = [p[1] for p in shoot.pts]
    ax.plot(xs, ys, color=style.lead_color, linewidth=style.centerline_width,
            solid_capstyle='round')

    ax.set_aspect('equal')
    ax.axis('off')

    return fig, ax


def render_shoot_scene(
    seed: int = 2524,
    canvas_size: tuple = (820, 620),
    n_leaves: int = 5,
    n_blossoms: int = 4,
    figsize: tuple = (10, 8),
    stress: StressProfile = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Render a complete shoot scene with leaves and blossoms.

    This is a convenience function that creates a shoot and places
    elements using the collision-based placement solver.

    Args:
        seed: Random seed for reproducibility
        canvas_size: (width, height) of the scene
        n_leaves: Number of leaves to place (before stress adjustment)
        n_blossoms: Number of blossoms to place (before stress adjustment)
        figsize: Figure size
        stress: Optional StressProfile to apply environmental stress effects

    Returns:
        (figure, axes) tuple

    Example:
        # Healthy plant
        fig, ax = render_shoot_scene(seed=42)

        # Drought-stressed plant
        fig, ax = render_shoot_scene(seed=42, stress=StressProfile.drought(0.7))

        # Composed stress - nitrogen-deficient drought
        stress = StressProfile.drought(0.6) * StressProfile.nitrogen_deficient(0.5)
        fig, ax = render_shoot_scene(seed=42, stress=stress)
    """
    if stress is None:
        stress = StressProfile.healthy()

    width, height = canvas_size

    # Apply stress to element counts
    effective_n_leaves = max(1, int(round(n_leaves * stress.leaf_count)))
    effective_n_blossoms = max(0, int(round(n_blossoms * stress.blossom_count)))

    # Base dimensions (will be scaled by stress)
    base_w0, base_w1 = 18, 7
    base_length_scale = 1.0

    # Apply stress to shoot dimensions
    stressed_w0 = base_w0 * stress.stem_width
    stressed_w1 = base_w1 * stress.stem_width

    # Create shoot geometry with stress-adjusted dimensions
    # Adjust control points to change apparent length
    length_mult = stress.stem_length
    shoot = make_shoot(
        p0=(width * 0.28, height * (0.93 - 0.1 * (1 - length_mult))),
        p1=(width * 0.24, height * (0.62 + 0.05 * (1 - length_mult))),
        p2=(width * 0.70, height * (0.50 - 0.05 * (1 - length_mult))),
        p3=(width * 0.63, height * (0.22 + 0.1 * (1 - length_mult))),
        steps=95,
        w0=stressed_w0,
        w1=stressed_w1,
        w_jitter=1.1,
        w_jitter_freq=2.1,
        jitter_seed=seed
    )

    # Base leaf dimensions
    base_leaf_length = [50, 100]
    base_leaf_width = [14, 22]
    base_leaf_sharpness = [1.1, 1.5]

    # Apply stress to leaf config
    leaf_config = {
        'seed': seed,
        'length': [l * stress.leaf_size for l in base_leaf_length],
        'max_width': [w * stress.leaf_size for w in base_leaf_width],
        'sharpness': [s * stress.leaf_sharpness for s in base_leaf_sharpness],
        'rotation_jitter': 0.22 + 0.15 * stress.leaf_curl,  # More jitter when curled
        'outward': 0.96,
        'forward': 0.26,
        'attach_gap': 5,
        'jitter': 0.75 + 0.5 * stress.leaf_curl,  # More wobble when stressed
        'outline_steps': 140,
        'vein_count': 5,
        'vein_angle': 55,
        'vein_curve': 0.0 + 0.3 * stress.leaf_curl,  # Curved veins when curled
        'edge_samples': 12,
        'vein_samples': 12,
        'collide_pad': 10 * stress.leaf_size  # Adjust collision padding
    }

    # Base blossom dimensions
    base_blossom_radius = [12, 18]
    base_petal_length = [30, 50]
    base_petal_width = [28, 42]

    # Apply stress to blossom config
    blossom_config = {
        'seed': seed,
        'petals': [9, 13],
        'base_radius': [r * stress.blossom_size for r in base_blossom_radius],
        'petal_length': [l * stress.blossom_size for l in base_petal_length],
        'petal_max_width': [w * stress.blossom_size for w in base_petal_width],
        'sharpness': [1.2, 1.45],
        'spread': [0.96, 1.0],
        'jitter': 1.0,
        'petal_steps': 80,
        'bands': 4,
        'edge_samples': 14,
        'branch_len': [30 * stress.stem_length, 60 * stress.stem_length],
        'collide_pad': 14 * stress.blossom_size,
        'center_disk': True,
        'center_r': 14 * stress.blossom_size
    }

    # Place elements with stress-adjusted counts
    placed = place_elements_no_overlap(
        shoot, effective_n_leaves, effective_n_blossoms,
        leaf_config, blossom_config,
        canvas_size, seed
    )

    # Create stressed style
    base_style = ShootStyle()
    stressed_style = base_style.with_stress(stress)

    # Render
    fig, ax = render_shoot_composition(
        shoot, placed,
        style=stressed_style,
        leaf_config=leaf_config,
        blossom_config=blossom_config,
        figsize=figsize
    )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    return fig, ax


def save_shoot_scene(
    filepath: str,
    seed: int = 2524,
    canvas_size: tuple = (820, 620),
    n_leaves: int = 5,
    n_blossoms: int = 4,
    dpi: int = 150,
    figsize: tuple = (10, 8),
    stress: StressProfile = None,
):
    """
    Render and save a shoot scene.

    Args:
        filepath: Output file path
        seed: Random seed for reproducibility
        canvas_size: (width, height) of the scene
        n_leaves: Number of leaves to place
        n_blossoms: Number of blossoms to place
        dpi: Output resolution
        figsize: Figure size
        stress: Optional StressProfile for environmental stress effects
    """
    fig, ax = render_shoot_scene(
        seed=seed,
        canvas_size=canvas_size,
        n_leaves=n_leaves,
        n_blossoms=n_blossoms,
        figsize=figsize,
        stress=stress
    )
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved to {filepath}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Geometry classes
    'BlossomGeom',
    'LeafGeom',
    'ShootGeom',
    'CollisionCircle',
    'PlacedElement',
    # Style classes
    'BlossomStyle',
    'LeafStyle',
    'ShootStyle',
    # Stress system
    'StressProfile',
    'apply_color_stress',
    # Factory functions
    'make_blossom',
    'make_leaf',
    'make_shoot',
    # Placement
    'place_elements_no_overlap',
    # Rendering
    'render_blossom',
    'render_leaf',
    'render_shoot_composition',
    'render_shoot_scene',
    'render_stained_glass',
    # Save utilities
    'save_shoot_scene',
    'save_stained_glass',
]
