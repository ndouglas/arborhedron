# Arborhedron

**Differentiable tree growth simulation under environmental stress**

A submission for the [Tesseract Hackathon 2025](https://pasteurlabs.ai/tesseract-hackathon-2025/) exploring morphogenetic neural cellular automata for growing tree-like structures.

![Arborhedron Stained Glass Tree](notebooks/stained_glass_hero.png)

## Vision

> *"Every tree is bonsai — a Platonic ideal geometric form shaped by environmental stress."*

Every tree in nature represents a perfect geometric form constrained by reality: soil irregularities, wind, asymmetric sunlight, drought, and genetic variation. This project uses differentiable simulation to explore how optimal morphogenetic rules adapt under perturbations, demonstrating the continuum between perfect form and adaptive resilience.

## Features

### Differentiable Growth Simulation

A complete tree growth simulator built with JAX, modeling:

- **Resource economics**: Energy, water, and nutrient flows
- **Structural dynamics**: Roots, trunk, shoots, leaves, flowers, and fruit
- **Environmental stress**: Light, moisture, and wind with sinusoidal variation
- **Biological constraints**: Transport bottlenecks, stomatal closure, self-shading

The simulation is fully differentiable, enabling gradient-based optimization of growth policies.

### Neural Growth Policies

Train neural networks to allocate resources optimally across different climate conditions:

- MLP-based allocation policies
- Features derived from tree state and environment
- Training via gradient descent on seed production

### Stained Glass Visualization

Beautiful L-system tree rendering with:

- Recursive branching structures
- Equal-area vein panel leaf geometry
- Blossoms and fruit
- Stress-responsive morphology (leaf color, density, form)

![Tree Gallery](notebooks/tree_gallery.png)

## Project Structure

```
arborhedron/
├── sim/                    # Core simulation module
│   ├── config.py           # Configuration dataclasses
│   ├── dynamics.py         # Growth step logic
│   ├── stress.py           # Environmental signal generation
│   ├── surrogates.py       # Biological surrogate functions
│   ├── policies.py         # Allocation policies (neural + baseline)
│   ├── rollout.py          # Full season simulation
│   ├── stained_glass.py    # L-system tree visualization
│   └── visualization.py    # Plotting utilities
├── notebooks/              # Jupyter notebooks
│   ├── 01_surrogates.ipynb         # Surrogate function exploration
│   ├── 02_rollout_baseline.ipynb   # Baseline policy testing
│   ├── 03A_gradient_optimization.ipynb  # Gradient-based training
│   ├── 03B_wind_trunk_experiment.ipynb  # Wind response analysis
│   ├── 03C_trunk_investigation.ipynb    # Structural dynamics
│   ├── 04_neural_policy.ipynb      # Neural policy training
│   ├── 05_robust_evaluation.ipynb  # Cross-climate evaluation
│   ├── 06_geometric_skeleton.ipynb # Stained glass rendering
│   └── 07_climate_morphology.ipynb # Stress-morphology mapping
├── tests/                  # Test suite
└── tesseracts/             # Tesseract definitions (for deployment)
```

## Installation

### Prerequisites

- Python 3.10+
- Docker (for Tesseract builds)

### Setup

```bash
# Clone the repository
git clone https://github.com/ndouglas/arborhedron.git
cd arborhedron

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run a simulation

```python
from sim import SimConfig, ClimateConfig, TreeState, run_season

# Configure simulation
config = SimConfig()
climate = ClimateConfig.mild()
initial_state = TreeState.seedling()

# Run a growing season
trajectory = run_season(initial_state, climate, config)
print(f"Final flowers: {trajectory.states[-1].flowers:.2f}")
print(f"Seeds produced: {trajectory.seeds:.2f}")
```

### Train a neural policy

```python
from sim import NeuralPolicy, make_neural_policy_fn
import jax.random as jr

# Initialize policy
key = jr.PRNGKey(42)
policy = NeuralPolicy.init(key, hidden_size=32)

# Create allocation function
allocate = make_neural_policy_fn(policy)

# Use in simulation
trajectory = run_season(initial_state, climate, config, policy_fn=allocate)
```

### Render a stained glass tree

```python
from sim import generate_tree_skeleton, render_tree, TreeParams, TreeStyle

# Generate tree structure
params = TreeParams(depth=4, branch_angle=0.4)
skeleton = generate_tree_skeleton(params, seed=42)

# Render with style
style = TreeStyle()
fig = render_tree(skeleton, style)
fig.savefig("my_tree.png", dpi=150)
```

## Key Concepts

### Growth Dynamics

Each simulation step models:

1. **Root uptake** — Water and nutrients from soil
2. **Transport** — Trunk limits resource delivery (bottleneck)
3. **Photosynthesis** — Energy production with self-shading
4. **Stomatal regulation** — Water conservation under drought
5. **Maintenance** — Costs proportional to biomass
6. **Allocation** — Policy decides investment distribution
7. **Growth** — New biomass from invested energy
8. **Damage** — Wind, drought stress on tender tissues
9. **Reproduction** — Flowers convert to fruit under maturity

### Environmental Stress

Three stress signals vary sinusoidally over the growing season:

- **Light** — Solar availability for photosynthesis
- **Moisture** — Soil water for uptake (inverted-U response)
- **Wind** — Mechanical stress on shoots, leaves, flowers

### Stabilization Mechanisms

The simulation includes biologically-motivated constraints:

- **Self-shading** (Beer-Lambert law) prevents runaway leaf growth
- **Transport bottleneck** requires trunk investment
- **Stomatal closure** conserves water under drought
- **Leaf crowding** requires shoot scaffolding
- **Flower gating** prevents premature reproduction

## Resources

- [Tesseract Core Documentation](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/)
- [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax)
- [Tesseract User Forums](https://si-tesseract.discourse.group/)
- [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) — Inspiration

## License

Apache License 2.0 — See [LICENSE](LICENSE) for details.
