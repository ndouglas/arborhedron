# Arborhedron

[![CI](https://github.com/ndouglas/arborhedron/actions/workflows/ci.yml/badge.svg)](https://github.com/ndouglas/arborhedron/actions/workflows/ci.yml)
[![Tesseract Build](https://github.com/ndouglas/arborhedron/actions/workflows/tesseract-build.yml/badge.svg)](https://github.com/ndouglas/arborhedron/actions/workflows/tesseract-build.yml)

**Differentiable tree growth simulation for climate-resilient reforestation planning**

A submission for the [Tesseract Hackathon 2025](https://pasteurlabs.ai/tesseract-hackathon-2025/).

![Arborhedron Stained Glass Tree](notebooks/stained_glass_hero.png)

## What This Is

A fully differentiable simulation of tree growth that enables gradient-based optimization of resource allocation strategies for climate objectives. The inverse problem: *given environmental conditions, what allocation policy maximizes carbon sequestration while maintaining survival under stress?*

Key capabilities:

- **Tree growth dynamics**: Energy, water, nutrients, and biomass (roots, trunk, shoots, leaves, flowers, fruit) evolve over ~100 days under environmental stress
- **Neural allocation policy**: An MLP learns to allocate resources, optimized via gradient descent
- **Carbon sequestration metrics**: Permanence-weighted carbon scoring that reflects long-term climate impact
- **Resilience analysis**: Gradient-based sensitivity reveals environmental tipping points where survival collapses
- **Tesseract composition**: Modular, deployable ML pipeline components

## Climate Relevance

Trees are critical for climate mitigation (carbon sequestration) and adaptation (ecosystem resilience). Effective reforestation requires understanding:

1. **Carbon tradeoffs**: Trunk wood stores carbon for decades; leaves decompose within a year. How should trees allocate resources to maximize durable sequestration?

2. **Tipping points**: Where do small environmental changes cause disproportionate fitness loss? What are the critical moisture/wind thresholds?

3. **Policy robustness**: How do allocation strategies perform across climate scenarios?

Arborhedron addresses these questions with a differentiable framework that exposes gradients for both optimization and sensitivity analysis.

## Features

### Differentiable Growth Dynamics

Built with JAX, modeling:

- Resource economics (energy, water, nutrient flows)
- Structural constraints (transport bottlenecks, self-shading)
- Environmental response (stomatal closure, wind damage)
- Reproduction (flowering, fruiting, seed production)

### Neural Allocation Policy

An MLP that observes tree state + environment and outputs resource allocation fractions. Trainable via gradient descent on seeds, carbon, or combined objectives.

### Carbon Sequestration Metrics

Biomass is converted to carbon content using tissue-specific fractions, then weighted by permanence:

| Tissue | Carbon Fraction | Permanence |
|--------|-----------------|------------|
| Trunk | 0.50 | 1.0 |
| Roots | 0.45 | 0.7 |
| Shoots | 0.45 | 0.3 |
| Leaves | 0.45 | 0.1 |
| Flowers | 0.40 | 0.05 |

The permanence-weighted carbon score rewards durable carbon storage (trunk wood persists for decades; leaves decompose within a year).

### Resilience & Tipping Point Analysis

Tools for identifying critical environmental thresholds:

- **2D parameter sweeps**: Map fitness across moisture × wind space
- **Gradient sensitivity**: ∂fitness/∂param reveals where small changes cause large effects
- **Tipping point detection**: Locate where gradients spike or fitness collapses
- **Resilience boundaries**: Contours separating viable from non-viable regions

### Tesseract Composition

Three Tesseracts that compose into a differentiable pipeline:

```
neural_policy → growth_step → seed_production
     ↑              │
     └──────────────┘ (loop N days)
```

### Visualization

L-system tree rendering with stained-glass style leaves and blossoms.

![Tree Gallery](notebooks/tree_gallery.png)

## Installation

```bash
git clone https://github.com/ndouglas/arborhedron.git
cd arborhedron
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build Tesseracts (requires Docker)
./buildall.sh
```

## Usage

### Run the Tesseract pipeline

```bash
python main.py
```

### Run the simulation directly

```python
from sim import SimConfig, ClimateConfig, TreeState, run_season

config = SimConfig()
climate = ClimateConfig.mild()
initial_state = TreeState.initial()

trajectory = run_season(initial_state, climate, config)
print(f"Seeds produced: {trajectory.seeds:.2f}")
```

### Render a tree

```python
from sim import generate_tree_skeleton, render_tree, TreeParams, TreeStyle

params = TreeParams(depth=4)
skeleton = generate_tree_skeleton(params, seed=42)
fig = render_tree(skeleton, TreeStyle())
fig.savefig("tree.png")
```

## Project Structure

```
arborhedron/
├── sim/                    # Core simulation
│   ├── config.py           # State and config definitions
│   ├── dynamics.py         # Growth step logic
│   ├── surrogates.py       # Biological response functions
│   ├── policies.py         # Allocation policies
│   ├── rollout.py          # Season simulation
│   ├── carbon.py           # Carbon sequestration metrics
│   ├── resilience.py       # Tipping point analysis
│   └── stained_glass.py    # Tree visualization
├── tesseracts/             # Tesseract definitions
│   ├── growth_step/        # Single-day dynamics
│   ├── neural_policy/      # Allocation policy
│   └── seed_production/    # Fitness computation
├── notebooks/              # Exploration notebooks
├── docs/                   # Technical documentation
│   └── technical_writeup.md # Hackathon submission writeup
├── tests/                  # Test suite
└── main.py                 # Tesseract composition demo
```

## Resources

- [Tesseract Core](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/)
- [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax)

## License

Apache License 2.0
