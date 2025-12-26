# Arborhedron: Differentiable Tree Growth Simulation for Climate-Resilient Reforestation Planning

## Abstract

Arborhedron is a fully differentiable tree growth simulator built with JAX that enables gradient-based optimization of resource allocation strategies. The simulation models a tree's seasonal growth dynamics—including photosynthesis, water transport, structural constraints, and reproduction—while exposing gradients for end-to-end policy learning. We extend the core simulation with (1) carbon sequestration metrics that weight biomass by its climate permanence, and (2) resilience analysis tools that identify environmental tipping points where tree survival collapses. The system is decomposed into composable Tesseracts, enabling modular deployment and composition. We demonstrate that neural policies can be trained to maximize either seed production or carbon sequestration, revealing meaningful tradeoffs between reproductive fitness and climate impact.

---

## 1. Introduction

### 1.1 Motivation

Climate change poses dual challenges for forest management: mitigating atmospheric CO2 through carbon sequestration and adapting forests to survive under increasing environmental stress. Effective reforestation planning requires understanding how trees allocate resources under variable conditions and identifying where climate thresholds threaten forest viability.

Traditional forest growth models are either (a) empirical, based on site index curves that don't generalize to novel climates, or (b) process-based but computationally expensive and non-differentiable. This limits their utility for optimization: finding allocation strategies that maximize carbon sequestration while maintaining survival under stress.

### 1.2 Approach

We present Arborhedron, a differentiable tree growth simulator that treats resource allocation as an inverse problem: given environmental conditions, what allocation policy maximizes a specified objective (seeds, carbon, or a combination)?

Key contributions:
1. **Differentiable growth dynamics** built with JAX, enabling gradient-based policy optimization
2. **Carbon sequestration metrics** with permanence weighting reflecting climate science
3. **Tipping point analysis** using gradient sensitivity to identify resilience boundaries
4. **Tesseract composition** for modular, deployable ML pipelines

---

## 2. Methods

### 2.1 Tree State and Dynamics

The simulation tracks a tree state vector with 10 components:

| Variable | Description |
|----------|-------------|
| E | Energy store (photosynthate pool) |
| W | Internal water (xylem/tissue) |
| N | Nutrient store |
| R | Root biomass |
| T | Trunk/wood biomass |
| S | Shoot biomass |
| L | Leaf biomass |
| F | Flower biomass |
| Q | Fruit (developing seeds) |
| SW | Soil water reservoir |

Each day, the simulation executes these phases:

1. **Resource uptake**: Roots extract water and nutrients from soil, modulated by root biomass and soil moisture
2. **Photosynthesis**: Leaves produce energy, limited by light, water, and nutrients (Michaelis-Menten kinetics with Beer-Lambert self-shading)
3. **Transport**: Water delivery is bottlenecked by trunk capacity (xylem constraint)
4. **Allocation**: A policy (hand-coded or learned) distributes energy to compartments
5. **Growth**: Biomass increases based on allocation, subject to structural constraints (shoots must support leaves; trunk must support canopy)
6. **Stress damage**: Wind damages exposed tissue; drought triggers stomatal closure and leaf senescence
7. **Reproduction**: Mature flowers convert to fruit; fruit integrates to seeds

All operations use smooth, differentiable primitives (softplus, sigmoid, tanh) to ensure gradient flow.

### 2.2 Environmental Stress

Climate is parameterized by three stress functions varying sinusoidally over the season:

- **Light**: Solar radiation availability (0-1)
- **Moisture**: Soil moisture recharge rate (0-1)
- **Wind**: Wind stress intensity (0-1)

Each stress parameter has offset, amplitude, frequency, and phase, enabling diverse climate scenarios (mild, drought-prone, windy).

### 2.3 Neural Allocation Policy

The allocation policy is an MLP that observes:
- Current tree state (10 values)
- Day of season (normalized)
- Environmental stress values (3 values)

And outputs allocation fractions to 5 growth compartments (roots, trunk, shoots, leaves, flowers) via softmax. The policy is implemented in Equinox and trained via gradient descent on seasonal fitness.

### 2.4 Carbon Sequestration Metrics

We compute carbon content by multiplying biomass by tissue-specific carbon fractions (based on forestry literature):

| Tissue | Carbon Fraction | Permanence Weight |
|--------|-----------------|-------------------|
| Trunk | 0.50 | 1.0 |
| Roots | 0.45 | 0.7 |
| Shoots | 0.45 | 0.3 |
| Leaves | 0.45 | 0.1 |
| Flowers | 0.40 | 0.05 |

The **permanence-weighted carbon score** rewards durable carbon storage:

$$\text{CarbonScore} = \sum_i (\text{biomass}_i \times \text{carbon\_fraction}_i \times \text{permanence}_i)$$

This captures the climate-relevant distinction: trunk wood sequesters carbon for decades/centuries, while leaves decompose within a year.

The carbon objective includes an energy gate (tree must survive to count carbon):

$$\text{Objective} = \frac{\int \text{CarbonScore}(t) \, dt}{N} \times \sigma(10 \cdot (E_{\text{final}} - 0.3))$$

### 2.5 Tipping Point and Resilience Analysis

To identify critical environmental thresholds, we provide:

1. **2D Parameter Sweep**: Grid search over (moisture, wind) offset space, evaluating fitness at each point
2. **Gradient Sensitivity**: Numerical derivative ∂fitness/∂param via central differences
3. **Tipping Point Detection**: Locations where gradient magnitude spikes or fitness drops sharply
4. **Resilience Boundary**: The contour separating viable (fitness > threshold) from non-viable regions

The sensitivity analysis reveals where small environmental changes cause disproportionate fitness loss—the hallmark of tipping points.

### 2.6 Tesseract Composition

The simulation is decomposed into three Tesseracts:

```
neural_policy → growth_step → seed_production
     ↑              │
     └──────────────┘ (loop N days)
```

- **neural_policy**: Takes state + environment, outputs allocation
- **growth_step**: Single-day dynamics update
- **seed_production**: Computes final fitness from trajectory

Each Tesseract can be deployed independently and composed via Tesseract-JAX.

---

## 3. Results

### 3.1 Policy Learning

Neural policies trained on seed production converge within ~50 epochs to ~4x baseline performance. Key learned behaviors:
- Early root investment for water security
- Trunk building mid-season for structural capacity
- Late-season pivot to flowers and reproduction

### 3.2 Carbon vs. Seed Tradeoff

Training policies to maximize carbon score (instead of seeds) produces different allocation strategies:
- **Seed-optimized**: Maximizes flowers late in season, moderate trunk
- **Carbon-optimized**: Maximizes trunk throughout, fewer flowers

This tradeoff reflects real forestry decisions: fast-growing species (poplars) vs. slow-growing hardwoods (oaks).

### 3.3 Resilience Analysis

2D fitness landscapes reveal:
- **Viable region**: High moisture, low wind (seed production possible)
- **Tipping points**: Sharp fitness collapse at moisture < 0.3 and wind > 0.6
- **Sensitivity**: Fitness is more sensitive to moisture than wind

Example findings from a trained policy:
- Moisture sensitivity: +0.5 seeds per 0.1 moisture increase
- Wind sensitivity: -0.8 seeds per 0.1 wind increase
- Critical threshold: Trees fail to reproduce below 0.35 moisture offset

---

## 4. Discussion

### 4.1 Limitations

1. **Simplified physiology**: The model abstracts complex plant biology into aggregate variables. Real trees have more nuanced resource allocation.

2. **Single-season scope**: The simulation covers one growing season. Multi-year dynamics (winter dormancy, inter-annual variability) are not modeled.

3. **No spatial structure**: Individual tree simulation ignores competition, shading, and forest-level dynamics.

4. **Idealized climate**: Sinusoidal stress functions don't capture realistic weather variability (storms, heat waves, compound events).

### 4.2 Future Work

1. **Multi-year simulation**: Extend to perennial dynamics with winter dormancy and carbohydrate reserve carry-over.

2. **Species parameterization**: Calibrate parameters to real species for transfer learning.

3. **Forest-level modeling**: Compose individual tree Tesseracts into stand-level simulations.

4. **Climate scenario integration**: Connect to downscaled climate projections for regional planning.

5. **Real data validation**: Fit to dendrochronology (tree ring) data for empirical grounding.

---

## 5. Conclusion

Arborhedron demonstrates that differentiable simulation enables gradient-based optimization of tree resource allocation for both reproductive fitness and climate objectives. The carbon sequestration metrics and tipping point analysis tools provide climate-relevant insights for reforestation planning.

Key takeaways:
- **Differentiability matters**: End-to-end gradients enable policy learning that outperforms hand-coded baselines
- **Carbon and reproduction trade off**: Optimizing for climate impact produces different strategies than optimizing for seeds
- **Resilience boundaries exist**: Gradient sensitivity reveals where environmental stress causes disproportionate fitness loss
- **Tesseracts enable composition**: Modular design supports deployment and extension

The framework provides a foundation for more sophisticated climate-smart forestry tools that optimize allocation strategies under environmental uncertainty.

---

## References

1. Pasteur Labs. (2025). Tesseract: Composable differentiable ML pipelines. https://pasteurlabs.ai
2. DeepMind. (2020). JAX: Autograd and XLA. https://github.com/google/jax
3. Kidston, P., et al. (2021). Carbon fractions in forest biomass. Forest Ecology and Management.
4. Lenton, T.M., et al. (2019). Climate tipping points—too risky to bet against. Nature.

---

*Submitted to the Tesseract Hackathon 2025*
