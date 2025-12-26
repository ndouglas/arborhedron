# Arborhedron Tesseracts

Differentiable components for tree growth simulation, designed for composition via Tesseract-JAX.

## Tesseracts

### growth_step

Single-day tree growth simulation step.

**Inputs:**
- `state`: Tree state (10 compartments: energy, water, nutrients, roots, trunk, shoots, leaves, flowers, fruit, soil_water)
- `allocation`: Resource allocation fractions (roots, trunk, shoots, leaves, flowers)
- `light`, `moisture`, `wind`: Environmental conditions [0, 1]
- `day`, `num_days`: Timing information

**Outputs:**
- `state`: New tree state after one day

**Differentiable w.r.t.:** state, allocation, light, moisture, wind

### neural_policy

Neural network policy for allocation decisions.

**Inputs:**
- `state`: Tree state (8 compartments visible to policy)
- `light`, `moisture`, `wind`: Environmental conditions
- `day`, `num_days`: Timing information
- `weights`: Neural network weights (w1, b1, w2, b2)

**Outputs:**
- `allocation`: Resource allocation fractions (sum to 1)
- `logits`: Raw logits before softmax

**Differentiable w.r.t.:** state, environment, weights

### seed_production

Fitness function: convert accumulated fruit to seeds.

**Inputs:**
- `fruit_integral`: Sum of fruit biomass over season
- `final_energy`: Energy level at end of season
- `num_days`, `seed_energy_threshold`, `seed_conversion`: Configuration

**Outputs:**
- `seeds`: Number of seeds produced
- `energy_gate`: Viability gate [0, 1]
- `normalized_fruit`: Fruit integral / num_days

**Differentiable w.r.t.:** fruit_integral, final_energy

## Building

```bash
./buildall.sh
```

## Usage

See `main.py` for composition examples using both Tesseract SDK and Tesseract-JAX.
