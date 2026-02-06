# GNN-based Online Conformance Checking

## Overview

A Graph Neural Network model that predicts next transitions in business processes and validates them against Petri net rules (of a specific process model) in real-time, enabling early deviation detection during process execution.

## Architecture

**Two-Stage Design**

1. **Transition Predictor**: Heterogeneous GNN processes the Petri net graph structure to predict probability distribution over next transitions
2. **Sequential Conformance Checker**: Validates predictions by simulating Petri net execution step-by-step, checking if each transition is enabled given the current marking

## Key Design Choices

**Why Sequential Checking?**
Examining transitions one-by-one without seeing the full enablement vector forces the model to learn actual Petri net semantics (pre-conditions, post-conditions, state transitions) rather than just comparing predicted vs. enabled vectors.

**Why GNNs?**
Petri nets are naturally heterogeneous graphs. Multi-layer message passing captures both local patterns and long-range dependencies between places and transitions.

**Multi-task Loss**
- Transition prediction: Binary cross-entropy per transition
- Conformance classification: Binary validity of entire sequence
- Equal weighting (1.0 each) balances both objectives

## Data Generation

- Simulates conformant traces over N1 Petri net (7 places, 8 transitions) (example from paper : Online conformance checking relating event streams)
- Introduces controlled deviations: skips, insertions, swaps, wrong activities
- Sliding window approach creates samples with marking, activity history, and labels
- 30% deviation ratio for balanced training

## Model Features

- **Place features**: Token counts
- **Transition features**: Enablement flag + activity one-hot encoding
- **Prefix encoding**: Recent activity sequence (window size: 3)
- **Graph structure**: Pre/post arcs defining Petri net topology
