# GNN-Based Sequential Conformance Checking

## core-idea
A two-stage neural + symbolic architecture for **online conformance checking** that:
1. Learns a latent process state using a **Graph Neural Network (GNN)**.
2. Enforces **Petri net semantics** using a sequential verifier.

This reframes conformance from *exact optimization* to *approximate neural inference with structural constraints*.

---

## architecture Two-Stage Predict–Then–Verify

### Stage 1 — Prediction
**Input:**  
- Heterogeneous Petri net graph  
- Current marking (tokens)  
- Prefix encoding of recent activities  

**Output:**  
Multi-label prediction of candidate next transitions.

This is a **neural orientation step**: it estimates where the case is in the process.

---

### Stage 2 — Verification
Each predicted transition is:
1. Checked for **enablement** using Petri net firing rules.
2. Fired only if legal.
3. Penalized if illegal.

**Conformance score** is accumulated step-by-step.

> This prevents the model from cheating by pre-comparing all enablement vectors.  
> The model must learn *process dynamics*, not shortcuts.

---

## gnn-design Heterogeneous Graph Convolution

### Node Types
- **Place nodes:** token counts (marking)
- **Transition nodes:** one-hot activity labels

### Edges
- Place → Transition (input arcs)
- Transition → Place (output arcs)

### Message Passing
3 bidirectional layers:
places → transitions → places → transitions → places

This enables **3-hop dependency capture**, allowing the model to learn:
> which transitions become enabled *and in which order* from distributed token patterns.

---

## state-of-the-art Comparison with HMMConf (Lee et al., 2021)

| Aspect | HMMConf | GNN Method |
|--------|--------|-----------|
| State | Probabilistic marking distribution | Deterministic marking + embeddings |
| Orientation | HMM forward recursion (α) | GNN message passing |
| Conformance | Σ P(m_i) × confmat[a, m_i] | Neural classifier + enablement checking |
| Non-conformance | EM learns deviations | Supervised penalty learning |
| Complexity | O(|Z|²) | O(E × L × D²) |

**Critical distinction:**  
- HMMConf = **Bayesian inference**  
- GNN = **Supervised neural classification with constraints**

---

## state-of-the-art Relation to Prefix Alignments

Prefix alignments (van Zelst et al., 2017):

| Prefix Alignments | This Method |
|------------------|------------|
| Exact search (A*) | Learned approximation |
| NP-hard | Polynomial time |
| Optimal cost | Approximate score |
| Expensive | Real-time streaming |

This trades optimality for speed.

---

## state-of-the-art Deep Learning Landscape (2025)

| Paradigm | Strength |
|----------|---------|
| Transformers (OREO) | Concept drift adaptation |
| LSTMs | Long-range temporal modeling |
| **GNNs** | **Preserve Petri net structure** |

**Unique advantage:**  
GNNs preserve the bipartite Petri net graph.  
Sequence models flatten structure.

---

## mechanism Graph Attention

For each node:

messages = W h_source
attention = softmax(a(messages))
aggregated = Σ attention_i × messages_i


This learns which places are **critical** for enabling transitions.

---

## temporal-context Prefix Encoding

- 3 recent activities  
- 6 activity types  
- One-hot → flattened → 18D vector

This is concatenated with graph embeddings before prediction.

Conceptually similar to:
- OREO sliding windows
- HMM observation histories

---

## strengths

- Structural inductive bias (Petri net topology)
- Hard semantic enforcement
- Step-by-step interpretability
- O(1) per event (fixed model)

---

## limitations

- No cascading simulation after violation
- Fixed penalty weights (0.1)
- No stochastic conformance modeling
- Only evaluated on toy dataset

---

## maturity

| Dimension | Status |
|-----------|--------|
| Algorithmic novelty | High |
| Empirical validation | Early |
| Production readiness | Prototype |

Requires:
- Concept drift handling
- Out-of-order events
- Object-centric processes

---

## thesis
This method positions **GNNs + Petri net semantics** as a new hybrid paradigm for online conformance checking:
> **Neural orientation + symbolic verification**  
> Fast, interpretable, structure-aware, but approximate.
