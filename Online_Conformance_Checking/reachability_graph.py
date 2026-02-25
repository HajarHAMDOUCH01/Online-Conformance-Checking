"""
reachability_graph.py — builds the reachability graph and derived structures
from the Petri net definition in example_2.

Exports:
    reachability_graph              : dict {marking: [(t_name, mode, next_marking)]}
    all_markings                    : list of markings (stable ordering)
    marking_to_idx                  : dict {marking: int}
    all_transition_names            : sorted list of transition name strings
    t_name_to_idx                   : dict {t_name: int}
    num_m                           : int  number of reachable markings
    num_t                           : int  number of transitions
    reachability_tensor             : torch.Tensor [num_t, num_m, num_m]
    transition_to_enabled_markings  : dict {t_name: [{'from', 'to', 'from_idx', 'to_idx'}]}
"""

import sys
import math
from collections import deque

import torch

sys.path.append("Online-conformance-checking")
from example_2.graph_example_2 import n   # the PetriNet object


# ── Build reachability graph ──────────────────────────────────────────────────

def build_reachability_graph(net):
    graph   = {}
    initial = net.get_marking()
    queue   = deque([initial])
    visited = {initial}

    while queue:
        marking = queue.popleft()
        graph[marking] = []
        net.set_marking(marking)

        for t_name in [t.name for t in net.transition()]:
            t = net.transition(t_name)
            net.set_marking(marking)
            for mode in t.modes():
                net.set_marking(marking)
                t.fire(mode)
                next_marking = net.get_marking()

                graph[marking].append((t_name, mode, next_marking))

                if next_marking not in visited:
                    visited.add(next_marking)
                    queue.append(next_marking)

    return graph


reachability_graph = build_reachability_graph(n)

print("\n=== Reachability Graph ===")
for marking, transitions in reachability_graph.items():
    print(f"\nFrom: {marking}")
    for t_name, mode, next_marking in transitions:
        print(f"  --[{t_name}]--> {next_marking}")
print(f"\nTotal states: {len(reachability_graph)}")


# ── Index structures ──────────────────────────────────────────────────────────

all_markings        = list(reachability_graph.keys())
marking_to_idx      = {m: i for i, m in enumerate(all_markings)}

all_transition_names = sorted([t.name for t in n.transition()])
t_name_to_idx        = {name: i for i, name in enumerate(all_transition_names)}

num_m = len(all_markings)
num_t = len(all_transition_names)


# ── Reachability tensor [num_t, num_m, num_m] ─────────────────────────────────

def build_reachability_tensor(num_t, num_m, reachability_graph):
    tensor = torch.zeros(num_t, num_m, num_m)
    angle  = math.pi / 2

    for marking, transitions in reachability_graph.items():
        src_idx = marking_to_idx[marking]
        for t_name, mode, next_marking in transitions:
            dst_idx = marking_to_idx[next_marking]
            t_idx   = t_name_to_idx[t_name]
            tensor[t_idx, src_idx, dst_idx] = -angle
            tensor[t_idx, dst_idx, src_idx] =  angle

    return tensor


reachability_tensor = build_reachability_tensor(num_t, num_m, reachability_graph)


# ── Enabled markings per transition ──────────────────────────────────────────

transition_to_enabled_markings = {}

for marking, transitions in reachability_graph.items():
    for t_name, mode, next_marking in transitions:
        if t_name not in transition_to_enabled_markings:
            transition_to_enabled_markings[t_name] = []
        transition_to_enabled_markings[t_name].append({
            'from':      marking,
            'to':        next_marking,
            'from_idx':  marking_to_idx[marking],
            'to_idx':    marking_to_idx[next_marking],
        })