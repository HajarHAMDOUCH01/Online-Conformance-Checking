from snakes.nets import *

n = PetriNet("experimentation_example")

# Places
n.add_place(Place("p0", [0]))
n.add_place(Place("p1", []))
n.add_place(Place("p2", []))
n.add_place(Place("p3", []))
n.add_place(Place("p4", []))
n.add_place(Place("p5", []))
n.add_place(Place("p6", []))

# Transitions (sans Expression pour les noms simples)
n.add_transition(Transition("t1"))
n.add_transition(Transition("t2"))
n.add_transition(Transition("t3"))
n.add_transition(Transition("t4"))
n.add_transition(Transition("t5"))
n.add_transition(Transition("t6"))
n.add_transition(Transition("t7"))
n.add_transition(Transition("t8"))

# Inputs
n.add_input("p0", "t1", Variable("x"))
n.add_input("p1", "t2", Variable("x"))
n.add_input("p2", "t3", Variable("x"))
n.add_input("p4", "t4", Variable("x"))
n.add_input("p1", "t4", Variable("x"))
n.add_input("p3", "t5", Variable("x"))
n.add_input("p4", "t5", Variable("x"))
n.add_input("p5", "t6", Variable("x"))
n.add_input("p5", "t7", Variable("x"))
n.add_input("p5", "t8", Variable("x"))

# Outputs
n.add_output("p1", "t1", Variable("x"))
n.add_output("p2", "t1", Variable("x"))
n.add_output("p3", "t2", Variable("x"))
n.add_output("p4", "t3", Variable("x"))
n.add_output("p5", "t4", Variable("x"))
n.add_output("p5", "t5", Variable("x"))
n.add_output("p1", "t6", Variable("x"))
n.add_output("p2", "t6", Variable("x"))
n.add_output("p6", "t7", Variable("x"))
n.add_output("p6", "t8", Variable("x"))

from collections import deque

def build_reachability_graph(net):
    graph = {}  # {marking: [(transition, mode, next_marking)]}
    
    initial = net.get_marking()
    queue = deque([initial])
    visited = {initial}
    
    while queue:
        marking = queue.popleft()
        graph[marking] = []
        
        net.set_marking(marking)  # restaure le marquage courant
        
        for t_name in [t.name for t in net.transition()]:
            t = net.transition(t_name)
            net.set_marking(marking)
            for mode in t.modes():
                print("Firing transition:", t_name, "with mode:", mode)
                # tire la transition
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


"""
=== Reachability Graph ===

From: {p0={0}}
  --[t1]--> {p1={0}, p2={0}}

From: {p1={0}, p2={0}}
  --[t2]--> {p2={0}, p3={0}}
  --[t3]--> {p1={0}, p4={0}}

From: {p2={0}, p3={0}}
  --[t3]--> {p3={0}, p4={0}}

From: {p1={0}, p4={0}}
  --[t2]--> {p3={0}, p4={0}}
  --[t4]--> {p5={0}}

From: {p3={0}, p4={0}}
  --[t5]--> {p5={0}}

From: {p5={0}}
  --[t6]--> {p1={0}, p2={0}}
  --[t7]--> {p6={0}}
  --[t8]--> {p6={0}}

From: {p6={0}}

Total states: 7
"""

all_markings = list(reachability_graph.keys())
marking_to_idx = {m: i for i, m in enumerate(all_markings)}

all_transition_names = sorted([t.name for t in n.transition()])
t_name_to_idx = {name: i for i, name in enumerate(all_transition_names)}

num_m = len(all_markings)
num_t = len(all_transition_names)

# reachability graph as a tensor of shape (Transitions, from_marking, to_marking)

# to doo : 
# to verify if training : 
# or between transitions from marking to the next making
# choosing a transtion each time and how 

import torch
import math

reachability_tensor = torch.zeros((num_t, num_m, num_m))
angle = math.pi / 2

for marking, transitions in reachability_graph.items():
    src_marking_idx = marking_to_idx[marking]

    for t_name, mode, next_marking in transitions:
        dst_marking_idx = marking_to_idx[next_marking]
        t_idx = t_name_to_idx[t_name]
        
        reachability_tensor[t_idx, src_marking_idx, dst_marking_idx] = -angle 
        reachability_tensor[t_idx, dst_marking_idx, src_marking_idx] = angle # to doo : verify that this never happens 

# print("\nReachability Tensor :", reachability_tensor)
