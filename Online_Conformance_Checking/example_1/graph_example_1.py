"""this example is N1 Petri Net from paper : relating event streams to process
models using prefix-alignments"""

from snakes.nets import *

n = PetriNet("N1_example_1")

# Places
n.add_place(Place("p0", [0]))
n.add_place(Place("p1", []))
n.add_place(Place("p2", []))
n.add_place(Place("p3", []))
n.add_place(Place("p4", []))
n.add_place(Place("p5", []))
n.add_place(Place("p6", []))

# Transitions 
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