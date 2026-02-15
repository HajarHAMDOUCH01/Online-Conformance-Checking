"""
A Function that can create the reachability graph of a Petri Net using snakes library
"""
"""
A Function that can create the reachability graph of a Petri Net using snakes library
"""
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

# Test de tir
modes = n.transition("t1").modes()
print(f"modes of t1: {modes}")
if modes:
    n.transition("t1").fire(modes[0])

print(f"marking after firing t1: {n.get_marking()}")
# output : 
# modes of t1: [Substitution(x=0)]
# marking after firing t1: {p1={0}, p2={0}}
