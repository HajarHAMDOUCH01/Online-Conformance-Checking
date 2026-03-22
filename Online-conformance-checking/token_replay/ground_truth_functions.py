import pm4py
from pm4py import *
from pm4py import convert_to_reachability_graph
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def choose_enabled_duplicate_transition(candidate_list):
    pass

def eventlog_petrinet_mapping(event_log, petri_net_model):
    """
    returns the transition of an activity 
    and
    handles if there is more than one transition for the same activity name
    """
    transitions = petri_net_model.transitions
    activities = event_log["concept:name"].unique()
    corresponding_transitions = []
    for activity_name in activities:
        for t in transitions:
            if activity_name == t.label:
                corresponding_transitions.append(t)
        # case when there is a single transition corresponding to the activity name
        if len(corresponding_transitions) == 1:
            return t
        else:
            if len(corresponding_transitions) == 0:
                # TO DO : 
                # this case doesn't occure because before this function 
                # another function for removing activities that don't correspond 
                # to any transition
                return None 
            # case where there is duplicate transitions
            else:
                # algorithm from paper : 
                # Conformance Checking of Processes Based on Monitoring Real Behavior
                pass
            
def transition_is_enabeled(marking, transition):
    """
    marking : is a set of cardinal number of 
    places each element of the set is an inetegr of 
    how many tokens are in that place
    transition : a pm4py petrinet tranistion obj
    """
    # an activity might be enabled if there is one or sequence of inivisible transitions
    # that make it enbaled before returning not enbaled

    # => TO DO : if using invisible transitions is causing higher future cost of future deviation
    # then better do alignement cost now for this activity if it is not enbaled
    pass

"""dataset generation from this petri net model :"""

"""
inputs needed : 
        event log
        trace
outputs :
        consumed tokens 
        produced tokens 
        missing tokens 
        remaining tokens
"""
def apply_replay(event_log_file_path):
    p = 1
