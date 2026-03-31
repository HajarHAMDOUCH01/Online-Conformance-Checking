import torch
def build_reachability_structures(reachability_graph):
    # All markings and transitions from pm4py reachability graph
    all_markings = list(reachability_graph.states)
    marking_to_idx = {m: i for i, m in enumerate(all_markings)}

    all_transitions = list({t.name for t in reachability_graph.transitions})
    all_transitions = sorted(all_transitions)
    t_name_to_idx = {name: i for i, name in enumerate(all_transitions)}

    num_m = len(all_markings)
    num_t = len(all_transitions)

    import math
    angle = math.pi / 2

    # Reachability tensor : shape (num_t, num_m, num_m)
    reachability_tensor = torch.zeros((num_t, num_m, num_m))

    # transition_to_enabled_markings
    transition_to_enabled_markings = {}

    for t in reachability_graph.transitions:
        src_idx = marking_to_idx[t.from_state]
        dst_idx = marking_to_idx[t.to_state]
        t_idx   = t_name_to_idx[t.name]

        reachability_tensor[t_idx, src_idx, dst_idx] = -angle
        reachability_tensor[t_idx, dst_idx, src_idx] =  angle

        if t.name not in transition_to_enabled_markings:
            transition_to_enabled_markings[t.name] = []
        transition_to_enabled_markings[t.name].append({
            'from':     t.from_state,
            'to':       t.to_state,
            'from_idx': src_idx,
            'to_idx':   dst_idx
        })

    return (all_markings, marking_to_idx,
            all_transitions, t_name_to_idx,
            num_m, num_t,
            reachability_tensor,
            transition_to_enabled_markings)