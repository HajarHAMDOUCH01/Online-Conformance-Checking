import pm4py
from pm4py import *
from pm4py import convert_to_reachability_graph
import pandas as pd
from pm4py.algo.discovery.declare import algorithm as declare_miner
from pm4py.objects.log.obj import Trace, Event, EventLog
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.conformance.tokenreplay.variants.token_replay import Parameters
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from Online_conformance_checking.new_model.fitness_model import LSTMFitnessModel

##############################################
import torch
from torch.utils.data import Dataset

class PrefixDataset(Dataset):
    def __init__(self, dataset, act_to_idx, max_len=185):
        self.max_len=max_len
        self.dataset = dataset
        self.act_to_idx=act_to_idx
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]

        activities = sample['prefix']
        fitness = sample['fitness']
        
        length = len(activities)  # real length before padding
        indices = [self.act_to_idx.get(act, 0) for act in activities]
        padded = indices + [0] * (self.max_len - len(indices))
        
        x = torch.tensor(padded, dtype=torch.long)
        y = torch.tensor(fitness, dtype=torch.float32)
        length = torch.tensor(length, dtype=torch.long)
        
        return x, length, y
##############################################

def activity_to_idx(activities_list):
    act_to_idx = {act:i+1 for i,act in enumerate(activities_list)}
    return act_to_idx

def import_petrinet_model(pnml_file_path, return_reachability_graph=False):
    petrinet, mi, mf = pm4py.read_pnml(pnml_file_path)
    reachbility_graph = pm4py.convert_to_reachability_graph(petrinet, mi, mf)
    if return_reachability_graph:
        return petrinet, mi, mf, reachbility_graph
    return petrinet, mi, mf

def import_event_log(event_log_file_path):
    event_log = pm4py.read_xes(event_log_file_path)
    return event_log

def build_prefix_trace(trace, k):
    prefix = trace[0:k]
    prefix = Trace(prefix)
    return prefix

def build_activity_index(filtered_rules):
    index = {}
    for template, constraints in filtered_rules.items():
        for activity_key, stats in constraints.items():
            if isinstance(activity_key, str):
                # single activity — use it directly
                if activity_key not in index:
                    index[activity_key] = []
                index[activity_key].append((template, activity_key, stats))
            else:
                # tuple of two activities
                for activity in activity_key:
                    if activity not in index:
                        index[activity] = []
                    index[activity].append((template, activity_key, stats))
    
    print("Activities in index:", len(index))
    for act, rules in sorted(index.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {act}: {len(rules)} rules")
    return index

import networkx as nx

def build_graph(rg):
    G = nx.DiGraph()

    for t in rg.transitions:
        src = str(t.from_state)
        tgt = str(t.to_state)

        label = t.name[1]
        weight = 0 if label is None else 1

        G.add_edge(src, tgt, weight=weight, label=label)

    return G

def find_target_markings(rg, activity):
    targets = set()

    for t in rg.transitions:
        if t.name[1] == activity:
            targets.add(str(t.from_state))

    return targets


def find_nearest_target(G, current_states, targets):
    best_target = None
    best_cost = float("inf")
    best_path = None

    for s in current_states:
        s = str(s)
        lengths, paths = nx.single_source_dijkstra(G, s)

        for t in targets:
            if t in lengths and lengths[t] < best_cost:
                best_cost = lengths[t]
                best_target = t
                best_path = paths[t]

    return best_target, best_path, best_cost


def align_prefix(prefix, rg, mi):
    valid_prefix = prefix[:-1]
    deviant = prefix[-1]

    current_states = {mi}
    for act in valid_prefix:
        current_states = get_tau_closure(current_states, rg)

        next_states = set()
        for s in current_states:
            for t in rg.transitions:
                if t.from_state == s and t.name[1] == act:
                    next_states.add(t.to_state)

        if not next_states:
            return None  

        current_states = next_states

    current_states = get_tau_closure(current_states, rg)

    G = build_graph(rg)

    targets = find_target_markings(rg, deviant)

    target, path, cost = find_nearest_target(G, current_states, targets)

    return path

def get_tau_closure(states, rg):
    state_names = {s.name for s in states}
    closure = set(states)
    stack = list(states)

    while stack:
        s = stack.pop()
        for t in rg.transitions:
            if t.from_state.name == s.name and get_transition_label(t) is None:
                if t.to_state.name not in state_names:
                    state_names.add(t.to_state.name)
                    closure.add(t.to_state)
                    stack.append(t.to_state)
    return closure

def get_transition_label(t):
    name = t.name  
    parts = name.split(', ', 1)  
    label_part = parts[1].rstrip(')')  
    if label_part == 'None':
        return None
    return label_part.strip("'")

def compute_fitness(prefix_activities, rg, initial_state):
    import ast
    sync_moves = 0
    current_states = {initial_state}

    for activity in prefix_activities:
        current_states = get_tau_closure(current_states, rg)
        current_names = {s.name for s in current_states}

        next_states = set()
        for t in rg.transitions:
            if t.from_state.name in current_names and get_transition_label(t) == activity:
                next_states.add(t.to_state)

        if next_states:
            current_states = next_states
            sync_moves += 1

    return sync_moves / len(prefix_activities) if prefix_activities else 1.0


def build_ground_truth(log_obj, rg, initial_state):
    dataset = []  # list of dicts
    for i, trace in enumerate(log_obj):
        trace_i = log_obj[i]
        case_id = trace_i.attributes['concept:name']
        for k in range(1, len(trace) + 1):
            activities = [e['concept:name'] for e in trace[:k]]
            # fitness = compute_fitness(activities, rg, initial_state)
            dataset.append({
                'case_id' : case_id,
                'prefix': activities,
                # 'fitness': fitnes

            })
    return dataset

import json

def save_dataset(dataset, path):
    with open(path, 'w') as f:
        json.dump(dataset, f)

def load_dataset(path):
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    log_file = r"C:\Users\LENONVO\OneDrive\Desktop\graphs\sujet-CRAN\datasets\spesis\spesis_full_log.xes"
    model = r"C:\Users\LENONVO\OneDrive\Desktop\graphs\sujet-CRAN\datasets\spesis\spesis_reference_model.pnml"
    dataset_path = r"C:\Users\LENONVO\OneDrive\Desktop\graphs\sujet-CRAN\datasets\spesis\ground_truth_dataset.json"
    ground_truth = r"C:\Users\LENONVO\OneDrive\Desktop\graphs\sujet-CRAN\datasets\spesis\ground_truth.csv"
    ground_truth_pd = pd.read_csv(ground_truth)
    
    from pm4py.objects.petri_net.importer import importer as pnml_importer
    from pm4py.visualization.petri_net import visualizer as pn_visualizer

    # net, initial_marking, final_marking = pnml_importer.apply(model)
    # gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    # pn_visualizer.view(gviz)

    petrinet, mi, mf = import_petrinet_model(model, return_reachability_graph=False)
    # print(rg.transitions)
    from pm4py.visualization.transition_system import visualizer as ts_visualizer
    from pm4py.objects.petri_net.utils import reachability_graph


    rg = reachability_graph.construct_reachability_graph(petrinet, mi)

    initial_state = None
    for state in rg.states:
        if state.name == 'n11':
            initial_state = state
            break

    print("Initial state found:", initial_state)

    closure = get_tau_closure({initial_state}, rg)
    print("Tau closure of initial marking:", closure)
    print("Types in closure:", [type(s) for s in closure])

    
    
    log = pm4py.read_xes(log_file)
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'],utc=True)
    
    unique_activities = log['concept:name'].unique().tolist()

    log_obj = pm4py.convert_to_event_log(log)

    print()
    print(ground_truth_pd.head(30))
    print()
    # import os
    # if os.path.exists(dataset_path):
    #     print("Loading existing dataset...")
    #     dataset = load_dataset(dataset_path)
    # else:
    #     print("Computing dataset...")
    #     dataset = build_ground_truth(log_obj, rg, initial_state)
    #     save_dataset(dataset, dataset_path)
    #     print(f"Dataset saved: {len(dataset)} samples")

    # # dataset = build_ground_truth(log_obj, rg, initial_state)
    # act_to_idx = activity_to_idx(unique_activities)
    # vocab_size = len(act_to_idx) + 1
    # model = LSTMFitnessModel(vocab_size=vocab_size)
    # ds = PrefixDataset(dataset, act_to_idx)
    # x, length, y = ds[0]
    # print(x)
    # x = x.unsqueeze(0)       # add batch dimension
    # length = length.unsqueeze(0)
    # pred, (h, c) = model(x, length)
    # print(pred.shape, pred)
    # print(h.shape, c.shape)

    # fitenss_prediction_model = LSTMFitnessModel()
    # y_model = fitenss_prediction_model(x)
    # print(y_model)

    # gviz = ts_visualizer.apply(rg)
    # ts_visualizer.view(gviz)

    # print(log['time:timestamp'].dtype)
    # declare_model = declare_miner.apply(log)
    # print(declare_model['init'].keys())

    # print(type(declare_model))
    # print(declare_model.keys() if hasattr(declare_model, 'keys') else dir(declare_model))
    
    # min_support = 840 
    # min_cof_ratio = 0.9
    # filtered_rules = {}
    # for template, constraints in declare_model.items():
    #     for activities, stats in constraints.items():
    #         sup = stats['support']
    #         conf = stats['confidence']
    #         if sup >= min_support and conf / sup >=  min_cof_ratio:
    #             if template not in filtered_rules:
    #                 filtered_rules[template] = {}
    #             filtered_rules[template][activities] = stats

    # total = sum(len(v) for v in filtered_rules.values())
    # print(f"Constraints surviving filter: {total}")
    # for template, constraints in filtered_rules.items():
    #     print(f"  {template}: {len(constraints)}")
    
    # index = build_activity_index(filtered_rules)

