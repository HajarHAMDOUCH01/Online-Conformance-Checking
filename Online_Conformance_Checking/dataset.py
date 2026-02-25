"""
dataset.py — builds train/test splits from the reachability graph.

Strategy:
  - Direct paths (length 1): always in train — these are the atomic building blocks
  - Complex paths (length > 1): 60% train / 40% test
  - Test set contains only complex paths never seen during training

Labels are shortest paths (BFS), represented as sequences of transition indices.
"""

import random
import heapq
import torch

from reachability_graph import (
    reachability_graph,
    all_markings,
    marking_to_idx,
    t_name_to_idx,
    num_m,
    num_t,
)


# ── Shortest path via Dijkstra/BFS ────────────────────────────────────────────

def get_shortest_path_sequence(graph, start_node, end_node, t_name_to_idx):
    """
    Returns the shortest sequence of transition indices from start_node to end_node,
    or None if unreachable.
    """
    queue   = [(0, start_node, [])]
    visited = {start_node: 0}

    while queue:
        dist, current, seq = heapq.heappop(queue)

        if current == end_node:
            return seq

        if current in graph:
            for t_name, mode, next_marking in graph[current]:
                new_dist = dist + 1
                if next_marking not in visited or new_dist < visited[next_marking]:
                    visited[next_marking] = new_dist
                    heapq.heappush(queue, (new_dist, next_marking,
                                           seq + [t_name_to_idx[t_name]]))
    return None


# ── Dataset construction ──────────────────────────────────────────────────────

def create_split_dataset(graph, all_markings, t_name_to_idx, train_ratio=0.6,
                          seed=42):
    """
    Returns (train_data, test_data) where each element is a dict:
        v_src_idx   : int
        v_tgt_idx   : int
        alphas_seq  : list of transition indices (shortest path)
        length      : int
    """
    random.seed(seed)

    direct_paths  = []   # length == 1
    complex_paths = []   # length  > 1

    for m_src in all_markings:
        for m_tgt in all_markings:
            if m_src == m_tgt:
                continue

            seq = get_shortest_path_sequence(graph, m_src, m_tgt, t_name_to_idx)
            if seq is None:
                continue

            entry = {
                'v_src_idx': marking_to_idx[m_src],
                'v_tgt_idx': marking_to_idx[m_tgt],
                'alphas_seq': seq,
                'length':     len(seq),
            }

            if len(seq) == 1:
                direct_paths.append(entry)
            else:
                complex_paths.append(entry)

    random.shuffle(complex_paths)
    split_idx = int(len(complex_paths) * train_ratio)

    train_data = direct_paths + complex_paths[:split_idx]
    test_data  = complex_paths[split_idx:]

    return train_data, test_data


# ── Tensor conversion ─────────────────────────────────────────────────────────

def prepare_tensors(data, num_m, num_t):
    """
    Returns:
        X_src        : [N, num_m]  one-hot source markings
        X_tgt        : [N, num_m]  one-hot target markings
        seq_alphas   : list of N tensors, each [L_i, num_t] one-hot transition seqs
    """
    v_src_list, v_tgt_list, seq_alphas_list = [], [], []

    for item in data:
        src = torch.zeros(num_m); src[item['v_src_idx']] = 1.0
        tgt = torch.zeros(num_m); tgt[item['v_tgt_idx']] = 1.0

        steps = []
        for t_idx in item['alphas_seq']:
            step = torch.zeros(num_t)
            step[t_idx] = 1.0
            steps.append(step)

        v_src_list.append(src)
        v_tgt_list.append(tgt)
        seq_alphas_list.append(torch.stack(steps))   # [L, num_t]

    return torch.stack(v_src_list), torch.stack(v_tgt_list), seq_alphas_list


# ── Build and export ──────────────────────────────────────────────────────────

train_data, test_data = create_split_dataset(
    reachability_graph, all_markings, t_name_to_idx
)
print(f"Train: {len(train_data)} | Test: {len(test_data)}")

X_src_train, X_tgt_train, y_alphas_train = prepare_tensors(train_data, num_m, num_t)
X_src_test,  X_tgt_test,  y_alphas_test  = prepare_tensors(test_data,  num_m, num_t)