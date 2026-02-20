"""
Input v_source : Un marquage quelconque du graphe de joignabilité.
Input v_target : Un autre marquage atteignable depuis v_source.
Label (Cible) : Le vecteur alpha idéal qui représente le chemin le plus court.
"""

import heapq
import queue
def get_shortest_path_alphas(graph, start_node, end_node, num_transitions, t_name_to_idx):
    queue = [(0, start_node, [0] * num_transitions)]
    visited = {start_node: 0}

    while queue:
        (dist, current, counts) = heapq.heappop(queue)

        if current == end_node:
            return counts
        for t_name, mode, next_marking in graph.get(current, []):
            new_dist = dist + 1
            if next_marking not in visited or new_dist < visited[next_marking]:
                visited[next_marking] = new_dist
                new_counts = list(counts)
                new_counts[t_name_to_idx[t_name]] += 1
                heapq.heappush(queue, (new_dist, next_marking, new_counts))
    return None

dataset = []

