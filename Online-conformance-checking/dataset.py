"""
Input v_source : Un marquage quelconque du graphe de joignabilité.
Input v_target : Un autre marquage atteignable depuis v_source.
Label (Cible) : Le vecteur alpha idéal qui représente le chemin le plus court.
"""
from reachability_graph_construction import *
import heapq

def get_shortest_path_alphas(graph, start_node, end_node, num_transitions, t_name_to_idx):
    # Dijkstra pour trouver le chemin le plus court des transitions : to do : pense à l'ordre !!
    # queue: (distance, current_marking, transitions_counts)
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

# Génération du Dataset
dataset = []

print("=== Generating Dataset ===")
for m_src in all_markings:
    for m_tgt in all_markings:
        if m_src == m_tgt:
            continue
            
        alphas = get_shortest_path_alphas(reachability_graph, m_src, m_tgt, num_t, t_name_to_idx)
        # print(alphas)
        if alphas:
            dataset.append({
                'v_src_idx': marking_to_idx[m_src],
                'v_tgt_idx': marking_to_idx[m_tgt],
                'alphas': alphas
            })

print(f"Dataset generated: {len(dataset)} examples found.")

example = dataset[4] 
print(f"\nExemple de trajet:")
print(f"Source: {all_markings[example['v_src_idx']]}")
print(f"Target: {all_markings[example['v_tgt_idx']]}")
print(f"Alphas cibles (transitions à tirer): {example['alphas']}")

