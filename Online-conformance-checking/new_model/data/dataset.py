import pm4py
from pm4py import *
from pm4py import convert_to_reachability_graph
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import pickle

ROOT_DIR = 'C:\\Users\\LENONVO\\OneDrive\\Desktop\\sujet-CRAN\\datasets'

def import_petrinet_model(pnml_file_path, return_reachability_graph=False):
    petrinet, mi, mf = pm4py.read_pnml(pnml_file_path)
    reachbility_graph = pm4py.convert_to_reachability_graph(petrinet, mi, mf)
    if return_reachability_graph:
        return petrinet, mi, mf, reachbility_graph
    return petrinet, mi, mf

def get_reachability_graph(petrinet, mi, mf):
    reachability_graph = pm4py.convert_to_reachability_graph(petrinet, mi, mf)
    print("\nlen(reachability_graph.states)\n", len(reachability_graph.states))
    return reachability_graph

def import_event_log(event_log_file_path):
    event_log = pm4py.read_xes(event_log_file_path)
    return event_log

def vizualise_petrinet(petrinet: PetriNet):
    G = nx.DiGraph()
    places = {p.name: p for p in petrinet.places}
    transitions = {t.name: t for t in petrinet.transitions}

    for p in petrinet.places:
        G.add_node(p.name, kind="place")
    for t in petrinet.transitions:
        G.add_node(t.name, kind="transition")
    for arc in petrinet.arcs:
        G.add_edge(arc.source.name, arc.target.name)
    
    pos = nx.spring_layout(G, seed=42, k=2)
    place_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "place"]
    transition_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "transition"]
    print(place_nodes)

    plt.figure(figsize=(16,10))
    nx.draw_networkx_nodes(G, pos, nodelist=place_nodes, node_shape="o", node_color="lightblue", node_size=800)
    nx.draw_networkx_nodes(G, pos, nodelist=transition_nodes, node_shape="s",
                           node_color="lightgreen", node_size=600)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=7)
    p1 = mpatches.Patch(color="lightblue", label="Place")
    p2 = mpatches.Patch(color="lightgreen", label="Transition")
    plt.legend(handles=[p1, p2])
    plt.title("Petri Net")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_reachability_graph(reachability_graph):
    G = nx.DiGraph()

    for state in reachability_graph.states:
        G.add_node(state.name)

    for transition in reachability_graph.transitions:
        G.add_edge(transition.from_state.name, transition.to_state.name, label=transition.name)

    pos = nx.spring_layout(G, seed=42, k=3)

    plt.figure(figsize=(24, 16))
    nx.draw_networkx_nodes(G, pos, node_color="orange", node_size=200)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=8, width=0.5, alpha=0.5)

    plt.title(f"Reachability Graph ({len(reachability_graph.states)} states)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

petrinet, mi, mf = import_petrinet_model(f"{ROOT_DIR}\\spesis\\spesis_reference_model.pnml")
vizualise_petrinet(petrinet)
reachability_graph = get_reachability_graph(petrinet, mi, mf)
visualize_reachability_graph(reachability_graph)
# get_reachability_graph(petrinet, mi, mf)
# 1050 cases , 15214 events 
# sur 18 months 
# 16 unique activities
base_event_log = import_event_log(f"{ROOT_DIR}\\spesis\\spesis_full_log.xes")
train_event_log = import_event_log(f"{ROOT_DIR}\\spesis\\spesis_training_log.xes")

# this is pefix aligenemnt experiment paper results
# without loops , with choice and parallelism
# with noises : add event , swap tasks , remove task
# noise is from 0% to 30%
# 8 window sizes : -1 , 1 , 2 , 3 , 4 , 5 , 10 , 20
# upper bound : True / False
# 19M lines (events)
# synthetic_results = pd.read_csv(
#     f"{ROOT_DIR}\\synthetic\\log.csv",
#     sep="\t",
#     skiprows=2,  # sauter les 2 lignes de commentaires (#)
#     header=None,
#     names=[
#         "Population", "Iteration", "Log_size", "Noise_type", "Noise_pct",
#         "Avg_queue_size", "Cost", "Cost_delta", "Enqueued_nodes",
#         "Traversed_arcs", "Prefix_length", "Search_time",
#         "Visited_nodes", "Upper_bound", "Window_size"
#     ]
# )

#this is results of the orinal reference model and the base log

# sepsis_results = {}
# for w in [1, 2, 3, 4, 5, 'infinite']:
#     df = pd.read_csv(
#         f"{ROOT_DIR}\\spesis\\window_{w}.csv",
#         sep=";",
#         skiprows=1,  # sauter la ligne header du fichier
#         names=["prefix_length", "cost", "cost_delta", "cost_delta_full",
#                "enqueued_nodes", "visited_nodes", "traversed_edges",
#                "avg_queue_size", "search_time", "window"]
#     )
#     # convertir en numérique
#     for col in ["prefix_length", "cost", "cost_delta", "cost_delta_full",
#                 "enqueued_nodes", "visited_nodes", "traversed_edges",
#                 "avg_queue_size", "search_time"]:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     sepsis_results[w] = df
#     # print(f"\n=== window_{w} === Shape: {df.shape}")
#     # print(df.head(3))


# # unique trace variants in the full_log.xes
# trace_variants_df = (
#     base_event_log
#     .sort_values(["case:concept:name", "time:timestamp"])
#     .groupby("case:concept:name")["concept:name"]
#     .apply(tuple)
#     .reset_index()
#     .rename(columns={"concept:name":"trace"})
# )
# unique_traces = trace_variants_df.drop_duplicates(subset="trace").reset_index(drop=True)
# print("unique traces : ", len(unique_traces))
# print(unique_traces.head(5))
# # with open(f"{ROOT_DIR}\\spesis\\unique_traces.pkl", "wb") as f:
# #     pickle.dump(unique_traces, f)
# # unique_traces.to_csv(f"{ROOT_DIR}\\spesis\\unique_traces.csv", index=False)

# # creation of the stream of unique events :
# unique_events = (
#     base_event_log[["case:concept:name", "concept:name", "time:timestamp"]]
#     .sort_values(["case:concept:name", "time:timestamp"])
#     .reset_index(drop=True)
# )
# # Garder seulement les events des traces uniques
# unique_case_ids = unique_traces["case:concept:name"].tolist()
# unique_events = unique_events[unique_events["case:concept:name"].isin(unique_case_ids)].reset_index(drop=True)

# print("Nombre d'événements uniques:", len(unique_events))  # 13775

# # Associer avec window_infinite comme ground truth
# df_inf = sepsis_results['infinite'].copy()
# df_inf = df_inf.reset_index(drop=True)

# ground_truth = pd.concat([unique_events, df_inf], axis=1)
# print(ground_truth.head(10))

# ground_truth.to_csv(f"{ROOT_DIR}\\spesis\\ground_truth.csv", index=False)
"""
Dans ground_truth.csv
Chaque ligne contient :

case:concept:name → case ID
concept:name → activité
time:timestamp → timestamp
prefix_length, cost, cost_delta, cost_delta_full → ground truth prefix-alignments
enqueued_nodes, visited_nodes, etc. → métriques de performance
"""

# print("synthetic_results.attrs : ", synthetic_results.attrs)
# print("synthetic_results.values : ", synthetic_results.values)
# print("\nsynthetic_results.columns.tolist() :\n", synthetic_results.columns.tolist())
# print("\nsynthetic_results.head(10) :\n", synthetic_results.head(10))
# print("Populations uniques:\n", synthetic_results["Population"].unique())
# print("\nNoise types:\n", synthetic_results["Noise_type"].unique())
# print("\nNoise pct uniques:\n", sorted(synthetic_results["Noise_pct"].unique()))
# print("\nWindow sizes:\n", sorted(synthetic_results["Window_size"].unique()))
# print("\nUpper bound:\n", synthetic_results["Upper_bound"].unique())
# print("\nTotal lignes:", len(synthetic_results))

# base event log data 
# print("\nBASE EVENT LOG DATA : \n")
# print("Type:", type(base_event_log))
# print("Shape:", base_event_log.shape)
# print("Columns:", base_event_log.columns.tolist())
# print("\nHead:\n", base_event_log.head())
# print("\nActivités uniques:\n", sorted(base_event_log["concept:name"].unique().tolist()))
# print("\nNombre de cas:", base_event_log["case:concept:name"].nunique())
# print("\nNombre d'événements:", len(base_event_log))
# print("\nPériode:", base_event_log["time:timestamp"].min(), "→", base_event_log["time:timestamp"].max())


# print("\nTRAINING EVENT LOG DATA : \n")
# print("Type:", type(train_event_log))
# print("Shape:", train_event_log.shape)
# print("Columns:", train_event_log.columns.tolist())
# print("\nHead:\n", train_event_log.head())
# print("\nActivités uniques:\n", sorted(train_event_log["concept:name"].unique().tolist()))
# print("\nNombre de cas:", train_event_log["case:concept:name"].nunique())
# print("\nNombre d'événements:", len(train_event_log))
# print("\nPériode:", train_event_log["time:timestamp"].min(), "→", train_event_log["time:timestamp"].max())