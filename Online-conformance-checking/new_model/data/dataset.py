import pm4py
from pm4py import *
from pm4py import convert_to_reachability_graph
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

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

petrinet, mi, mf = import_petrinet_model(f"{ROOT_DIR}\\spesis\\spesis_reference_model.pnml")
get_reachability_graph(petrinet, mi, mf)
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

# train_stream = train_event_log[["case:concept:name", "concept:name", "time:timestamp"]].copy()
# train_stream = train_stream.sort_values("time:timestamp").reset_index(drop=True)
# print(train_stream)