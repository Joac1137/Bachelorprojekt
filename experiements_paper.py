import math
import os
import random
import Graphs
import networkx as nx
from Active_Node_Chooser import High_node_degree,Low_node_degree,Centrality,Temperature,Random,Lazy_Greedy,Vertex_Cover
from Moran_Process import simulate
from Active_Node_Chooser import Active_Node_Chooser

iterations = 10000

def activate_and_simulate(G,r,nodes_to_activate):
    graph = G.copy()
    for j in nodes_to_activate:
        graph.nodes[j]['active'] = True
    fixation_list, simulated_fixation_prob = simulate(iterations,graph,r)
    return simulated_fixation_prob


def compare_heuristics(G_new,r,j):
    num_active_nodes = math.floor(len(G_new.nodes)*j)

    high_fixation_probabilities = []
    low_fixation_probabilities = []
    centrality_fixation_probabilities = []
    temperature_fixation_probabilities = []
    random_fixation_probabilities = []
    lazy_greedy_fixation_probabilities = []
    vertex_cover_probabilities = []
    weak_selection_fixation_probabilities = []

    high_degree_chooser = Active_Node_Chooser(num_active_nodes,G_new,r,High_node_degree())
    print("choosing high degree")
    high_degree_nodes = high_degree_chooser.choose_nodes()
    print("High Degree nodes to activate list", high_degree_nodes)
    fix_prop = activate_and_simulate(G_new,r,high_degree_nodes)
    high_fixation_probabilities.append(fix_prop)

    low_degree_chooser = Active_Node_Chooser(num_active_nodes,G_new,r,Low_node_degree())
    print("choosing low degree")
    low_degree_nodes = low_degree_chooser.choose_nodes()
    print("Low Degree nodes to activate list", low_degree_nodes)
    fix_prop = activate_and_simulate(G_new,r,low_degree_nodes)
    low_fixation_probabilities.append(fix_prop)

    centrality_chooser = Active_Node_Chooser(num_active_nodes,G_new,r,Centrality())
    centrality_nodes = centrality_chooser.choose_nodes()
    fix_prop = activate_and_simulate(G_new,r,centrality_nodes)
    centrality_fixation_probabilities.append(fix_prop)

    temperature_chooser = Active_Node_Chooser(num_active_nodes,G_new,r,Temperature())
    temperature_nodes = temperature_chooser.choose_nodes()
    fix_prop = activate_and_simulate(G_new,r,temperature_nodes)
    temperature_fixation_probabilities.append(fix_prop)


    random_chooser = Active_Node_Chooser(num_active_nodes,G_new,r,Random())
    random_nodes = random_chooser.choose_nodes()
    fix_prop = activate_and_simulate(G_new,r,random_nodes)
    random_fixation_probabilities.append(fix_prop)

    lazy_chooser = Active_Node_Chooser(num_active_nodes,G_new,r,Lazy_Greedy())
    lazy_nodes = lazy_chooser.choose_nodes()
    fix_prop = activate_and_simulate(G_new,r,lazy_nodes)
    lazy_greedy_fixation_probabilities.append(fix_prop)

    vertex_chooser = Active_Node_Chooser(num_active_nodes,G_new,r,Vertex_Cover())
    vertex_nodes = vertex_chooser.choose_nodes()
    fix_prop = activate_and_simulate(G_new,r,vertex_nodes)
    vertex_cover_probabilities.append(fix_prop)


    #missing weak selections strategy here

if __name__=="__main__":
    files = os.listdir("Graphs_for_paper_experiments\\prepared_graphs\\")
    # for i in files[]:
    i=files[0]
    G = nx.read_edgelist("Graphs_for_paper_experiments\\prepared_graphs\\"+i)
    G = Graphs.initialize_nodes_as_resident(G)
    for r in [0.05, 0.1, 0.2, 0.5, 1, 5, 10, "inf"]:
        G_new = G.copy()
        for j in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,]: #percentage of active node
            if not r == "inf":
                compare_heuristics(G_new,r,j)

