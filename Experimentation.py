import matplotlib.pyplot as plt
from networkx import betweenness_centrality
from pandas import np

import Graphs
from Active_Node_Chooser import Active_Node_Chooser, Greedy
from Moran_Process import numeric_fixation_probability, simulate, plot_fixation_iteration, get_all_graphs_of_size_n


def plot_degree(degree_list, numeric_data):
    fig, axs = plt.subplots()
    fig.suptitle('Degree Heuristic')

    axs.scatter(degree_list, numeric_data[1:])
    axs.axhline(y=round(numeric_data[0],5), color='r', linestyle='-', label='No Active Node Probability')
    axs.legend(loc=0, prop={'size': 6})
    axs.set_ylabel("Fixation Probability")
    axs.set_xlabel("Degree")

    plt.show()


def plot_temperature(temp_list, numeric_data):
    fig, axs = plt.subplots()
    fig.suptitle('Temperature Heuristic')

    axs.scatter(temp_list, numeric_data[1:])
    axs.axhline(y=round(numeric_data[0],5), color='r', linestyle='-', label='No Active Node Probability')
    axs.legend(loc=0, prop={'size': 6})
    axs.set_ylabel("Fixation Probability")
    axs.set_xlabel("Temperature")

    plt.show()


def plot_centrality(centrality_list, numeric_data):
    fig, axs = plt.subplots()
    fig.suptitle('Centrality Heuristic')

    axs.scatter(centrality_list, numeric_data[1:])
    axs.axhline(y=round(numeric_data[0],5), color='r', linestyle='-', label='No Active Node Probability')
    axs.legend(loc=0, prop={'size': 6})
    axs.set_ylabel("Fixation Probability")
    axs.set_xlabel("Centrality")

    plt.show()


def make_one_active(G):
    #Iterate all nodes and make them active one by one and see how this changes the fixation probability
    #Further plot this marginal increase as a function of heuristics and check for correlations
    numeric_data = []
    numeric_fixation_prob = numeric_fixation_probability(G, fitness)
    numeric_data.append(numeric_fixation_prob)
    for i in range(len(G.nodes())):
        G.nodes[i]['active'] = True

        numeric_fixation_prob = numeric_fixation_probability(G, fitness)
        numeric_data.append(numeric_fixation_prob)
        G.nodes[i]['active'] = False

    #Degree Heuristics
    degree_list = [v for k,v in G.degree()]
    plot_degree(degree_list,numeric_data)

    #Temperature Heuristic
    temp_list = np.zeros(len(G.nodes()))
    for node1, node2, data in G.edges(data=True):
        temp_list[node2] += list(data.values())[0]
    plot_temperature(temp_list, numeric_data)

    #Centrality Heuristic
    centrality_list = list(betweenness_centrality(G).values())
    plot_centrality(centrality_list,numeric_data)


def make_one_passive(G):
    #Iterate all nodes and make them all active.
    #Then iterate all nodes and one by one make a single one passive and see how this changes the fixation probability
    #Further plot this marginal decrease as a function of heuristics and check for correlations
    numeric_data = []
    numeric_fixation_prob = numeric_fixation_probability(G, fitness)
    numeric_data.append(numeric_fixation_prob)
    for i in range(len(G.nodes())):
        G.nodes[i]['active'] = True

    for i in range(len(G.nodes())):
        G.nodes[i]['active'] = False

        numeric_fixation_prob = numeric_fixation_probability(G, fitness)
        numeric_data.append(numeric_fixation_prob)
        G.nodes[i]['active'] = True

    #Degree Heuristics
    degree_list = [v for k,v in G.degree()]
    plot_degree(degree_list,numeric_data)

    #Temperature Heuristic
    temp_list = np.zeros(len(G.nodes()))
    for node1, node2, data in G.edges(data=True):
        temp_list[node2] += list(data.values())[0]
    plot_temperature(temp_list, numeric_data)

    #Centrality Heuristic
    centrality_list = list(betweenness_centrality(G).values())
    plot_centrality(centrality_list,numeric_data)


if __name__ == "__main__":
    fitness = 0.1
    multiplier = 1
    graph_size = 3
    eps = 0.0015

    #G = Graphs.create_complete_graph(graph_size)
    #G = Graphs.create_star_graph(graph_size)

    all_graphs_of_size_n = get_all_graphs_of_size_n("6c")
    G = all_graphs_of_size_n[35]

    Graphs.initialize_nodes_as_resident(G,multiplier)
    Graphs.draw_graph(G)

    make_one_active(G)
    make_one_passive(G)

