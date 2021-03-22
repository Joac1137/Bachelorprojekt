import matplotlib.pyplot as plt
from networkx import betweenness_centrality
from pandas import np
from Active_Node_Chooser import *
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


def make_one_passive(graph):
    G = graph.copy()
    #Iterate all nodes and make them all active.
    #Then iterate all nodes and one by one make a single one passive and see how this changes the fixation probability
    #Further plot this marginal decrease as a function of heuristics and check for correlations
    numeric_data = []

    for i in range(len(G.nodes())):
        G.nodes[i]['active'] = True

    numeric_fixation_prob = numeric_fixation_probability(G, fitness)
    numeric_data.append(numeric_fixation_prob)

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


def compare_active_node_strategies(G,fitness):
    nodes_list = list(range(len(G)))
    greedy_fixation_probabilities = []
    high_fixation_probabilities = []
    low_fixation_probabilities = []
    centrality_fixation_probabilities = []
    temperature_fixation_probabilities = []
    random_fixation_probabilities = []
    optimal_fixation_probabilities = []

    k_nodes = len(G)
    #The strategies

    print("Number of Active nodes to choose ", k_nodes)

    #Greedy
    greedy_chooser = Active_Node_Chooser(k_nodes,G,fitness,Greedy())
    greedy_nodes = greedy_chooser.choose_nodes()
    print("Greedy nodes to activate list", greedy_nodes)
    graph = G.copy()
    for j in greedy_nodes:
        graph.nodes[j]['active'] = True
        numeric_fixation_prob = numeric_fixation_probability(graph, fitness)
        greedy_fixation_probabilities.append(numeric_fixation_prob)

    #High Degree
    high_degree_chooser = Active_Node_Chooser(k_nodes,G,fitness,High_node_degree())
    high_degree_nodes = high_degree_chooser.choose_nodes()
    print("High Degree nodes to activate list", high_degree_nodes)
    graph = G.copy()
    for j in high_degree_nodes:
        graph.nodes[j]['active'] = True
        numeric_fixation_prob = numeric_fixation_probability(graph, fitness)
        high_fixation_probabilities.append(numeric_fixation_prob)

    #Low Degree
    low_degree_chooser = Active_Node_Chooser(k_nodes,G,fitness,Low_node_degree())
    low_degree_nodes = low_degree_chooser.choose_nodes()
    print("Low Degree nodes to activate list", low_degree_nodes)
    graph = G.copy()
    for j in low_degree_nodes:
        graph.nodes[j]['active'] = True
        numeric_fixation_prob = numeric_fixation_probability(graph, fitness)
        low_fixation_probabilities.append(numeric_fixation_prob)

    #Centrality
    centrality_chooser = Active_Node_Chooser(k_nodes,G,fitness,Centrality())
    centrality_nodes = centrality_chooser.choose_nodes()
    print("Centrality nodes to activate list", centrality_nodes)
    graph = G.copy()
    for j in centrality_nodes:
        graph.nodes[j]['active'] = True
        numeric_fixation_prob = numeric_fixation_probability(graph, fitness)
        centrality_fixation_probabilities.append(numeric_fixation_prob)

    #Temperature
    temperature_chooser = Active_Node_Chooser(k_nodes,G,fitness,Temperature())
    temperature_nodes = temperature_chooser.choose_nodes()
    print("Temperature nodes to activate list", temperature_nodes)
    graph = G.copy()
    for j in temperature_nodes:
        graph.nodes[j]['active'] = True
        numeric_fixation_prob = numeric_fixation_probability(graph, fitness)
        temperature_fixation_probabilities.append(numeric_fixation_prob)

    #Random
    random_chooser = Active_Node_Chooser(k_nodes,G,fitness,Random())
    random_nodes = random_chooser.choose_nodes()
    print("Random nodes to activate list", random_nodes)
    graph = G.copy()
    for j in random_nodes:
        graph.nodes[j]['active'] = True
        numeric_fixation_prob = numeric_fixation_probability(graph, fitness)
        random_fixation_probabilities.append(numeric_fixation_prob)

    #For the optimal we need to iterate the k nodes to make active as otherwise we only end at a good fixation probability
    #This enables us to also get a stepwise optimal solution
    for i in range(1, k_nodes + 1):
        #Optimal
        optimal_chooser = Active_Node_Chooser(i,G,fitness,Optimal())
        optimal_nodes = optimal_chooser.choose_nodes()
        print("Optimal nodes to activate list", optimal_nodes, "\n")
        graph = G.copy()
        for j in optimal_nodes:
            graph.nodes[j]['active'] = True
        numeric_fixation_prob = numeric_fixation_probability(graph, fitness)
        optimal_fixation_probabilities.append(numeric_fixation_prob)

    plt.plot(nodes_list,greedy_fixation_probabilities, label='Greedy')
    plt.plot(nodes_list,high_fixation_probabilities, label='High Degree')
    plt.plot(nodes_list,low_fixation_probabilities, label='Low Degree')
    plt.plot(nodes_list,centrality_fixation_probabilities, label='Centrality')
    plt.plot(nodes_list,temperature_fixation_probabilities, label='Temperature')
    plt.plot(nodes_list,random_fixation_probabilities, label='Random')
    plt.plot(nodes_list,optimal_fixation_probabilities, label='Optimal')

    plt.xlabel('Active Nodes')
    plt.ylabel('Fixation Probability')
    plt.legend()
    plt.show()

    pass


if __name__ == "__main__":
    fitness = 0.1
    multiplier = 1
    graph_size = 3
    eps = 0.0015

    #G = Graphs.create_complete_graph(graph_size)
    #G = Graphs.create_star_graph(graph_size)

    #6, 35
    all_graphs_of_size_n = get_all_graphs_of_size_n("8c")
    G = all_graphs_of_size_n[35]

    Graphs.initialize_nodes_as_resident(G,multiplier)
    Graphs.draw_graph(G)

    make_one_active(G)
    make_one_passive(G)
    compare_active_node_strategies(G,fitness)

