import matplotlib.pyplot as plt
import nx as nx
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


def make_one_active_numeric(G):
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
        temp_list[node1] += list(data.values())[0]
        temp_list[node2] += list(data.values())[0]
    plot_temperature(temp_list, numeric_data)

    #Centrality Heuristic
    centrality_list = list(betweenness_centrality(G).values())
    plot_centrality(centrality_list,numeric_data)


def make_one_passive_numeric(graph):
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
        temp_list[node1] += list(data.values())[0]
        temp_list[node2] += list(data.values())[0]
    plot_temperature(temp_list, numeric_data)

    #Centrality Heuristic
    centrality_list = list(betweenness_centrality(G).values())
    plot_centrality(centrality_list,numeric_data)


def compare_active_node_strategies_numeric(G,fitness):
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


def compare_active_node_strategies_simulation(G, fitness, eps):
    """
    We don't know the numeric fixation prob for large graphs so we just set it to zero and force the simulation to run 20000 iterations
    We hope to see a converge anyway. We are just explicit in knowing the value it converges towards
    :param G: The graph
    :param fitness: Fitness
    :param eps: Epilon
    :return:
    """
    nodes_list = list(range(len(G)))
    high_fixation_probabilities = []
    low_fixation_probabilities = []
    centrality_fixation_probabilities = []
    temperature_fixation_probabilities = []
    random_fixation_probabilities = []

    min_iterations=2000
    max_iterations=2000

    k_nodes = len(G)
    #The strategies

    """
    Try to run it without the greedy strategy
    #Greedy
    greedy_chooser = Active_Node_Chooser(k_nodes,G,fitness,Greedy())
    greedy_nodes = greedy_chooser.choose_nodes()
    print("Greedy nodes to activate list", greedy_nodes)
    graph = G.copy()
    for j in greedy_nodes:
        graph.nodes[j]['active'] = True

        iteration_list, fixation_list, simulated_fixation_prob = simulate(3000,graph,fitness,0,eps,max_iterations)
        greedy_fixation_probabilities.append(simulated_fixation_prob)
        print("Simulated fixation probability for Greedy = ", simulated_fixation_prob)
        plot_fixation_iteration(iteration_list, fixation_list, 0)"""

    #High Degree
    high_degree_chooser = Active_Node_Chooser(k_nodes,G,fitness,High_node_degree())
    high_degree_nodes = high_degree_chooser.choose_nodes()
    print("High Degree nodes to activate list", high_degree_nodes)
    graph = G.copy()
    for j in high_degree_nodes:
        print("Iteration ",j, " of ", high_degree_nodes)
        graph.nodes[j]['active'] = True

        iteration_list, fixation_list, simulated_fixation_prob = simulate(min_iterations,graph,fitness,0,eps,max_iterations)
        high_fixation_probabilities.append(simulated_fixation_prob)
        print("Simulated fixation probability for High Degree = ", simulated_fixation_prob)
        plot_fixation_iteration(iteration_list, fixation_list, 0)

    #Low Degree
    low_degree_chooser = Active_Node_Chooser(k_nodes,G,fitness,Low_node_degree())
    low_degree_nodes = low_degree_chooser.choose_nodes()
    print("Low Degree nodes to activate list", low_degree_nodes)
    graph = G.copy()
    for j in low_degree_nodes:
        print("Iteration ",j, " of ", low_degree_nodes)
        graph.nodes[j]['active'] = True

        iteration_list, fixation_list, simulated_fixation_prob = simulate(min_iterations,graph,fitness,0,eps,max_iterations)
        low_fixation_probabilities.append(simulated_fixation_prob)
        print("Simulated fixation probability for Low Degree = ", simulated_fixation_prob)
        plot_fixation_iteration(iteration_list, fixation_list, 0)

    #Centrality
    centrality_chooser = Active_Node_Chooser(k_nodes,G,fitness,Centrality())
    centrality_nodes = centrality_chooser.choose_nodes()
    print("Centrality nodes to activate list", centrality_nodes)
    graph = G.copy()
    for j in centrality_nodes:
        print("Iteration ",j, " of ", centrality_nodes)
        graph.nodes[j]['active'] = True

        iteration_list, fixation_list, simulated_fixation_prob = simulate(min_iterations,graph,fitness,0,eps,max_iterations)
        centrality_fixation_probabilities.append(simulated_fixation_prob)
        print("Simulated fixation probability for Centrality = ", simulated_fixation_prob)
        plot_fixation_iteration(iteration_list, fixation_list, 0)

    #Temperature
    temperature_chooser = Active_Node_Chooser(k_nodes,G,fitness,Temperature())
    temperature_nodes = temperature_chooser.choose_nodes()
    print("Temperature nodes to activate list", temperature_nodes)
    graph = G.copy()
    for j in temperature_nodes:
        print("Iteration ",j, " of ", temperature_nodes)
        graph.nodes[j]['active'] = True

        iteration_list, fixation_list, simulated_fixation_prob = simulate(min_iterations,graph,fitness,0,eps,max_iterations)
        temperature_fixation_probabilities.append(simulated_fixation_prob)
        print("Simulated fixation probability for Temperature = ", simulated_fixation_prob)
        plot_fixation_iteration(iteration_list, fixation_list, 0)

    #Random
    random_chooser = Active_Node_Chooser(k_nodes,G,fitness,Random())
    random_nodes = random_chooser.choose_nodes()
    print("Random nodes to activate list", random_nodes)
    graph = G.copy()
    for j in random_nodes:
        print("Iteration ",j, " of ", random_nodes)
        graph.nodes[j]['active'] = True

        iteration_list, fixation_list, simulated_fixation_prob = simulate(min_iterations,graph,fitness,0,eps,max_iterations)
        random_fixation_probabilities.append(simulated_fixation_prob)
        print("Simulated fixation probability for Random = ", simulated_fixation_prob)
        plot_fixation_iteration(iteration_list, fixation_list, 0)

    plt.plot(nodes_list,high_fixation_probabilities, label='High Degree')
    plt.plot(nodes_list,low_fixation_probabilities, label='Low Degree')
    plt.plot(nodes_list,centrality_fixation_probabilities, label='Centrality')
    plt.plot(nodes_list,temperature_fixation_probabilities, label='Temperature')
    plt.plot(nodes_list,random_fixation_probabilities, label='Random')

    plt.xlabel('Active Nodes')
    plt.ylabel('Fixation Probability')
    plt.legend()
    plt.show()

    pass



def make_one_passive_simulation(G):
    graph = G.copy()
    #Iterate all nodes and make them all active.
    #Then iterate all nodes and one by one make a single one passive and see how this changes the fixation probability
    #Further plot this marginal decrease as a function of heuristics and check for correlations

    min_iterations=2000
    max_iterations=2000
    eps = 0.0015

    simulation_data = []
    for i in range(len(G.nodes())):
        G.nodes[i]['active'] = True

    iteration_list, fixation_list, simulated_fixation_prob = simulate(min_iterations,graph,fitness,0,eps,max_iterations)
    simulation_data.append(simulated_fixation_prob)
    print("Simulated fixation probability for Random = ", simulated_fixation_prob)
    plot_fixation_iteration(iteration_list, fixation_list, 0)

    for i in range(len(G.nodes())):
        G.nodes[i]['active'] = False

        iteration_list, fixation_list, simulated_fixation_prob = simulate(min_iterations,graph,fitness,0,eps,max_iterations)
        simulation_data.append(simulated_fixation_prob)
        print("Simulated fixation probability for Random = ", simulated_fixation_prob)
        plot_fixation_iteration(iteration_list, fixation_list, 0)

        G.nodes[i]['active'] = True

    #Degree Heuristics
    degree_list = [v for k,v in G.degree()]
    plot_degree(degree_list,simulation_data)

    #Temperature Heuristic
    temp_list = np.zeros(len(G.nodes()))
    for node1, node2, data in G.edges(data=True):
        temp_list[node1] += list(data.values())[0]
        temp_list[node2] += list(data.values())[0]
    plot_temperature(temp_list, simulation_data)

    #Centrality Heuristic
    centrality_list = list(betweenness_centrality(G).values())
    plot_centrality(centrality_list,simulation_data)

def make_one_active_simulation(graph):
    #Iterate all nodes and make them active one by one and see how this changes the fixation probability
    #Further plot this marginal increase as a function of heuristics and check for correlations

    min_iterations=2000
    max_iterations=2000
    eps = 0.0015

    simulation_data = []
    iteration_list, fixation_list, simulated_fixation_prob = simulate(min_iterations,graph,fitness,0,eps,max_iterations)
    simulation_data.append(simulated_fixation_prob)
    print("Simulated fixation probability for Random = ", simulated_fixation_prob)
    plot_fixation_iteration(iteration_list, fixation_list, 0)

    for i in range(len(G.nodes())):
        G.nodes[i]['active'] = True

        iteration_list, fixation_list, simulated_fixation_prob = simulate(min_iterations,graph,fitness,0,eps,max_iterations)
        simulation_data.append(simulated_fixation_prob)
        print("Simulated fixation probability for Random = ", simulated_fixation_prob)
        plot_fixation_iteration(iteration_list, fixation_list, 0)

        G.nodes[i]['active'] = False

    #Degree Heuristics
    degree_list = [v for k,v in G.degree()]
    plot_degree(degree_list,simulation_data)

    #Temperature Heuristic
    temp_list = np.zeros(len(G.nodes()))
    for node1, node2, data in G.edges(data=True):
        temp_list[node1] += list(data.values())[0]
        temp_list[node2] += list(data.values())[0]
    plot_temperature(temp_list, simulation_data)

    #Centrality Heuristic
    centrality_list = list(betweenness_centrality(G).values())
    plot_centrality(centrality_list,simulation_data)


if __name__ == "__main__":
    fitness = 0.1
    multiplier = 1
    graph_size = 3
    eps = 0.0015

    #G = Graphs.create_complete_graph(graph_size)
    #G = Graphs.create_star_graph(graph_size)
    star1 = Graphs.create_star_graph(3)
    star2 = Graphs.create_star_graph(5)
    mega_star = nx.union(star1,star2,rename=('a','b'))
    mega_star = nx.convert_node_labels_to_integers(mega_star,first_label=0)
    mega_star.add_edge(1,6)
    Graphs.initialize_nodes_as_resident(mega_star,multiplier)
    Graphs.draw_graph(mega_star)

    #6, 35, 29
    #all_graphs_of_size_n = get_all_graphs_of_size_n("8c")
    #G = all_graphs_of_size_n[35]
    G = mega_star

    Graphs.initialize_nodes_as_resident(G,multiplier)
    Graphs.draw_graph(G)

    #make_one_active_numeric(G)
    #make_one_passive_numeric(G)
    #compare_active_node_strategies_numeric(G,fitness)



    #Big graph for simulation based comparison between heuristics
    star1 = Graphs.create_star_graph(5)
    star2 = Graphs.create_star_graph(8)
    mega_star = nx.union(star1,star2,rename=('a','b'))
    mega_star = nx.convert_node_labels_to_integers(mega_star,first_label=0)
    mega_star.add_edge(1,7)
    Graphs.initialize_nodes_as_resident(mega_star,multiplier)

    Graphs.initialize_nodes_as_resident(mega_star,multiplier)
    Graphs.draw_graph(mega_star)

    #compare_active_node_strategies_simulation(mega_star,fitness,eps)
    make_one_active_simulation(mega_star)
    make_one_passive_simulation(mega_star)