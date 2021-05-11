import matplotlib.pyplot as plt

from networkx import betweenness_centrality
from pandas import np
from Active_Node_Chooser import *
from itertools import combinations
import Graphs
import collections
from Active_Node_Chooser import Active_Node_Chooser, Greedy
from Moran_Process import numeric_fixation_probability, simulate, plot_fixation_iteration, get_all_graphs_of_size_n
import pandas as pd


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


def compare_active_node_strategies_simulation(G, fitness,name):
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
    # lazy_greedy_fixation_probabilities = []
    #
    min_iterations=1000
    min_iterations=30000
    iteration_list = range(min_iterations)
    k_nodes = len(G)
    #The strategies


    #Try to run it without the greedy strategy
    #Greedy
    # simulated_fixation_prob = 0
    # lazy_greedy_chooser = Active_Node_Chooser(k_nodes,G,fitness,Lazy_Greedy())
    # lazy_greedy_nodes = lazy_greedy_chooser.choose_nodes()
    # print("Lazy greedy nodes to activate list", lazy_greedy_nodes)
    # graph = G.copy()
    # for j in lazy_greedy_nodes:
    #     graph.nodes[j]['active'] = True
    #
    #     fixation_list, simulated_fixation_prob = simulate(min_iterations,graph,fitness,lowest_acceptable_fitness=simulated_fixation_prob)
    #     lazy_greedy_fixation_probabilities.append(simulated_fixation_prob)
    #     print("Simulated fixation probability for Lazy greedy = ", simulated_fixation_prob)
    #     plot_fixation_iteration(iteration_list, fixation_list, 0)


    simulated_fixation_prob = 0
    #High Degree
    high_degree_chooser = Active_Node_Chooser(k_nodes,G,fitness,High_node_degree())
    high_degree_nodes = high_degree_chooser.choose_nodes()
    print("High Degree nodes to activate list", high_degree_nodes)
    graph = G.copy()
    for j in high_degree_nodes:
        print("Iteration ",j, " of ", high_degree_nodes)
        graph.nodes[j]['active'] = True
        #print("The graph ", graph.nodes(data=True))
        #numeric_fixation_prob = numeric_fixation_probability(graph, fitness)
        #print("The numeric solution is the following ", numeric_fixation_prob)
        fixation_list, simulated_fixation_prob = simulate(min_iterations,graph,fitness,lowest_acceptable_fitness=simulated_fixation_prob)

        high_fixation_probabilities.append(simulated_fixation_prob)
        print("Simulated fixation probability for High Degree = ", simulated_fixation_prob)
        plot_fixation_iteration(iteration_list, fixation_list, 0)

    simulated_fixation_prob = 0

    #Try to run it without the Low Degree strategy
    #Low Degree
    low_degree_chooser = Active_Node_Chooser(k_nodes,G,fitness,Low_node_degree())
    low_degree_nodes = low_degree_chooser.choose_nodes()
    print("Low Degree nodes to activate list", low_degree_nodes)
    graph = G.copy()
    for j in low_degree_nodes:
        print("Iteration ",j, " of ", low_degree_nodes)
        graph.nodes[j]['active'] = True
        #print("The graph ", graph.nodes(data=True))
        #numeric_fixation_prob = numeric_fixation_probability(graph, fitness)
        #print("The numeric solution is the following ", numeric_fixation_prob)

        fixation_list, simulated_fixation_prob = simulate(min_iterations,graph,fitness,lowest_acceptable_fitness=simulated_fixation_prob)
        low_fixation_probabilities.append(simulated_fixation_prob)
        print("Simulated fixation probability for Low Degree = ", simulated_fixation_prob)
        plot_fixation_iteration(iteration_list, fixation_list, 0)

    simulated_fixation_prob = 0
    #Centrality
    centrality_chooser = Active_Node_Chooser(k_nodes,G,fitness,Centrality())
    centrality_nodes = centrality_chooser.choose_nodes()
    print("Centrality nodes to activate list", centrality_nodes)
    graph = G.copy()
    for j in centrality_nodes:
        print("Iteration ",j, " of ", centrality_nodes)
        graph.nodes[j]['active'] = True

        fixation_list, simulated_fixation_prob = simulate(min_iterations,graph,fitness,lowest_acceptable_fitness=simulated_fixation_prob)
        centrality_fixation_probabilities.append(simulated_fixation_prob)
        print("Simulated fixation probability for Centrality = ", simulated_fixation_prob)
        plot_fixation_iteration(iteration_list, fixation_list, 0)

    simulated_fixation_prob = 0
    #Temperature
    temperature_chooser = Active_Node_Chooser(k_nodes,G,fitness,Temperature())
    temperature_nodes = temperature_chooser.choose_nodes()
    print("Temperature nodes to activate list", temperature_nodes)
    graph = G.copy()
    for j in temperature_nodes:
        print("Iteration ",j, " of ", temperature_nodes)
        graph.nodes[j]['active'] = True

        fixation_list, simulated_fixation_prob = simulate(min_iterations,graph,fitness,lowest_acceptable_fitness=simulated_fixation_prob)
        temperature_fixation_probabilities.append(simulated_fixation_prob)
        print("Simulated fixation probability for Temperature = ", simulated_fixation_prob)
        plot_fixation_iteration(iteration_list, fixation_list, 0)

    simulated_fixation_prob = 0
    #Random
    random_chooser = Active_Node_Chooser(k_nodes,G,fitness,Random())
    random_nodes = random_chooser.choose_nodes()
    print("Random nodes to activate list", random_nodes)
    graph = G.copy()
    for j in random_nodes:
        print("Iteration ",j, " of ", random_nodes)
        graph.nodes[j]['active'] = True

        fixation_list, simulated_fixation_prob = simulate(min_iterations,graph,fitness,lowest_acceptable_fitness=simulated_fixation_prob)
        random_fixation_probabilities.append(simulated_fixation_prob)
        print("Simulated fixation probability for Random = ", simulated_fixation_prob)
        plot_fixation_iteration(iteration_list, fixation_list, 0)

    simulated_fixation_prob = 0

    plt.plot(nodes_list,high_fixation_probabilities, label='High Degree')
    plt.plot(nodes_list,low_fixation_probabilities, label='Low degree')
    plt.plot(nodes_list,centrality_fixation_probabilities, label='Centrality')
    plt.plot(nodes_list,temperature_fixation_probabilities, label='Temperature')
    plt.plot(nodes_list,random_fixation_probabilities, label='Random')

    plt.xlabel('Active Nodes')
    plt.ylabel('Fixation Probability')
    plt.title(name)
    plt.legend()
    plt.show()

    fixation_list_dict = {'High Degree': high_fixation_probabilities,'High Degree nodes': high_degree_nodes, 'Centrality':centrality_fixation_probabilities, 'Centrality nodes': centrality_nodes, 'Temparature':temperature_fixation_probabilities, 'Temperature nodes':temperature_nodes,'Random':random_fixation_probabilities, 'Random nodes': random_nodes,'Low Degree':low_fixation_probabilities, 'Low degree nodes': low_degree_nodes}
    # fixation_list_dict = {'High Degree': high_fixation_probabilities, 'Greedy':greedy_fixation_probabilities, 'Centrality':centrality_fixation_probabilities, 'Temparature':temperature_fixation_probabilities, 'Random':random_fixation_probabilities}

    df = pd.DataFrame(fixation_list_dict)
    path = "C:\\Users\\AsgerUllerstedRasmus\\Desktop\\bachelor\\" + name + "_" + str(k_nodes) + "_f_" + str(fitness) + ".csv"
    df.to_csv(path,index_col = False)



def make_one_passive_simulation(graph):
    G = graph.copy()
    #Iterate all nodes and make them all active.
    #Then iterate all nodes and one by one make a single one passive and see how this changes the fixation probability
    #Further plot this marginal decrease as a function of heuristics and check for correlations

    min_iterations=1000
    max_iterations=1000
    eps = 0.0015

    simulation_data = []
    for i in range(len(G.nodes())):
        G.nodes[i]['active'] = True

    iteration_list, fixation_list, simulated_fixation_prob = simulate(min_iterations,G,fitness,0,eps,max_iterations)
    simulation_data.append(simulated_fixation_prob)
    print("Simulated fixation probability for all active = ", simulated_fixation_prob)
    plot_fixation_iteration(iteration_list, fixation_list, 0)

    for i in range(len(G.nodes())):
        G.nodes[i]['active'] = False

        iteration_list, fixation_list, simulated_fixation_prob = simulate(min_iterations,G,fitness,0,eps,max_iterations)
        simulation_data.append(simulated_fixation_prob)
        print("Simulated fixation probability for one passive = ", simulated_fixation_prob)
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

    min_iterations=1000
    max_iterations=1000
    eps = 0.0015

    simulation_data = []
    iteration_list, fixation_list, simulated_fixation_prob = simulate(min_iterations,graph,fitness,0,eps,max_iterations)
    simulation_data.append(simulated_fixation_prob)
    print("Simulated fixation probability for none active = ", simulated_fixation_prob)
    plot_fixation_iteration(iteration_list, fixation_list, 0)

    for i in range(len(G.nodes())):
        G.nodes[i]['active'] = True

        iteration_list, fixation_list, simulated_fixation_prob = simulate(min_iterations,graph,fitness,0,eps,max_iterations)
        simulation_data.append(simulated_fixation_prob)
        print("Simulated fixation probability for one active = ", simulated_fixation_prob)
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


def submodularity(G,fitness):
    """
    Power set code from: https://www.geeksforgeeks.org/power-set/

    :param G: The graph
    :return:
    """
    #Python program to find powerset
    number_of_nodes = len(G.nodes)
    node_set = set(range(0, number_of_nodes))
    all_pairs = []
    for i in range(0, number_of_nodes + 1):
        for element in itertools.combinations(node_set,i):
            int_of_node = [int(s) for s in element if not isinstance(s,str)]
            sets = set(int_of_node)
            all_pairs.append(list(sets))

    #Iterate the powerset in order to get all combinations
    #i is the smaller set
    for i in all_pairs:
        for j in all_pairs:
            set_i = set(i)
            set_j = set(j)

            if set_i.issubset(set_j):
                #print("i ", set_i)
                #print("j ", set_j, "\n")

                #Iterate all nodes not in j
                #Calculate the fixation probability for the the graph activating the nodes in the sets
                #Check that Submodular formula holds
                for x in node_set.difference(j):
                    graph_i = G.copy()
                    graph_j = G.copy()

                    for i_element in set_i:
                        graph_i.nodes[i_element]['active'] = True

                    set_i_numeric_fixation_prob = numeric_fixation_probability(graph_i, fitness)
                    graph_i.nodes[x]['active'] = True
                    large_i_numeric_fixation_prob = numeric_fixation_probability(graph_i, fitness)


                    for j_element in set_j:
                        graph_j.nodes[j_element]['active'] = True

                    set_j_numeric_fixation_prob = numeric_fixation_probability(graph_j, fitness)
                    graph_j.nodes[x]['active'] = True
                    large_j_numeric_fixation_prob = numeric_fixation_probability(graph_j, fitness)


                    i_diff = large_i_numeric_fixation_prob - set_i_numeric_fixation_prob
                    j_diff = large_i_numeric_fixation_prob - set_i_numeric_fixation_prob

                    #print("i diff ", i_diff)
                    #print("j diff ", j_diff)

                    if i_diff < j_diff:
                        #print("Large i ", large_i_numeric_fixation_prob)
                        #print("i ", set_i_numeric_fixation_prob)
                        #print("Large j ", large_j_numeric_fixation_prob)
                        #print("j ", set_j_numeric_fixation_prob)

                        return False
    return True


def greedy_optimal_choices(size):
    #Number 72 for 6 nodes might not have same Optimal as Greedy
    #Number 125 for 7 nodes might not have same Optimal as Greedy
    counter = 0
    all_graphs_of_size_n = get_all_graphs_of_size_n(str(size)+"c")
    for graph in all_graphs_of_size_n:
        #graph = all_graphs_of_size_n[72]
        #graph = all_graphs_of_size_n[125]
        Graphs.initialize_nodes_as_resident(graph)

        G = graph

        #Greedy
        greedy_fixation_probabilities = []
        greedy_chooser = Active_Node_Chooser(2,G,fitness,Greedy())
        greedy_nodes = greedy_chooser.choose_nodes()

        graph = G.copy()
        for j in greedy_nodes:
            graph.nodes[j]['active'] = True
            numeric_fixation_prob = numeric_fixation_probability(graph, fitness)
            greedy_fixation_probabilities.append(numeric_fixation_prob)



        #Optimal
        optimal_fixation_probabilities = []
        optimal_chooser = Active_Node_Chooser(2,G,fitness,Optimal())
        optimal_nodes = optimal_chooser.choose_nodes()

        graph = G.copy()
        for j in optimal_nodes:
            graph.nodes[j]['active'] = True
            numeric_fixation_prob = numeric_fixation_probability(graph, fitness)
            optimal_fixation_probabilities.append(numeric_fixation_prob)



        greedy_fixation_probabilities = [round(x,5) for x in greedy_fixation_probabilities]
        optimal_fixation_probabilities = [round(x,5) for x in optimal_fixation_probabilities]

        same_nodes = collections.Counter(greedy_nodes) != collections.Counter(optimal_nodes)
        same_probs = greedy_fixation_probabilities[len(greedy_fixation_probabilities)-1] != optimal_fixation_probabilities[len(optimal_fixation_probabilities)-1]
        print("Round ", counter, " of ",len(all_graphs_of_size_n))
        if same_nodes and same_probs:

            print("We found an error \n")
            print("Greedy nodes to activate list", greedy_nodes)
            print("The fixation pobabilities for Greedy ", greedy_fixation_probabilities)

            print("Optimal nodes to activate list", optimal_nodes)
            print("The fixation pobabilities for Optimal ", optimal_fixation_probabilities, "\n")
            Graphs.draw_graph(graph)
        counter +=1

def calculate_submodularity(size, fitness):
    """
    Compute submodularity for all graph of a given size
    :param size:
    :return:
    """
    all_graphs_of_size_n = get_all_graphs_of_size_n(str(size)+"c")
    counter = 1
    for graph in all_graphs_of_size_n:
        print("Graph ", counter, " of ", len(all_graphs_of_size_n))
        counter += 1

        Graphs.initialize_nodes_as_resident(graph)
        Graphs.draw_graph(graph)
        sub = submodularity(graph,fitness)
        if not sub:
            print("The graph did NOT have the submodularity property")
            break
    print("Submodularity does hold for all graphs of size ", size, "\n")

def compare_greedy_lazygreedy(G, fitness):
    nodes_list = list(range(len(G)))
    lazy_fixation_probabilities = []
    greedy_fixation_probabilities = []

    # min_iterations=1000
    min_iterations=10000
    iteration_list = range(min_iterations)
    k_nodes = len(G)


    #Try to run it without the greedy strategy
    Greedy
    simulated_fixation_prob = 0
    greedy_chooser = Active_Node_Chooser(k_nodes,G,fitness,Greedy())
    greedy_nodes = greedy_chooser.choose_nodes()
    print("Greedy nodes to activate list", greedy_nodes)
    graph = G.copy()
    for j in greedy_nodes:
        graph.nodes[j]['active'] = True

        fixation_list, simulated_fixation_prob = simulate(min_iterations,graph,fitness,lowest_acceptable_fitness=simulated_fixation_prob)
        greedy_fixation_probabilities.append(simulated_fixation_prob)
        print("Simulated fixation probability for Greedy = ", simulated_fixation_prob)
        plot_fixation_iteration(iteration_list, fixation_list, 0)
    #

    simulated_fixation_prob = 0
    #Lazy
    lazy_chooser = Active_Node_Chooser(k_nodes,G,fitness,Lazy_Greedy())
    lazy_nodes = lazy_chooser.choose_nodes()
    print("Lazy nodes to activate list", lazy_nodes)
    graph = G.copy()
    for j in lazy_nodes:
        print("Iteration ",j, " of ", lazy_nodes)
        graph.nodes[j]['active'] = True
        #print("The graph ", graph.nodes(data=True))
        #numeric_fixation_prob = numeric_fixation_probability(graph, fitness)
        #print("The numeric solution is the following ", numeric_fixation_prob)
        fixation_list, simulated_fixation_prob = simulate(min_iterations,graph,fitness,lowest_acceptable_fitness=simulated_fixation_prob)

        lazy_fixation_probabilities.append(simulated_fixation_prob)
        print("Simulated fixation probability for lazy = ", simulated_fixation_prob)
        plot_fixation_iteration(iteration_list, fixation_list, 0)

    plt.plot(nodes_list,lazy_fixation_probabilities, label='Lazy Greedy')
    plt.plot(nodes_list,greedy_fixation_probabilities, label='Greedy')

    plt.xlabel('Active Nodes')
    plt.ylabel('Fixation Probability')
    plt.legend()
    plt.show()

    fixation_list_dict = {'Lazy Greedy': lazy_fixation_probabilities,'Lazy nodes': lazy_nodes, 'Greedy':greedy_fixation_probabilities, 'Greedy nodes': greedy_nodes}
    # fixation_list_dict = {'High Degree': high_fixation_probabilities, 'Greedy':greedy_fixation_probabilities, 'Centrality':centrality_fixation_probabilities, 'Temparature':temperature_fixation_probabilities, 'Random':random_fixation_probabilities}

    df = pd.DataFrame(fixation_list_dict)
    path = "C:\\Users\\AsgerUllerstedRasmus\\Desktop\\bachelor\\karate_club_data_greedy_vs_lazy.csv"
    df.to_csv(path)


def heuristic_comparison_caveman(fitneses):
    for fitness in fitneses:
        name = "connected_caveman_f_" + str(fitness)
        graph = nx.connected_caveman_graph(2, 4)
        Graphs.initialize_nodes_as_resident(graph, multiplier)
        Graphs.draw_graph(graph)
        compare_active_node_strategies_simulation(graph, fitness, name)


def heuristic_comparison_davis_southern_women(fitneses):
    graph = nx.davis_southern_women_graph()
    Graphs.initialize_nodes_as_resident(graph, multiplier)
    Graphs.draw_graph(graph)

def heuristic_comparison_florentine_families(fitneses):
    graph = nx.florentine_families_graph()
    Graphs.initialize_nodes_as_resident(graph, multiplier)
    Graphs.draw_graph(graph)

def heuristic_comparison_random_internet(fitneses):
    graph = nx.random_internet_as_graph(50)
    Graphs.initialize_nodes_as_resident(graph, multiplier)
    Graphs.draw_graph(graph)


def heuristic_comparison_erdos_renyi(fitneses):
    p = 0.1
    graph = nx.erdos_renyi_graph(50, p, directed=True)
    # G = G.to_directed()
    while not nx.is_strongly_connected(graph):
        p = p+0.03
        graph = nx.erdos_renyi_graph(50, p, directed=True)
    Graphs.initialize_nodes_as_resident(graph, multiplier)
    Graphs.draw_graph(graph)


if __name__ == "__main__":
    fitness = 5
    multiplier = 1
    # graph_size = 4
    # eps = 0.0015

    # G = Graphs.create_complete_graph(graph_size)
    #G = Graphs.create_star_graph(graph_size)
    #Graphs.initialize_nodes_as_resident(G,multiplier)
    #Graphs.draw_graph(G)
    #compare_active_node_strategies_simulation(G,fitness)

    # star1 = Graphs.create_star_graph(3)
    # star2 = Graphs.create_star_graph(5)
    # mega_star = nx.union(star1,star2,rename=('a','b'))
    # mega_star = nx.convert_node_labels_to_integers(mega_star,first_label=0)
    # mega_star.add_edge(1,6)
    # Graphs.initialize_nodes_as_resident(mega_star,multiplier)
    # Graphs.draw_graph(mega_star)
    # compare_active_node_strategies_simulation(mega_star,fitness)

    #6, 35, 29
    #all_graphs_of_size_n = get_all_graphs_of_size_n("6c")
    #G = all_graphs_of_size_n[35]
    #G = mega_star

    #Graphs.initialize_nodes_as_resident(G,multiplier)
    #Graphs.draw_graph(G)

    #make_one_active_numeric(G)
    #make_one_passive_numeric(G)
    #compare_active_node_strategies_numeric(G,fitness)



    #Big graph for simulation based comparison between heuristics
    # star1 = Graphs.create_star_graph(3)
    # star2 = Graphs.create_star_graph(4)
    # mega_star = nx.union(star1,star2,rename=('a','b'))
    # mega_star = nx.convert_node_labels_to_integers(mega_star,first_label=0)
    # mega_star.add_edge(1,5)
    #
    # Graphs.initialize_nodes_as_resident(mega_star,multiplier)
    #Graphs.draw_graph(mega_star)

    # compare_active_node_strategies_simulation(mega_star,fitness,eps)
    # make_one_active_simulation(mega_star)
    # make_one_passive_simulation(mega_star)
    #compare_active_node_strategies_simulation(mega_star,fitness)
    #make_one_active_simulation(mega_star)
    #make_one_passive_simulation(mega_star)


    #Calculate submodularity for all graphs for parameter specified size
    # calculate_submodularity(6,fitness)

    #Calculate Greedy and optimal choice for active nodes for all graph of given size
    #greedy_optimal_choices(7)
    # graph = Graphs.create_karate_club_graph()
    # graph = nx.barabasi_albert_graph(10, 3)

    # Graphs.initialize_nodes_as_resident(graph,multiplier)
    # Graphs.draw_graph(graph)
    # compare_active_node_strategies_simulation(graph,fitness)
    # compare_greedy_lazygreedy(graph,fitness)

    # Experiments
    fitneses = [0.1, 0.2, 0.5, 1, 1.5]
    heuristic_comparison_caveman(fitneses)
    # heuristic_comparison_davis_southern_women(fitneses)
    # heuristic_comparison_florentine_families(fitneses)
    # heuristic_comparison_random_internet(fitneses)
    # heuristic_comparison_erdos_renyi(fitneses)