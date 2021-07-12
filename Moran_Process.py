import matplotlib as matplotlib
from random import uniform, random, sample, randint, choices
import itertools
from os import environ
import draw_nx_beautiful
import networkx as nx
import time
import Graphs
import matplotlib.pyplot as plt
import random
import numpy as np
from networkx import betweenness_centrality
import os

def plot_degree(degree_list, numeric_data,legend):
    fig, axs = plt.subplots()
    #fig.suptitle('Degree Heuristic')

    axs.scatter(degree_list, numeric_data[1:])
    axs.axhline(y=round(numeric_data[0],5), color='r', linestyle='-', label=str(legend) + ' Active Probability')
    axs.legend(loc=0, prop={'size': 12})
    axs.set_ylabel("Fixation Probability", fontsize = 12)
    axs.set_xlabel("Degree", fontsize = 12)
    path = 'Preliminary_Experiments/degree_' + str(legend) + '_Active_Probability'
    plt.savefig(path + ".png")

    plt.show()


def plot_temperature(temp_list, numeric_data, legend):
    fig, axs = plt.subplots()
    #fig.suptitle('Temperature Heuristic')

    axs.scatter(temp_list, numeric_data[1:])
    axs.axhline(y=round(numeric_data[0],5), color='r', linestyle='-', label=str(legend) + ' Active Probability')
    axs.legend(loc=0, prop={'size': 12})
    axs.set_ylabel("Fixation Probability", fontsize = 12)
    axs.set_xlabel("Temperature", fontsize = 12)

    path = 'Preliminary_Experiments/temperature_' + str(legend) + '_Active_Probability'
    plt.savefig(path.replace('\\', '\\\\') + ".png")

    plt.show()


def plot_centrality(centrality_list, numeric_data, legend):
    fig, axs = plt.subplots()
    #fig.suptitle('Centrality Heuristic')

    axs.scatter(centrality_list, numeric_data[1:])
    axs.axhline(y=round(numeric_data[0],5), color='r', linestyle='-', label=str(legend) + ' Active Probability')
    axs.legend(loc=0, prop={'size': 12})
    axs.set_ylabel("Fixation Probability", fontsize = 12)
    axs.set_xlabel("Centrality", fontsize = 12)

    path = 'Preliminary_Experiments/centrality_' + str(legend) + '_Active_Probability'
    plt.savefig(path.replace('\\', '\\\\') + ".png")

    plt.show()


def make_one_passive_simulation(graph,fitness):
    G = graph.copy()
    #Iterate all nodes and make them all active.
    #Then iterate all nodes and one by one make a single one passive and see how this changes the fixation probability
    #Further plot this marginal decrease as a function of heuristics and check for correlations

    min_iterations=20000
    simulation_data = []
    for i in range(len(G.nodes())):
        G.nodes[i]['active'] = True

    #fixation_list, simulated_fixation_prob = simulate(min_iterations, G, fitness)
    simulated_fixation_prob = numeric_fixation_probability(G,fitness)
    simulation_data.append(simulated_fixation_prob)

    for i in range(len(G.nodes())):
        G.nodes[i]['active'] = False

        #fixation_list, simulated_fixation_prob = simulate(min_iterations, G, fitness)
        simulated_fixation_prob = numeric_fixation_probability(G,fitness)
        simulation_data.append(simulated_fixation_prob)

        G.nodes[i]['active'] = True

    #Degree Heuristics
    degree_list = [v for k,v in G.degree()]
    plot_degree(degree_list,simulation_data,'All Nodes')

    #Temperature Heuristic
    temp_list = np.zeros(len(G.nodes()))
    for node1, node2, data in G.edges(data=True):
        temp_list[node1] += list(data.values())[0]
        temp_list[node2] += list(data.values())[0]
    plot_temperature(temp_list, simulation_data,'All Nodes')

    #Centrality Heuristic
    centrality_list = list(betweenness_centrality(G).values())
    plot_centrality(centrality_list,simulation_data,'All Nodes')

def make_one_active_simulation(graph,fitness):
    G = graph.copy()
    #Iterate all nodes and make them active one by one and see how this changes the fixation probability
    #Further plot this marginal increase as a function of heuristics and check for correlations

    min_iterations=20000

    simulation_data = []
    #fixation_list, simulated_fixation_prob = simulate(min_iterations,G,fitness)
    simulated_fixation_prob = numeric_fixation_probability(G,fitness)
    simulation_data.append(simulated_fixation_prob)

    for i in range(len(G.nodes())):
        G.nodes[i]['active'] = True

        #fixation_list, simulated_fixation_prob = simulate(min_iterations,G,fitness)
        simulated_fixation_prob = numeric_fixation_probability(G,fitness)
        simulation_data.append(simulated_fixation_prob)

        G.nodes[i]['active'] = False

    #Degree Heuristics
    degree_list = [v for k,v in G.degree()]
    plot_degree(degree_list,simulation_data, 'No Nodes')

    #Temperature Heuristic
    temp_list = np.zeros(len(G.nodes()))
    for node1, node2, data in G.edges(data=True):
        temp_list[node1] += list(data.values())[0]
        temp_list[node2] += list(data.values())[0]
    plot_temperature(temp_list, simulation_data, 'No Nodes')

    #Centrality Heuristic
    centrality_list = list(betweenness_centrality(G).values())
    plot_centrality(centrality_list,simulation_data, 'No Nodes')


def get_all_graphs_of_size_n(n):
    start_time = time.time()

    # graph10c.g6 is 11716571 elements long

    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, "graph_files\\graph{}.g6".format(n))
    all_graphs = nx.read_graph6(path)
    all_connected = []
    for i in all_graphs:
        if nx.is_connected(i):
            all_connected.append(i)
    end_time = time.time()
    print("loaded the graphs in ", end_time - start_time, " seconds")
    non_iso_all_graphs = []
    # for i in range(0,len(all_graphs)-1):
    #     start_time = time.time()
    #     print("how far are we?", (i/len(all_graphs))*100, "%")
    #     for j in range(0,len(all_graphs)-1):
    #         if i != j:
    #             if nx.is_isomorphic(all_graphs[i],all_graphs[j]):
    #                 print("isomorthicccc")
    #     end_time = time.time()
    #     round_time = end_time-start_time
    #     print(round_time)
    #     print("estimated time left", (round_time*(len(all_graphs)-1-i)) / (60*60), "hours")

    # print("length",len(all_graphs))
    # for i in all_graphs:
    #     if not isIsomorphicDuplicate(non_iso_all_graphs,i):
    #         non_iso_all_graphs.append(i)
    # print("length of non isomorphic graphs", len(non_iso_all_graphs))
    # print("difference", len(all_graphs)-len(non_iso_all_graphs))
    return all_connected


def isIsomorphicDuplicate(hcL, hc):
    """checks if hc is an isomorphism of any of the hc's in hcL
    Returns True if hcL contains an isomorphism of hc
    Returns False if it is not found"""
    # for each cube in hcL, check if hc could be isomorphic
    # if it could be isomorphic, then check if it is
    # if it is isomorphic, then return True
    # if all comparisons have been made already, then it is not an isomorphism and return False

    for saved_hc in hcL:
        if nx.faster_could_be_isomorphic(saved_hc, hc):
            if nx.fast_could_be_isomorphic(saved_hc, hc):
                if nx.is_isomorphic(saved_hc, hc):
                    return True
    return False


class Mutant:
    def __init__(self, fitness, id_n='mutant', color='red'):
        self.fitness = fitness
        self.color = color
        self.id_n = id_n

    def __hash__(self):
        return self.id_n

    def __cmp__(self, p):
        if self.id_n < p.id_n:
            return -1
        elif self.id_n == p.id_n:
            return 0
        else:
            return 1

    def __eq__(self, p):
        if p.id_n == self.id_n:
            return True
        else:
            return False

    def __repr__(self):
        return str(self.id_n)


class Resident:
    def __init__(self, fitness, id_n='resident', color='blue'):
        self.id_n = id_n
        self.fitness = fitness
        self.color = color

    def __hash__(self):
        return self.id_n

    def __cmp__(self, p):
        if self.id_n < p.id_n:
            return -1
        elif self.id_n == p.id_n:
            return 0
        else:
            return 1

    def __eq__(self, p):
        if p.id_n == self.id_n:
            return True
        else:
            return False

    def __repr__(self):
        return str(self.id_n)


def step_infinite_fitness(G,uniform_weights):
    #Pick reproducing node uniformly at random
    replicating_node_index = random.randint(0,len(G.nodes())-1)
    # Mutate a neighbor based on the weights of the edges
    # Find all node neighbors
    neighbors = G.edges(replicating_node_index)
    # Get the corresponding weights
    neighbor_nodes = [y for x, y in neighbors]
    if uniform_weights:
        num_of_neighbors = len(neighbors)
        index_of_node_to_mutate = random.randint(0,num_of_neighbors-1)
        node_to_mutate = neighbor_nodes[index_of_node_to_mutate]
    else:
        edge_weights = [G.get_edge_data(x, y)['weight'] for x, y in neighbors]
        node_to_mutate = random.choices(neighbor_nodes, weights=edge_weights, k=1)[0]
    # Choose one edge to walk on

    found_active = False
    if G.nodes[node_to_mutate]['active'] == True:
        found_active = True

    if G.nodes[node_to_mutate]['type'] != G.nodes[replicating_node_index]['type']:
        if G.nodes[replicating_node_index]['type'].id_n == 'resident':
            res = -1
        else:
            res = 1
        G.nodes[node_to_mutate]['type'] = G.nodes[replicating_node_index]['type']
    else:
        res = 0



    return res, found_active



def step(G,fitness, active_mutant, uniform_weights):

    if active_mutant > 0:
        chosen_node = False
        while not chosen_node:
            u = random.randint(0,len(G.nodes())-1)
            u_fitness = 1 + G.nodes[u]['type'].fitness
            chosen_node = random.choices([True,False], weights=[u_fitness / (1 + fitness), 1 - (u_fitness / (1 + fitness))], k=1)[0]

    else:
        u = random.randint(0,len(G.nodes())-1)

    replicating_node_index = u

    # Mutate a neighbor based on the weights of the edges
    # Find all node neighbors
    neighbors = G.edges(replicating_node_index)
    # Get the corresponding weights
    neighbor_nodes = [y for x, y in neighbors]
    if uniform_weights:
        num_of_neighbors = len(neighbors)
        node_to_mutate = random.choice(neighbor_nodes)
        # node_to_mutate = neighbor_nodes[index_of_node_to_mutate]
    else:
        edge_weights = [G.get_edge_data(x, y)['weight'] for x, y in neighbors]
        node_to_mutate = random.choices(neighbor_nodes, weights=edge_weights, k=1)[0]
    # Choose one edge to walk on

    if G.nodes[node_to_mutate]['type'] != G.nodes[replicating_node_index]['type']:
        if G.nodes[replicating_node_index]['type'].id_n == 'resident':
            if G.nodes[node_to_mutate]['active']:
                active_mutant -= 1
            res = -1
        else:
            if G.nodes[node_to_mutate]['active']:
                active_mutant += 1
            res = 1
        G.nodes[node_to_mutate]['type'] = G.nodes[replicating_node_index]['type']
    else:
        res = 0

    return res,active_mutant

# Uniformly picks a node to initially mutate
def mutate_a_random_node(G, fitness):
    # Generate 'random' node to mutate
    node = randint(0, len(G.nodes()) - 1)
    node_type = create_mutant_node(fitness)
    G.nodes[node]['type'] = node_type

    return G.nodes[node]['active']


def create_mutant_node(fitness=1):
    return Mutant(fitness)


# Checks whether or not we have the same color in all nodes of the graph
def have_we_terminated(G):
    first_type = G.nodes[0]['type']
    for i in G.nodes():
        if G.nodes[i]['type'] != first_type:
            return False
    return True


def is_the_first_node_mutant(G):
    if G.nodes[0]['type'].id_n == 'mutant':
        return 1
    return 0


# Plotting iterations and fixation fraction
def plot_fixation_iteration(x, y, expected):
    x = range(len(y))

    plt.plot(x, y)

    # Plot expected value for well-mixed graph (0.2) - might need to change based on numeric solution
    if expected != 0: plt.axhline(y=expected, color='r', linestyle='--', label='Expected Probability')

    # Name x-axis
    plt.xlabel('Iterations',fontsize = 12)

    # Name y-axis
    plt.ylabel('Fixation/Iterations', fontsize = 12)

    # Title
    #plt.title('Fixation Fraction as a function of Iterations', fontsize = 14)

    plt.legend(loc=1, prop={'size': 12})
    plt.show()


# Computes the numerical fixation probability
def numeric_fixation_probability(G, fitness):
    number_of_nodes = len(G.nodes)
    node_list = range(0, number_of_nodes)
    all_pairs = []
    for i in range(1, number_of_nodes + 1):
        all_pairs.append(list(itertools.combinations(node_list, i)))
    markov_model_graph = Graphs.create_markov_model(G, all_pairs, fitness)
    Graphs.draw_markov_model(markov_model_graph)
    fixation_prob = compute_fixation_probability(markov_model_graph, G)

    return fixation_prob

def compute_fixation_probability_weak(G):
    size = len(G.nodes())
    A = np.zeros((size, size))
    b = np.zeros(size)
    b[size-1] = 1
    sum_of_weights_array = np.zeros(size)
    p = np.zeros((size, size))
    for node1, node2, data in G.edges(data=True):
        sum_of_weights_array[node1] += data['weight']
        p[node1][node2] = data['weight']
    for j in range(size):
        p[j] = p[j]/sum_of_weights_array[j]
    temperature_array = p.sum(axis=0)

    for i in range(size):
        for node1, node2, data in G.edges(i,data=True):
            print("What is this",node1,node2,data['weight'])
            A[i][node2] = data['weight']
        if sum_of_weights_array[i] != 0:
            A[i] = A[i] / sum_of_weights_array[i]
        A[i][i] -= temperature_array[i]

    A[size-1] = [1 for x in range(size)]
    print("A",A)
    print("b",b)

    X = np.linalg.solve(A, b)

    return X


def compute_fixation_probability(markov, G):
    rename_nodes(markov)
    size = len(markov.nodes())
    A = np.zeros((size, size))
    b = np.zeros(size)
    b[size - 1] = 1

    for node1, node2, data in markov.edges(data=True):
        name_of_node_1 = int(markov.nodes[node1]['name'][1:])
        name_of_node_2 = int(markov.nodes[node2]['name'][1:])
        weight_between_nodes = data['weight']
        if name_of_node_1 == name_of_node_2:
            if name_of_node_1 != 0 and name_of_node_1 != (size - 1):
                A[name_of_node_1][name_of_node_2] -= 1
        A[name_of_node_1][name_of_node_2] += weight_between_nodes
    #print("The matrix 123", A)
    X = np.linalg.solve(A, b)
    #print("Sol my frind", X)
    node_size = len(G.nodes())
    probabilities = X[1:node_size + 1]
    average = np.average(probabilities)
    if np.isclose(0, average):
        Graphs.draw_markov_model(markov)
        print("something is fucky")
    return average


# Give nodes name corresponding to their variable in the linear system
def rename_nodes(markov):
    # print("nodes",markov.nodes())
    values = list(range(len(markov.nodes())))
    values = [str(x) for x in values]
    counter = 0
    for i in markov.nodes():
        number = values[counter]
        markov.nodes[i]['name'] = 'x' + str(number)
        counter += 1


def simulate_infinite_fitness(n,G,lowest_acceptable_fitness = 0,uniform_weights=True):
    fixation_counter = 0
    fixation_list = list()
    number_of_nodes = len(G.nodes)
    old_graph = G.copy()
    counter = 1
    fix_prop_this_round =0
    start_time = time.time()
    while (counter < n or fix_prop_this_round <= lowest_acceptable_fitness) and not counter > 50000:
        G = old_graph.copy()
        initial_mutation = mutate_a_random_node(G, 0)

        k = 1
        found_active = initial_mutation
        #We stop if we start by mutating an active node
        terminated = initial_mutation
        while not terminated:
            res, found_active = step_infinite_fitness(G,uniform_weights)
            k += res # Step() now returns the difference in number of mutants after_step - before_step
            terminated = k == number_of_nodes or k == 0 or found_active
        if k == number_of_nodes or found_active:
            fixation_counter += 1

        fix_prop_this_round = fixation_counter / counter
        fixation_list.append(fix_prop_this_round)
        counter += 1
    print("Rounds ", counter)
    end_time = time.time()
    print("Simulation took", end_time-start_time, "Seconds")
    return fixation_list, fixation_counter / counter


def simulate(n, G, fitness_mutant,lowest_acceptable_fitness=0, uniform_weights=True):
    fixation_counter = 0
    fixation_list = list()
    number_of_nodes = len(G.nodes)
    old_graph = G.copy()
    counter = 1
    fix_prop_this_round =0
    start_time = time.time()
    while counter < n and not counter > 50000:
    # while (counter < n or fix_prop_this_round <= lowest_acceptable_fitness) and not counter > 50000:
        G = old_graph.copy()
        initial_mutant = mutate_a_random_node(G, fitness_mutant)


        k = 1
        amount_of_mutants_in_active = 1 if initial_mutant else 0
        terminated = False
        while not terminated:
            i, amount_of_mutants_in_active = step(G,fitness_mutant,amount_of_mutants_in_active,uniform_weights)
            k += i # Step() now returns the difference in number of mutants after_step - before_step
            terminated = k == number_of_nodes or k == 0
        if k == number_of_nodes:
            fixation_counter += 1


        fix_prop_this_round = fixation_counter / counter
        fixation_list.append(fix_prop_this_round)
        counter += 1
    print("Rounds ", counter)
    end_time = time.time()
    print("New Simulation took", end_time-start_time, "Seconds")
    return fixation_list, fixation_counter / counter



def make_histogram(fitness,graph_size):
    numeric_data = []
    all_graphs_of_size_n = get_all_graphs_of_size_n(str(graph_size) + "c")
    start_time = time.time()
    for i in range(0,len(all_graphs_of_size_n)-1):
        g = all_graphs_of_size_n[i]
        #Initialize the graphs
        g = Graphs.initialize_nodes_as_resident(g)

        g.nodes[0]['active'] = True

        numeric_fixation_prob = numeric_fixation_probability(g, fitness)
        numeric_data.append(numeric_fixation_prob)
        print("Progress: ", i, "/",len(all_graphs_of_size_n)-1)
    end_time = time.time()
    total_time = end_time-start_time
    print("Done, numerical analysis took", total_time, " seconds")


    simulation_prop_data = []
    start_time = time.time()
    for i in range(0,len(all_graphs_of_size_n)-1):
        g = all_graphs_of_size_n[i]
        #Initialize the graphs
        g = Graphs.initialize_nodes_as_resident(g)

        g.nodes[0]['active'] = True

        it,fix, simulation_prop = simulate(3000,g,fitness,numeric_data[i], eps=0.005)
        simulation_prop_data.append(simulation_prop)
        print("Progress: ", i, "/",len(all_graphs_of_size_n)-1)
    end_time = time.time()
    total_time = end_time-start_time
    print("Done, simulation took", total_time, " seconds")

    #Create numerical solution for fully connected graph for reference
    G = Graphs.create_complete_graph(graph_size)
    G = Graphs.initialize_nodes_as_resident(G)
    G.nodes[0]['active'] = True
    numeric_fixation_prob = numeric_fixation_probability(G, fitness)

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Numeric Probability")
    plt.ylabel("Number of Graphs")

    #We can be 'outside' the range for 2*eps
    #Might need max_value for later when trying to make a better plot
    max_value = max(max(simulation_prop_data), max(numeric_data))
    if max_value < 0.9:
        max_value += 0.1
    bin_size = np.arange(0,max_value,0.005)

    #Round numeric data to 5 decimals
    numeric_data = [round(x,10) for x in numeric_data]

    axs[0].hist(numeric_data, bins=bin_size)
    axs[0].axvline(numeric_fixation_prob, color='k', linestyle='dashed', linewidth=1,label='Complete Graph')
    axs[0].legend(loc='upper left')

    axs[1].hist(simulation_prop_data, bins=bin_size)
    axs[1].axvline(numeric_fixation_prob, color='k', linestyle='dashed', linewidth=1, label='Complete Graph')
    axs[1].legend(loc='upper left')

    plt.show()

    print("The numeric data", numeric_data)
    print("The sim data", simulation_prop_data)


    index_of_largest_fixation_prop =  numeric_data.index(max(numeric_data))
    index_of_lowest_fixation_prop = numeric_data.index(min(numeric_data))
    ampl_graph = all_graphs_of_size_n[index_of_largest_fixation_prop]
    suppr_graph = all_graphs_of_size_n[index_of_lowest_fixation_prop]
    nx.draw_circular(ampl_graph, with_labels=True)
    plt.show()
    nx.draw_circular(suppr_graph, with_labels=True)
    plt.show()


def old_step(G,fitness_distribution):
    # Get all neighboring nodes and walk on a edge based upon the weights


    # Nodes as a list
    nodes = range(0, len(G.nodes()))
    replicating_node_index = random.choices(nodes, weights=fitness_distribution, k=1)[0]
    # Mutate a neighbor based on the weights of the edges
    # Find all node neighbors
    neighbors = G.edges(replicating_node_index)
    # Get the corresponding weights
    edge_weights = [G.get_edge_data(x, y)['weight'] for x, y in neighbors]
    neighbor_nodes = [y for x, y in neighbors]
    # Choose one edge to walk on
    #print("Neighbors ", neighbor_nodes)
    node_to_mutate = random.choices(neighbor_nodes, weights=edge_weights, k=1)[0]
    #print("Node ", node_to_mutate)
    if G.nodes[node_to_mutate]['type'] != G.nodes[replicating_node_index]['type']:
        if G.nodes[replicating_node_index]['type'].id_n == 'resident':
            res = -1
        else:
            res = 1
        G.nodes[node_to_mutate]['type'] = G.nodes[replicating_node_index]['type']
        node_fitness = G.nodes[node_to_mutate]['type'].fitness
        is_active = G.nodes[node_to_mutate]['active']
        # Multiplier for node
        multiplier = 1 if is_active else 0
        fitness_distribution[node_to_mutate] = 1 + multiplier * node_fitness
    else:
        res = 0

    return res,fitness_distribution

def old_simulate(n, G, fitness_mutant,lowest_acceptable_fitness=0):
    fixation_counter = 0
    fixation_list = list()
    number_of_nodes = len(G.nodes)
    old_graph = G.copy()
    counter = 1
    fix_prop_this_round =0
    start_time = time.time()
    while counter < n and not counter > 50000:
    # while (counter < n or fix_prop_this_round <= lowest_acceptable_fitness) and not counter > 50000:
        G = old_graph.copy()
        mutate_a_random_node(G, fitness_mutant)
        #print("Whats the fitness ", fitness_mutant)
        #print("The graph ", G.nodes(data=True))
        # Choose a node based on fitness and the multiplier
        fitness_distribution = list()
        for i in G.nodes():
            # The below logic implements the fact that only active nodes can take advantage of their multiplier
            # Fitness
            fitness = G.nodes[i]['type'].fitness
            #print("Fitness ", fitness)
            is_active = G.nodes[i]['active']
            #print("Is active ", is_active)
            # Multiplier for node
            multiplier = 1 if is_active else 0
            fitness_distribution.append(1 + multiplier * fitness)
        #print("Distribution ", fitness_distribution)

        # Does a Moran Step whenever we do not have the same color in the graph
        k = 1
        terminated = False
        while not terminated:
            i, fitness_distribution = old_step(G,fitness_distribution)
            k += i # Step() now returns the difference in number of mutants after_step - before_step
            terminated = k == number_of_nodes or k == 0
        if k == number_of_nodes:
            fixation_counter += 1
        # while not have_we_terminated(G):
        #     step(G)
        # fixation_counter += is_the_first_node_mutant(G)
        fix_prop_this_round = fixation_counter / counter
        fixation_list.append(fix_prop_this_round)
        counter += 1
    print("Rounds ", counter)
    end_time = time.time()
    print("Old Simulation took", end_time-start_time, "Seconds")
    return fixation_list, fixation_counter / counter

if __name__ == "__main__":
    fitness = 0
    graph_size = 5

    # G = Graphs.create_circle_graph(graph_size)
    #G = Graphs.create_complete_graph(graph_size)
    G = Graphs.create_star_graph(graph_size)
    #G = Graphs.create_karate_club_graph()
    #G = nx.barabasi_albert_graph(10, 3, seed=None)

    #all_graphs_of_size_n = get_all_graphs_of_size_n("6c")
    G = Graphs.initialize_nodes_as_resident(G)
    for i in range(len(G.nodes())):
        G.nodes[i]['active'] = True
    weak = compute_fixation_probability_weak(G)

    strong = numeric_fixation_probability(G,fitness)

    print("Weak", weak.round(3))
    print("Mean of weak", np.mean(weak))
    print("Strong",strong)



    #Graphs.draw_graph(G)

    # fixation_list, simulated_fixation_prob = simulate(100,G,fitness)
    # old_fixation_list, old_simulated_fixation_prob = old_simulate(100,G,fitness)

    # plot_fixation_iteration(0,fixation_list,0)
    # plot_fixation_iteration(0,old_fixation_list,0)

