from random import uniform, random, sample, randint, choices
import itertools

import networkx as nx

import Graphs
import matplotlib.pyplot as plt
import random
import numpy as np


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


# Mutate neighbor
def step(G):
    # Get all neighboring nodes and walk on a edge based upon the weights

    # Choose a node based on fitness and the multiplier
    fitness_distribution = list()
    for i in G.nodes():
        # Fitness
        fitness = G.nodes[i]['type'].fitness
        # Multiplier for node
        multiplier = G.nodes[i]['multiplier']
        # Only Mutants should benefit of the multiplier
        if G.nodes[i]['type'].id_n == 'resident':
            multiplier = 1
        fitness_distribution.append(multiplier * fitness)
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
    node_to_mutate = random.choices(neighbor_nodes, weights=edge_weights, k=1)[0]
    G.nodes[node_to_mutate]['type'] = G.nodes[replicating_node_index]['type']


# Uniformly picks a node to initially mutate
def mutate_a_random_node(G):
    # Generate 'random' node to mutate
    node = randint(0, len(G.nodes()) - 1)
    node_type = create_mutant_node(1)
    G.nodes[node]['type'] = node_type


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
    plt.plot(x, y)

    # Plot expected value for well-mixed graph (0.2) - might need to change based on numeric solution
    plt.axhline(y=expected, color='r', linestyle='-', label='Expected Probability')

    # Name x-axis
    plt.xlabel('Iterations')

    # Name y-axis
    plt.ylabel('Fixation/Iterations')

    # Title
    plt.title('Fixation Fraction as a function of Iterations')
    plt.legend(loc=1, prop={'size': 6})
    plt.show()


# Computes the numerical fixation probability
def numeric_fixation_probability(G, fitness):
    if len(G.nodes) > 10:
        print("Do you wanna fucking die?")
        print("Smaller graph please....")
        return 0
    number_of_nodes = len(G.nodes)
    node_list = range(0, number_of_nodes)
    all_pairs = []
    for i in range(1, number_of_nodes + 1):
        all_pairs.append(list(itertools.combinations(node_list, i)))
    markov_model_graph = Graphs.create_markov_model(G, all_pairs, fitness)
    Graphs.draw_markov_model(markov_model_graph)
    fixation_prob = compute_fixation_probability(markov_model_graph, G)
    return fixation_prob


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
    X = np.linalg.solve(A, b)
    node_size = len(G.nodes())
    probabilities = X[1:node_size + 1]
    average = np.average(probabilities)
    return average


# Give nodes name corresponding to their variable in the linear system
def rename_nodes(markov):
    values = list(range(len(markov.nodes())))
    values = [str(x) for x in values]
    counter = 0
    for i in markov.nodes():
        number = values[counter]
        markov.nodes[i]['name'] = 'x' + str(number)
        counter += 1


def simulate(n, G):
    fixation_counter = 0
    fixation_list = list()
    iteration_list = list(range(0, n))
    for i in range(1, n + 1):
        mutate_a_random_node(G)
        # Does a Moran Step whenever we do not have the same color in the graph
        while not have_we_terminated(G):
            step(G)
        fixation_counter += is_the_first_node_mutant(G)
        fixation_list.append(fixation_counter / i)
        Graphs.initialize_nodes_as_resident(G)
    return iteration_list, fixation_list, fixation_counter / n


if __name__ == "__main__":
    #G = Graphs.create_complete_graph(3)
    # G = Graphs.create_karate_club_graph()
    G = Graphs.create_star_graph(5)
    iteration_list, fixation_list, simulated_fixation_prob = simulate(10000, G)

    numeric_fixation_prob = numeric_fixation_probability(G, 1)
    plot_fixation_iteration(iteration_list, fixation_list, numeric_fixation_prob)
    print("Simulated fixation probability = ", simulated_fixation_prob)
    print("Numeric fixation probability = ", numeric_fixation_prob)
    print("Difference = ", abs(simulated_fixation_prob - numeric_fixation_prob))
