import matplotlib.pyplot as plt
import networkx as nx

import Moran_Process as mp


def create_complete_graph(size):
    G = nx.complete_graph(size)
    return G


def create_star_graph(size):
    G = nx.star_graph(size)
    # G.nodes[0]['multiplier'] = 1.1
    return G


def create_karate_club_graph():
    G = nx.karate_club_graph()
    # G.nodes[0]['multiplier'] = 10
    return G


def draw_graph(G):
    # The drawn colors are created from whether the node is a resident or a mutant
    colors = [G.nodes[i]['type'].color for i in G.nodes()]
    nx.draw_circular(G, with_labels=True, node_color=colors)
    plt.show()


def draw_markov_model(G):
    nx.draw_circular(G, with_labels=True)
    plt.show()


def create_markov_model(G, all_pairs, fitness):
    extinction_node = 'extinct'
    markov = nx.DiGraph()
    markov.add_node(extinction_node)
    markov.add_edge(extinction_node, extinction_node)

    # Initial setup from extinction node to nodes with single mutant
    # We assume that all nodes in the moran graph have some connection (not isolated)
    for i in all_pairs[0]:
        node = str(i)
        markov.add_node(node)
        markov.add_edge(node, extinction_node)

        # Make selfloop for the initial nodes
        markov.add_edge(node, node)

    # This clusterfuck is off limits....
    # We go through the list of all subsets of G
    for h in range(1, len(all_pairs)):
        # extract a list of all subsets of given size
        all_sets_of_size_h = all_pairs[h]
        # Each subset must have a node in our markov model
        for set_of_mutants in all_sets_of_size_h:
            markov.add_node(str(set_of_mutants))
            # For each of the added nodes we also want to connect it to the previous nodes
            for previous_element in all_pairs[h - 1]:
                # We only bother with adding an edge if the
                # difference in the nodes is at most 1
                node_set_difference = (set(set_of_mutants) - set(previous_element))
                is_there_more_than_one_new_node = len(node_set_difference) > 1
                if not is_there_more_than_one_new_node:
                    for x in previous_element:
                        y = next(iter(node_set_difference))  # Get single element from node__set_difference
                        # Create an edge between nodes if the transition is
                        # possible in our "moran-graph"
                        y_neighbors = list(G.neighbors(y))
                        # If the node being removed/added between previous_element and set_of_mutants has a neighbor
                        # which is not a mutant, then we should create an backwards edge
                        backwards_edge = any([h not in set_of_mutants for h in y_neighbors])
                        if backwards_edge:
                            markov.add_edge(str(set_of_mutants), str(previous_element))
                        if G.has_edge(x, y):
                            markov.add_edge(str(previous_element), str(set_of_mutants))
            # Make selfloops for all nodes
            markov.add_edge(str(set_of_mutants), str(set_of_mutants))

    markov = add_weight_to_edges_markov_model(markov, G, fitness)
    return markov


def calculate_weights(k, i, graph, fitness):
    bad_char = 'extinct(,) '
    k_list = list(k)
    k_set = set([int(x) for x in k_list if x not in bad_char])
    i_list = list(i)
    i_set = set([int(x) for x in i_list if x not in bad_char])

    prob = 0
    if len(k_set) > len(i_set):
        number_of_nodes = len(graph.nodes)
        # Take each nodes fitness based on it's type and multiply it with the nodes multiplier
        fitness_temp = 0
        for id, val in enumerate(k_set):
            # The below logic implements the fact that only active nodes can take advantage of their multiplier
            # Fitness
            fitness_individual = graph.nodes[val]['type'].fitness
            # Multiplier for node
            multiplier = 1
            # Only Mutants should benefit from the multiplier
            is_mutant = graph.nodes[val]['type'].id_n == 'mutant'
            is_active = graph.nodes[val]['active']
            if is_active:
                multiplier = graph.nodes[val]['multiplier']
            fitness_temp += fitness_individual * multiplier

        total_fitness = (fitness_temp + number_of_nodes - len(k_set))

        node_k_i = next(iter(k_set - i_set))
        neighbors = list(graph.neighbors(int(node_k_i)))
        resident_neigbors = [x for x in neighbors if x not in k_set]
        for i in resident_neigbors:
            prob_of_reproducing_resident = 1 / total_fitness
            prob_of_dying_mutant = 1 / (len(list(graph.neighbors(i))))
            prob += prob_of_reproducing_resident * prob_of_dying_mutant
    elif len(k_set) < len(i_set):
        number_of_nodes = len(graph.nodes)
        # Take each nodes fitness based on it's type and multiply it with the nodes multiplier
        fitness_temp = 0
        for id, val in enumerate(k_set):
            # The below logic implements the fact that only active nodes can take advantage of their multiplier
            # Fitness
            fitness_individual = graph.nodes[val]['type'].fitness
            # Multiplier for node
            multiplier = 1
            # Only Mutants should benefit from the multiplier
            is_mutant = graph.nodes[val]['type'].id_n == 'mutant'
            is_active = graph.nodes[val]['active']
            if is_active:
                multiplier = graph.nodes[val]['multiplier']
            fitness_temp += fitness_individual * multiplier

        total_fitness = (fitness_temp + number_of_nodes - len(k_set))

        node_k_i = next(iter(i_set - k_set))
        neighbors = list(graph.neighbors(int(node_k_i)))
        mutant_neighbors = [x for x in neighbors if x in k_set]
        for i in mutant_neighbors:
            # The below logic implements the fact that only active nodes can take advantage of their multiplier
            # Fitness
            fitness_individual = graph.nodes[val]['type'].fitness
            # Multiplier for node
            multiplier = 1
            # Only Mutants should benefit from the multiplier
            is_mutant = graph.nodes[val]['type'].id_n == 'mutant'
            is_active = graph.nodes[val]['active']
            if is_active:
                multiplier = graph.nodes[val]['multiplier']
            fitness_temp += fitness_individual * multiplier


            prob_of_reproducing_mutant = (multiplier * fitness) / total_fitness
            prob_of_dying_resident = 1 / (len(list(graph.neighbors(i))))
            prob += prob_of_reproducing_mutant * prob_of_dying_resident
    return prob


def update_selfloop(node, markov):
    if markov.has_edge(node, node):
        sum_so_far = 0
        markov_edges = markov.edges(node, data=True)
        for i in markov_edges:
            if i[0] != i[1]:
                sum_so_far += i[2]['weight']
        markov.get_edge_data(node, node)['weight'] = 1 - sum_so_far


def add_weight_to_edges_markov_model(markov, graph, fitness):
    for k, i, data in markov.edges(data=True):
        data['weight'] = calculate_weights(k, i, graph, fitness)

    for i in markov.nodes():
        update_selfloop(i, markov)

    return markov


def initialize_nodes_as_resident(G,multiplier=1):
    # Initialize edge weights to be uniformly distributed
    for node1, node2, data in G.edges(data=True):
        data['weight'] = 1 / len(G.adj[node1])
    for i in G.nodes():
        # Initialize node as Resident
        nodeType = mp.Resident(1)
        G.nodes[i]['type'] = nodeType
        # Initialize multiplier to one
        G.nodes[i]['multiplier'] = multiplier
        # Initialize Active node value
        G.nodes[i]['active'] = False
    # Make graph bidirectional
    G = G.to_directed()
    return G


if __name__ == "__main__":
    create_complete_graph()
