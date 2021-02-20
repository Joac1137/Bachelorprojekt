import networkx as nx
import matplotlib.pyplot as plt
import Moran_Process as mp


def createCompleteGraph():
    G = nx.complete_graph(5)
    G = initializeNodesAsResident(G)
    # drawGraph(G)

    return G


def create_star_graph():
    G = nx.star_graph(3)
    G = initializeNodesAsResident(G)
    drawGraph(G)
    G.nodes[0]['multiplier'] = 1.1
    return G


def createKarateClubGraph():
    G = nx.karate_club_graph()
    G = initializeNodesAsResident(G)
    G.nodes[0]['multiplier'] = 10
    # drawGraph(G)
    return G


def drawGraph(G):
    # The drawn colors are created from whether the node is a resident or a mutant
    colors = [G.nodes[i]['type'].color for i in G.nodes()]

    nx.draw_circular(G, with_labels=True, node_color=colors)
    plt.show()


def draw_markov_model(G):
    nx.draw(G, with_labels=True)
    plt.show()


def create_markov_model(G, all_pairs):
    extinction_node = 'extinct'
    markov = nx.DiGraph()
    markov.add_node(extinction_node)

    # Inital setup from extinction node to nodes with single mutant
    # We assume that all nodes in the moran graph have some connection (not isolated)
    for i in all_pairs[0]:
        node = str(i)
        markov.add_node(node)
        markov.add_edge(node, extinction_node)

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

    return markov


def initializeNodesAsResident(G):
    # Initialize edge weights to be uniformly distributed
    for node1, node2, data in G.edges(data=True):
        data['weight'] = 1 / len(G.adj[node1])

    for i in G.nodes():
        # Initialize node as Resident
        nodeType = mp.Resident(1)
        G.nodes[i]['type'] = nodeType

        # Initialize multiplier to one
        G.nodes[i]['multiplier'] = 1

        # Print edges and their weight
        # print(G.adj[i])

    # Make graph bidirectional
    G = G.to_directed()
    return G


if __name__ == "__main__":
    createCompleteGraph()
