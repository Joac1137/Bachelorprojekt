import nx as nx

import Graphs
import networkx as nx

def create_circle_markov_chain(G):
    N = len(G.nodes)
    markov = nx.DiGraph()

    for i in range(0,N):
        for j in range(0,N):
            initial_node = str(i) + ',' + str(j)
            markov.add_node(initial_node)

    print("Markov", markov.nodes())
    for i in markov.nodes():
        for j in markov.nodes():
            first_number_i = int(i.split(',')[0])
            second_number_i = int(i.split(',')[1])

            first_set = set(range(min(first_number_i,second_number_i), max(first_number_i,second_number_i)+1))
            print("First set of nodes", first_set)

            first_number_j = int(j.split(',')[0])
            second_number_j = int(j.split(',')[1])
            second_set = set(range(min(first_number_j,second_number_j), max(first_number_j,second_number_j)+1))
            print("Second set of nodes", second_set)








    Graphs.draw_markov_model(markov)

if __name__ == '__main__':
    graph_size = 3
    active_nodes = 0
    continuous_active_nodes = set(range(0,active_nodes))
    every_other_active_nodes = set()

    for x in range(0,graph_size-1,2):
        if len(every_other_active_nodes) < active_nodes:
            every_other_active_nodes.add(x)
    for x in range(0,graph_size):
        if len(every_other_active_nodes) < active_nodes:
            every_other_active_nodes.add(x)


    print("Continuous active nodes", continuous_active_nodes)
    print("Every other node is active", every_other_active_nodes)

    G = Graphs.create_circle_graph(graph_size)
    Graphs.initialize_nodes_as_resident(G,1)

    Graphs.draw_graph(G)
    markov = create_circle_markov_chain(G)
    #markov = create_star_markov_chain(G,active_leaves,fitness)
    #Graphs.draw_markov_model(markov)

    #fixation_prob = compute_fixation_probability_star(markov, G,active_leaves)
    #print("The fixation prob", fixation_prob)

