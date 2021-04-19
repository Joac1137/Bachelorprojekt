from pandas import np

import Graphs
import networkx as nx

from Moran_Process import numeric_fixation_probability, simulate, plot_fixation_iteration


def calculate_markov_weights(i, j, G, fitness, active_nodes):
    active_mutant_before = int(i.split(',')[0])
    passive_mutant_before = int(i.split(',')[1])
    active_mutant_after = int(j.split(',')[0])
    passive_mutant_after = int(j.split(',')[1])

    """
    print("Active before", active_mutant_before)
    print("Passive before", passive_mutant_before)
    print("Active after", active_mutant_after)
    print("Passive adter", passive_mutant_after)
    """

    total_fitness = active_mutant_before*(1 + fitness) + len(G.nodes()) - active_mutant_before
    weight = 0
    if active_mutant_before < active_mutant_after:
        weight = ((active_mutant_before*(1+fitness)/total_fitness) + (passive_mutant_before/total_fitness))*((active_nodes-active_mutant_before)/len(G.nodes()))

    elif passive_mutant_before < passive_mutant_after:
        first_left = (active_mutant_before*(1+fitness)/total_fitness)
        #print("First left", first_left)
        second_left = (passive_mutant_before/total_fitness)
        #print("Second left", second_left)
        first_right = (len(G.nodes())-(active_mutant_before + passive_mutant_before))
        #print("First right", first_right)
        weight = (first_left + second_left)*(first_right/len(G.nodes()))
        #print("W",weight)
    elif active_mutant_before > active_mutant_after:
        weight = ((len(G.nodes()) - active_mutant_before - passive_mutant_before)/total_fitness) * (active_mutant_before/len(G.nodes()))

    elif passive_mutant_before > passive_mutant_after:
        weight = ((len(G.nodes()) - active_mutant_before - passive_mutant_before)/total_fitness) * (passive_mutant_before/len(G.nodes()))

    #print("The weight", weight)
    return weight


def update_selfloop(node, markov):

    edges = markov.out_edges(node,data=True)
    #print("Edges", edges)
    sum = 0
    for i,j,data in edges:
        sum += data['weight']
        #print("Haj", data['weight'])
    markov.add_edge(node,node)
    markov[node][node]['weight'] = 1 - sum


def add_weights_to_edges(markov, G, fitness,active_nodes):

    for i, j, data in markov.edges(data = True):
        data['weight'] = calculate_markov_weights(i,j,G,fitness,active_nodes)

    for i in markov.nodes():
        update_selfloop(i, markov)

    return markov


def create_complete_markov_chain(G, active_nodes,fitness):
    extinction_node = '0,0'
    markov = nx.DiGraph()
    markov.add_node(extinction_node)
    markov.add_edge(extinction_node,extinction_node)
    markov[extinction_node][extinction_node]['weight'] = 1

    previous_nodes = [extinction_node]
    for i in range(1,len(G.nodes()) + 1):
        current_nodes = []
        x = i if i <= active_nodes else active_nodes
        y = i - x

        while x >= 0 and y <= len(G.nodes()) - active_nodes:
            node = str(x) + ',' + str(y)
            markov.add_node(node)

            for i in previous_nodes:
                node_first = int(node.split(',')[0])
                node_second = int(node.split(',')[1])
                prev_node_first = int(i.split(',')[0])
                prev_node_second = int(i.split(',')[1])

                #We cannot go from node (1,0) to (0,2)
                if (node_first - prev_node_first)**2 <= 1 and (node_second - prev_node_second)**2 <= 1:
                    markov.add_edge(node,i)
                    markov.add_edge(i,node)
            current_nodes.append(node)
            x = x - 1
            y = y + 1
        previous_nodes = current_nodes

    markov = add_weights_to_edges(markov,G,fitness,active_nodes)
    return markov


# Give nodes name corresponding to their variable in the linear system
def rename_nodes(markov):
    values = list(range(len(markov.nodes())))
    values = [str(x) for x in values]
    counter = 0
    for i in markov.nodes():
        number = values[counter]
        markov.nodes[i]['name'] = 'x' + str(number)
        counter += 1

def compute_fixation_probability_complete(markov, G):
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
    print("The matrix", A)
    X = np.linalg.solve(A, b)
    print("The solution", X)
    probabilities = X[1:active_nodes + 2] if active_nodes <= 1 else X[1:3]
    average = np.average(probabilities)
    return average


if __name__ == '__main__':
    fitness = 2
    multiplier = 1
    graph_size = 3
    active_nodes = 1

    G = Graphs.create_complete_graph(graph_size)
    Graphs.initialize_nodes_as_resident(G,multiplier)
    Graphs.draw_graph(G)

    markov_chain = create_complete_markov_chain(G,active_nodes,fitness)
    print("The edges", markov_chain.edges(data = True))

    Graphs.draw_markov_model(markov_chain)

    fixation_prob = compute_fixation_probability_complete(markov_chain, G)
    print("The fixation prob", fixation_prob)


    n = 20000

    G.nodes[0]['active'] = True
    #G.nodes[1]['active'] = True
    #G.nodes[2]['active'] = True
    fixation_list, simulated_fixation_prob = simulate(n, G,fitness)
    iteration_list = list(range(0, n))
    numeric_fixation_prob = numeric_fixation_probability(G, fitness)
    print("The real numeric prob", numeric_fixation_prob)

    plot_fixation_iteration(iteration_list, fixation_list,numeric_fixation_prob)


