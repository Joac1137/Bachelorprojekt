from pandas import np

import Graphs
import networkx as nx
import matplotlib.pyplot as plt

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
        weight = ((active_mutant_before*(1+fitness)/total_fitness) + (passive_mutant_before/total_fitness))*((active_nodes-active_mutant_before)/(len(G.nodes())-1))

    elif passive_mutant_before < passive_mutant_after:
        first_left = (active_mutant_before*(1+fitness)/total_fitness)
        #print("First left", first_left)
        second_left = (passive_mutant_before/total_fitness)
        #print("Second left", second_left)
        first_right = (len(G.nodes())-(active_nodes + passive_mutant_before))
        #print("First right", first_right)
        weight = (first_left + second_left)*(first_right/(len(G.nodes())-1))
        #print("W",weight)
    elif active_mutant_before > active_mutant_after:
        weight = ((len(G.nodes()) - active_mutant_before - passive_mutant_before)/total_fitness) * (active_mutant_before/(len(G.nodes())-1))

    elif passive_mutant_before > passive_mutant_after:
        weight = ((len(G.nodes()) - active_mutant_before - passive_mutant_before)/total_fitness) * (passive_mutant_before/(len(G.nodes())-1))

    #print("The weight", weight)
    return weight


def update_selfloop(node, markov):

    edges = markov.out_edges(node,data=True)
    #print("The node",node)
    #print("Edges", edges)
    sum = 0
    for i,j,data in edges:
        if data['weight'] < 0:
            print("bad edge ", i, j)
        sum += data['weight']
        #print("Haj", data['weight'])
    #print("The sum", sum)
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
                    if not (node_first + node_second == len(G.nodes)):
                        markov.add_edge(node,i)
                    if not (prev_node_first + prev_node_second == 0):
                        markov.add_edge(i,node)
            current_nodes.append(node)
            x = x - 1
            y = y + 1
        previous_nodes = current_nodes

    markov = add_weights_to_edges(markov,G,fitness,active_nodes)
    Graphs.draw_markov_model(markov)
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

def compute_fixation_probability_complete(markov, G, active_nodes):
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
    #print("The matrix", A)
    X = np.linalg.solve(A, b)
    # print("The solution", X)
    probabilities = X[1:2] if (active_nodes == 0 or active_nodes == len(G.nodes())) else X[1:3]
    print("probabilities ", probabilities)
    #The weights assume that the one with the mutant in the active node is the first probability
    start_prob = [1] if (active_nodes == 0 or active_nodes == len(G.nodes())) else [active_nodes/len(G.nodes()), 1 - active_nodes/len(G.nodes())]
    average = np.average(probabilities,weights = start_prob)
    return average

def well_mixed_experiment(G,fitness):
    active_nodes = range(0,len(G.nodes())+1)
    fixation_prob_list = []

    for i in range(0,len(G.nodes())+1):
        print("Progress", i , "of", len(G.nodes()))
        markov_chain = create_complete_markov_chain(G,i,fitness)
        fixation_prob = compute_fixation_probability_complete(markov_chain, G,i)
        fixation_prob_list.append(fixation_prob)

    plt.plot(active_nodes,fixation_prob_list)

    f = open('Complete_Graph_Experiments/complete_experiments' + str(fitness) + '_g_size_' + str(len(G.nodes)) + '.txt', '+w')
    active = ["{:2d}".format(x) for x in active_nodes]
    f.write('Active:' + ', '.join(active))
    f.write('\n')

    fixation = ["{0:10.50f}".format(x) for x in fixation_prob_list]
    f.write('Fixation probabilities: ' + ', '.join(fixation))

    # Name x-axis
    plt.xlabel('Active Nodes')

    # Name y-axis
    plt.ylabel('Fixation Probability')

    # Title
    plt.title('Fixation Probability as a function of Active Nodes with fitness of ' + str(fitness))
    plt.legend(loc=1, prop={'size': 6})
    plt.show()
    return active_nodes, fixation_prob_list


if __name__ == '__main__':
    multiplier = 1
    graph_size =5


    G = Graphs.create_complete_graph(graph_size)
    Graphs.initialize_nodes_as_resident(G,multiplier)
    Graphs.draw_graph(G)

    markov_chain = create_complete_markov_chain(G,1,1)
    Graphs.draw_markov_model(markov_chain)
    """
    fixation_prob = compute_fixation_probability_complete(markov_chain, G,active_nodes)
    print("The fixation prob", fixation_prob)

    # fitness = 0.01
    # active_nodes, fixation_prob_list_01 = well_mixed_experiment(G,fitness)
    # fitness = 0.05
    # active_nodes, fixation_prob_list_05 = well_mixed_experiment(G,fitness)
    # fitness = 0.1
    # active_nodes, fixation_prob_list_1 = well_mixed_experiment(G,fitness)
    # fitness = 0.2
    # active_nodes, fixation_prob_list_2 = well_mixed_experiment(G,fitness)
    fitness = 0.3
    active_nodes, fixation_prob_list_3 = well_mixed_experiment(G,fitness)

    # plt.plot(active_nodes,fixation_prob_list_01, label='0.01')
    # plt.plot(active_nodes,fixation_prob_list_05, label='0.05')
    # plt.plot(active_nodes,fixation_prob_list_1, label='0.1')
    # plt.plot(active_nodes,fixation_prob_list_2, label='0.2')
    print("active ",active_nodes)
    print("fix ", fixation_prob_list_3)
    print("difference ", [fixation_prob_list_3[i]-fixation_prob_list_3[i-1] for i in range(1,len(fixation_prob_list_3))])
    plt.plot(active_nodes,fixation_prob_list_3, label='0.3')

    plt.xlabel('Active Nodes')
    plt.ylabel('Fixation Probability')
    plt.legend()
    plt.show()


    
    n = 20000

    G.nodes[0]['active'] = True
    #G.nodes[1]['active'] = True
    #G.nodes[2]['active'] = True

    numeric_fixation_prob = numeric_fixation_probability(G, fitness)
    print("The real numeric prob", numeric_fixation_prob)"""



