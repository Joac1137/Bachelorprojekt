import math

from numpy import argmax
from pandas import np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

import Graphs
import networkx as nx
import matplotlib.pyplot as plt

from Moran_Process import numeric_fixation_probability, simulate, plot_fixation_iteration


def calculate_markov_weights(i, j, G, fitness, active_leaves):
    i_split = i.split(',')
    j_split = j.split(',')
    x_from = int(i_split[0])
    y_from = int(i_split[1])
    z_from = int(i_split[2])
    x_to = int(j_split[0])
    y_to = int(j_split[1])
    z_to = int(j_split[2])

    """
    print("Active before", active_mutant_before)
    print("Passive before", passive_mutant_before)
    print("Active after", active_mutant_after)
    print("Passive adter", passive_mutant_after)
    """
    N = len(G.nodes())
    total_fitness = (x_from+z_from)*(1 + fitness) + (N-1) - (x_from + z_from)
    weight = 0
    if x_from < x_to:
        weight = (z_from*(1 + fitness) / total_fitness)*((active_leaves-x_from)/(N-1))
    elif x_from > x_to:
        weight = ((1-z_from) / total_fitness) * (x_from / (N-1))
    elif y_from < y_to:
        weight = (z_from*(1 + fitness) / total_fitness)*((N - 1 - active_leaves- y_from)/(N-1))
    elif y_from > y_to:
        weight = ((1-z_from) / total_fitness) * (y_from / (N-1))
    elif z_from < z_to:
        weight = (x_from*(1 + fitness) / total_fitness) + (y_from/total_fitness)
    elif z_from > z_to:
        weight = (N-1-x_from-y_from) / total_fitness

    return weight



def update_selfloop(node, markov):
    edges = markov.out_edges(node,data=True)
    #print("The node",node)
    #print("Edges", edges)
    sum = 0
    for i,j,data in edges:
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

def create_star_markov_chain(G, active_leaves,fitness):
    N = len(G.nodes)
    extinction_node = '0,0,0'
    markov = nx.DiGraph()
    markov.add_node(extinction_node)
    markov.add_edge(extinction_node,extinction_node)
    markov[extinction_node][extinction_node]['weight'] = 1
    previous_nodes = [extinction_node]
    for i in range(1,N+1):
        #print("Iter",i,"of",N+1)
        node_list = create_integer_partitions(i,active_leaves,N)
        for node in node_list:
            markov.add_node(node)
            #Graphs.draw_markov_model(markov)
            values = node.split(',')
            x = int(values[0])
            y = int(values[1])
            z = int(values[2])
            for prev_node in previous_nodes:
                values = prev_node.split(',')
                prev_x = int(values[0])
                prev_y = int(values[1])
                prev_z = int(values[2])

                if (x-prev_x)**2 + (y-prev_y)**2 + (z-prev_z)**2 <= 1:

                    change = argmax([abs(x-prev_x), abs(y-prev_y), abs(z-prev_z)])
                    #print("Change", change)
                    #print("Node", x,y,z)
                    #print("Prev Node", prev_x,prev_y,prev_z)
                    if change == 0:
                        #We have a change in x
                        if x > prev_x and prev_z == 1:
                            markov.add_edge(prev_node,node)
                        if x > prev_x and z == 0:
                            markov.add_edge(node,prev_node)
                    elif change == 1:
                        #We have a change in y
                        if y > prev_y and prev_z == 1:
                            markov.add_edge(prev_node,node)
                        if y > prev_y and z == 0:
                            markov.add_edge(node,prev_node)
                    elif change == 2:
                        #We have change in z
                        if z > prev_z and prev_x+prev_y > 0:
                            markov.add_edge(prev_node,node)
                        if z > prev_z and N - 1 - x - y > 0:
                            markov.add_edge(node,prev_node)

        previous_nodes = node_list

    markov = add_weights_to_edges(markov,G,fitness,active_leaves)
    Graphs.draw_markov_model(markov)
    return markov


def create_integer_partitions(number,active_leaves,N):
    answer = []
    for z in range(2):
        number = number - z
        for x in range(number+1):
            y = number-x
            if x <= active_leaves and y < N-active_leaves:
                #Case for when center is zero but we have mutants in all leaves
                if not (z == 0 and x + y > N-2):
                    node_string = str(x) + "," + str(y) + "," + str(z)
                    answer.append(node_string)
    return answer


# Give nodes name corresponding to their variable in the linear system
def rename_nodes(markov):
    values = list(range(len(markov.nodes())))
    values = [str(x) for x in values]
    counter = 0
    for i in markov.nodes():
        number = values[counter]
        markov.nodes[i]['name'] = 'x' + str(number)
        counter += 1

def compute_fixation_probability_star(markov, G, active_leaves):
    rename_nodes(markov)
    size = len(markov.nodes())
    #A = np.zeros((size, size))
    b_value = [1]
    b_row = [size-1]

    #b = np.zeros(size)

    row = []
    col =[]
    value = []

    #b[size - 1] = 1
    N = len(G.nodes())
    #print("Nodes", markov.nodes(data=True))
    for node1, node2, data in markov.edges(data=True):
        name_of_node_1 = int(markov.nodes[node1]['name'][1:])
        name_of_node_2 = int(markov.nodes[node2]['name'][1:])


        weight_between_nodes = data['weight']
        if name_of_node_1 == name_of_node_2:
            if name_of_node_1 != 0 and name_of_node_1 != (size - 1):
                #A[name_of_node_1][name_of_node_2] -= 1
                row.append(name_of_node_1)
                col.append(name_of_node_2)
                value.append(-1)
        #A[name_of_node_1][name_of_node_2] += weight_between_nodes
        row.append(name_of_node_1)
        col.append(name_of_node_2)
        value.append(weight_between_nodes)

    A = csr_matrix((value, (row, col)), shape = (size, size))
    b = csr_matrix((b_value, (b_row, [0])), shape = (size, 1))
    #print("The matrix", A)

    X = spsolve(A, b)
    #X = np.linalg.solve(A, b)
    #print("The solution", X)
    probabilities = X[1:3] if active_leaves == 0 else X[1:4]
    #The weights assume that the one with the mutant in the active node is the first probability
    start_prob = [(N-1)/N,1/N] if active_leaves == 0 else [(N-1-active_leaves)/N, active_leaves / N, 1/N]
    #print("Probs", probabilities)
    #print("Star prob", start_prob)
    average = np.average(probabilities,weights = start_prob)
    return average


def star_experiment(G,fitness):
    active_nodes = range(0,len(G.nodes()) - 1)
    fixation_prob_list = []

    for i in range(0,len(G.nodes()) - 1):
        print("Progress", i , "of", len(G.nodes()) - 1)
        markov_chain = create_star_markov_chain(G,i,fitness)
        fixation_prob = compute_fixation_probability_star(markov_chain, G,i)
        fixation_prob_list.append(fixation_prob)

    plt.plot(active_nodes,fixation_prob_list)

    f = open('Star_Graph_Experiments/star_experiments' + str(fitness)+ '_g_size_' + str(len(G.nodes)) + '.txt', '+w')
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
    graph_size = 5

    G = Graphs.create_star_graph(graph_size)
    Graphs.initialize_nodes_as_resident(G,multiplier)

    """
    #Graphs.draw_graph(G)
    markov = create_star_markov_chain(G,active_leaves,fitness)
    #Graphs.draw_markov_model(markov)

    fixation_prob = compute_fixation_probability_star(markov, G,active_leaves)
    print("The fixation prob", fixation_prob)"""


    fitness = 0.01
    active_nodes, fixation_prob_list_01 = star_experiment(G,fitness)
    # fitness = 0.05
    # active_nodes, fixation_prob_list_05 = star_experiment(G,fitness)
    # fitness = 0.1
    # active_nodes, fixation_prob_list_1 = star_experiment(G,fitness)
    # fitness = 0.2
    # active_nodes, fixation_prob_list_2 = star_experiment(G,fitness)
    # fitness = 0.3
    # active_nodes, fixation_prob_list_3 = star_experiment(G,fitness)

    plt.plot(active_nodes,fixation_prob_list_01, label='0.01')
    # plt.plot(active_nodes,fixation_prob_list_05, label='0.05')
    # plt.plot(active_nodes,fixation_prob_list_1, label='0.1')
    # plt.plot(active_nodes,fixation_prob_list_2, label='0.2')
    # plt.plot(active_nodes,fixation_prob_list_3, label='0.3')

    plt.xlabel('Active Nodes')
    plt.ylabel('Fixation Probability')
    plt.legend()
    plt.show()


    """    
    n = 20000

    G.nodes[0]['active'] = True
    G.nodes[1]['active'] = True
    #G.nodes[2]['active'] = True

    numeric_fixation_prob = numeric_fixation_probability(G, fitness)
    print("The real numeric prob", numeric_fixation_prob)
    """