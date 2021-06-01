import time

import networkx as nx
from numpy import mat
from pandas import np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import Graphs
import networkx as nx
from enum import Enum
import math

from Moran_Process import numeric_fixation_probability


def size_of_inteval(start,end,N):
    if start == end:
        return 1
    elif start < end:
        return end - start + 1
    elif start > end:
        return N - start + end + 1


def calculate_total_fitness(G,node,fitness):
    total_fitness = 0
    if node == 'extinct':
        total_fitness = len(G.nodes())
    else:
        first_number_i = int(node.split(',')[0])
        second_number_i = int(node.split(',')[1])

        number_of_mutant_in_active = 0

        mutants = list()
        if first_number_i == second_number_i:
            mutants = [first_number_i]
        elif first_number_i < second_number_i:
            mutants = list(range(first_number_i,second_number_i + 1))
        elif first_number_i > second_number_i:
            mutants = list(range(first_number_i,len(G.nodes()))) + list(range(0,second_number_i + 1))

        for i in mutants:
            if G.nodes[i]['active']:
                number_of_mutant_in_active += 1

        total_fitness = number_of_mutant_in_active*(1 + fitness) + len(G.nodes()) - number_of_mutant_in_active

    return total_fitness

def calculate_markov_weights(i, j, G, fitness):
    N = len(G.nodes())

    weight = 0
    if i != j:

        first_number_i = int(i.split(',')[0])
        second_number_i = int(i.split(',')[1])
        length_of_from_node = size_of_inteval(first_number_i,second_number_i,N)

        if j == 'extinct' and first_number_i == second_number_i:
            total_fitness = calculate_total_fitness(G,i,fitness)
            r_value = 0
            weight = 2*(r_value + 1) / total_fitness * 1/2
        else:

            first_number_j = int(j.split(',')[0])
            second_number_j = int(j.split(',')[1])
            length_of_to_node = size_of_inteval(first_number_j,second_number_j,N)

            if length_of_to_node - length_of_from_node > 0:
                if (second_number_i + 1) % N == second_number_j and first_number_i == first_number_j:
                    total_fitness = calculate_total_fitness(G,i,fitness)
                    r_value = fitness if G.nodes[second_number_i]['active'] else 0
                    weight = (r_value + 1) / total_fitness * 1/2

                elif (first_number_i + N - 1) % N == first_number_j and second_number_i == second_number_j:
                    total_fitness = calculate_total_fitness(G,i,fitness)
                    r_value = fitness if G.nodes[first_number_i]['active'] else 0
                    weight = (r_value + 1) / total_fitness * 1/2

            elif length_of_from_node - length_of_to_node > 0:
                if (second_number_j + 1) % N == second_number_i and first_number_i == first_number_j:
                    total_fitness = calculate_total_fitness(G,i,fitness)
                    r_value = 0
                    weight = (r_value + 1) / total_fitness * 1/2
                elif (first_number_j + N - 1) % N == first_number_i and second_number_i == second_number_j:
                    total_fitness = calculate_total_fitness(G,i,fitness)
                    r_value = 0
                    weight = (r_value + 1) / total_fitness * 1/2

    return weight

def update_selfloop(node, markov):
    edges = markov.out_edges(node,data=True)
    sum = 0
    for i,j,data in edges:
        sum += data['weight']
    markov.add_edge(node,node)
    markov[node][node]['weight'] = 1 - sum


def add_weights_to_edges(markov, G, fitness):
    for i, j, data in markov.edges(data = True):
        data['weight'] = calculate_markov_weights(i,j,G,fitness)

    for i in markov.nodes():
        update_selfloop(i, markov)

    return markov


def create_circle_markov_chain(G,fitness):
    N = len(G.nodes)
    markov = nx.DiGraph()
    extinction_node = 'extinct'
    markov.add_node(extinction_node)
    markov.add_edge(extinction_node,extinction_node)
    markov[extinction_node][extinction_node]['weight'] = 1

    for i in range(0,N):
        for j in range(0,N):
            initial_node = str(i) + ',' + str(j)
            markov.add_node(initial_node)
            if i == j:
                markov.add_edge(initial_node,extinction_node)

    iter_nodes = iter(markov.nodes())
    next(iter_nodes)
    for i in iter_nodes:
        iter_nodes_inner = iter(markov.nodes())
        next(iter_nodes_inner)
        for j in iter_nodes_inner:
            first_number_i = int(i.split(',')[0])
            second_number_i = int(i.split(',')[1])

            first_number_j = int(j.split(',')[0])
            second_number_j = int(j.split(',')[1])

            if (first_number_i + N - 1) % N != second_number_i:
                if (first_number_j + 1) % N == first_number_i and second_number_i == second_number_j:
                    markov.add_edge(i,j)
                    if (first_number_j + N - 1) % N != second_number_j:
                        markov.add_edge(j,i)
                elif (second_number_i + 1) % N == second_number_j and first_number_i == first_number_j:
                    markov.add_edge(i,j)
                    if (first_number_j + N - 1) % N != second_number_j:
                        markov.add_edge(j,i)
            else:
                markov.add_edge(i,i)

    markov = add_weights_to_edges(markov,G,fitness)
    #Graphs.draw_markov_model(markov)

    return markov

class Active_Node_Setup(Enum):
    continuous = 1
    evenly_distributed = 2
    every_other = 3

def initialize_active_nodes(G, setup, active_nodes):
    if setup == Active_Node_Setup(1):
        for i in range(0,active_nodes):
            G.nodes[i]['active'] = True
        #print("Active Nodes", list(range(0,active_nodes)))

    elif setup == Active_Node_Setup(2):
        #list_of_active = [(x*round(len(G.nodes())/active_nodes)) % len(G.nodes()) for x in range(active_nodes)]
        list_of_active = []
        if active_nodes > 0:
            number = round(len(G.nodes())/active_nodes) if round(len(G.nodes())/active_nodes) >= 2 else 2
            list_of_active = list(range(0,len(G.nodes()),number))
        if len(list_of_active) < active_nodes:
            for i in range(0,len(G.nodes())):
                if len(list_of_active) < active_nodes:
                    if i not in list_of_active:
                        list_of_active.append(i)
                else:
                    break
        else:
            list_of_active = list_of_active[:active_nodes]
        for i in list_of_active:
            G.nodes[i]['active'] = True
        #print("Active Nodes", list_of_active)

    elif setup == Active_Node_Setup(3):
        set_active_nodes = set()
        for i in range(0,len(G.nodes()),2):
            if len(set_active_nodes) != active_nodes:
                set_active_nodes.add(i)
        for i in range(0,len(G.nodes())):
            if len(set_active_nodes) != active_nodes:
                set_active_nodes.add(i)
            else:
                break
        list_of_active = list(set_active_nodes)
        for i in list_of_active:
            G.nodes[i]['active'] = True
        #print("Active Nodes", list_of_active)
    return G


def rename_nodes(markov):
    values = list(range(len(markov.nodes())))
    values = [str(x) for x in values]
    counter = 0
    for i in markov.nodes():
        number = values[counter]
        markov.nodes[i]['name'] = 'x' + str(number)
        counter += 1

def compute_fixation_probability_circle(markov, G):
    rename_nodes(markov)
    #print("Markov",markov.nodes(data=True))
    size = len(markov.nodes())
    A = np.zeros((size, size))
    #b_value = [1]
    #b_row = [size-1]

    b = np.zeros(size)

    #row = []
    #col =[]
    #value = []
    final_nodes = []
    start_nodes = []
    iter_nodes = iter(markov.nodes(data=True))
    next(iter_nodes)
    for i,data in iter_nodes:
        first_number_i = int(i.split(',')[0])
        second_number_i = int(i.split(',')[1])
        length_of_i_node = size_of_inteval(first_number_i,second_number_i,len(G.nodes()))
        if length_of_i_node == len(G.nodes()):
            #print("i", i)
            #print("data", data)
            number_of_node = int(markov.nodes[i]['name'][1:])
            final_nodes.append(number_of_node)
            b[number_of_node] = 1
        if first_number_i == second_number_i:
            number_of_node = int(markov.nodes[i]['name'][1:])
            start_nodes.append(number_of_node)

    #b[size - 1] = 1
    N = len(G.nodes())
    #print("Nodes", markov.nodes(data=True))
    for node1, node2, data in markov.edges(data=True):
        name_of_node_1 = int(markov.nodes[node1]['name'][1:])
        name_of_node_2 = int(markov.nodes[node2]['name'][1:])
        #print("Name 1", name_of_node_1)
        #print("Name 2", name_of_node_2)
        weight_between_nodes = data['weight']
        if name_of_node_1 == name_of_node_2:
            if name_of_node_1 != 0 and name_of_node_1 not in final_nodes:
                A[name_of_node_1][name_of_node_2] -= 1
                #row.append(name_of_node_1)
                #col.append(name_of_node_2)
                #value.append(-1)
        A[name_of_node_1][name_of_node_2] += weight_between_nodes
        #row.append(name_of_node_1)
        #col.append(name_of_node_2)
        #value.append(weight_between_nodes)

    #A = csr_matrix((value, (row, col)), shape = (size, size))
    #b = csr_matrix((b_value, (b_row, [0])), shape = (size, 1))
    #print("The A", A)
    #print("The b",b)

    #X = spsolve(A, b)
    X = np.linalg.solve(A, b)
    #print("The solution", X)
    probabilities = [X[x] for x in start_nodes]
    #The weights assume that the one with the mutant in the active node is the first probability
    #start_prob = [(N-1)/N,1/N] if active_leaves == 0 else [(N-1-active_leaves)/N, active_leaves / N, 1/N]
    #print("Probs", probabilities)
    #print("Star prob", start_prob)
    average = np.average(probabilities)
    return average

def circle_experiments(graph_size,active_nodes,fitness,setup):
    G = Graphs.create_circle_graph(graph_size)
    Graphs.initialize_nodes_as_resident(G,1)
    G = initialize_active_nodes(G,Active_Node_Setup(setup),active_nodes)
    i = []
    for j in range(graph_size):
        if G.nodes[j]['active']:
            i.append(j)
    print("Set of active nodes", i)
    #Graphs.draw_graph(G)
    markov = create_circle_markov_chain(G,fitness)

    #numeric_fixation_prob = numeric_fixation_probability(G, fitness)
    fixation_prob = compute_fixation_probability_circle(markov, G)
    return fixation_prob


def compare_active_node_choosing_strategies(graph_size, fitness):
    setup_1 = []
    setup_2 = []
    setup_3 = []
    active_node_list = list(range(0,graph_size + 1))
    before_time = time.time()
    for data in Active_Node_Setup:
        fixation_list = []
        #numeric_list = []
        for i in range(0,graph_size + 1):
            print("The i ", i)
            fixation_prob = circle_experiments(graph_size,i,fitness,data.value)
            fixation_list.append(fixation_prob)
            #numeric_list.append(numeric_fixation)

        #print("Fixation prob list", fixation_list)
        #print("Numeric Fixation list", numeric_list)
        #print("Active node list", active_node_list)

        globals()['setup_' + str(data.value)] = fixation_list

        # Name x-axis
        plt.xlabel('Active Nodes')

        # Name y-axis
        plt.ylabel('Fixation Probability')

        plt.plot(active_node_list,fixation_list)

        # Title
        plt.title('Setup {} '.format(data.name))
        plt.show()

    after_time = time.time()
    print("Plotting took", after_time-before_time,"Seconds")

    # Name x-axis
    plt.xlabel('Active Nodes')

    # Name y-axis
    plt.ylabel('Fixation Probability')

    plt.plot(active_node_list,setup_1,label = 'Setup ' + str(Active_Node_Setup(1).name))
    plt.plot(active_node_list,setup_2,label = 'Setup ' + str(Active_Node_Setup(2).name))
    plt.plot(active_node_list,setup_3,label = 'Setup ' + str(Active_Node_Setup(3).name))

    # Title
    plt.title('Fixation Probability as a Function of Active Nodes')
    plt.legend()
    plt.show()


    f = open('Circle_Graph_Experiments/cycle_experiments' + str(fitness)+ '_g_size_' + str(graph_size) + '.txt', '+w')
    f.write("Data with setup " + str(Active_Node_Setup(1).name) + "\n")
    active = ["{:2d}".format(x) for x in active_node_list]
    f.write('Active:' + ', '.join(active))
    f.write('\n')

    fixation = ["{0:10.50f}".format(x) for x in setup_1]
    f.write('Fixation probabilities: ' + ', '.join(fixation))
    f.write('\n')


    f.write("Data with setup " + str(Active_Node_Setup(2).name) + "\n")
    active = ["{:2d}".format(x) for x in active_node_list]
    f.write('Active:' + ', '.join(active))
    f.write('\n')

    fixation = ["{0:10.50f}".format(x) for x in setup_2]
    f.write('Fixation probabilities: ' + ', '.join(fixation))
    f.write('\n')



    f.write("Data with setup " + str(Active_Node_Setup(3).name) + "\n")
    active = ["{:2d}".format(x) for x in active_node_list]
    f.write('Active:' + ', '.join(active))
    f.write('\n')

    fixation = ["{0:10.50f}".format(x) for x in setup_3]
    f.write('Fixation probabilities: ' + ', '.join(fixation))
    f.write('\n')


def fixation_prob_active_nodes(graph_size,setup,fitneses):
    active_node_list = list(range(0,graph_size + 1))
    before_time = time.time()
    path = 'Circle_Graph_Experiments/cycle_fitness_experiment_new2/cycle_experiments_f_' + str(fitneses)+ '_g_size_' + str(graph_size) + '_setup_' + str(setup.name)
    f = open(path + '.txt', '+w')
    for fitness in fitneses:
        fixation_list = []
        for i in range(0,graph_size + 1):
            print('Fitness ' + str(fitness) + ' iteration ' + str(i) + ' of ' + str(graph_size))
            fixation_prob = circle_experiments(graph_size,i,fitness,setup.value)
            fixation_list.append(fixation_prob)
            if i > 0:
                if fixation_list[i]<fixation_list[i-1]:
                    print("This set of active is smaller than the previous")
        fixation = ["{0:10.50f}".format(x) for x in fixation_list]
        f.write('Fixation probabilities: ' + ', '.join(fixation))
        f.write('\n')

        f.write("Data with setup " + str(setup.name) + " and fitness " + str(fitness) + "\n")
        active = ["{:2d}".format(x) for x in active_node_list]
        f.write('Active:' + ', '.join(active))
        f.write('\n')

        # Name x-axis
        plt.xlabel('Active Nodes')

        # Name y-axis
        plt.ylabel('Fixation Probability')

        plt.plot(active_node_list,fixation_list,label = str(fitness))

        # Title
        plt.title('Fixation Probability with {}'.format(setup.name))
        plt.legend()
    plt.savefig(path + ".png")
    plt.show()

    after_time = time.time()
    print("Plotting took", after_time-before_time,"Seconds")


if __name__ == '__main__':
    graph_size = 50
    fitness = 5
    # fitneses = [0.1, 0.2, 0.5, 1, 1.5,10,100]
    fitneses = [100]


    # G = Graphs.create_circle_graph(3)
    # Graphs.initialize_nodes_as_resident(G,1)
    # G = initialize_active_nodes(G,Active_Node_Setup(1),1)
    # Graphs.draw_graph(G)
    # markov = create_circle_markov_chain(G,fitness)
    # Graphs.draw_markov_model(markov)
    # #

    # compare_active_node_choosing_strategies(graph_size,fitness)
    # setup = Active_Node_Setup(1)
    # fixation_prob_active_nodes(graph_size,setup,fitneses)
    setup = Active_Node_Setup(2)
    fixation_prob_active_nodes(graph_size,setup,fitneses)


