import math

from pandas import np

import Graphs
import networkx as nx
import matplotlib.pyplot as plt

from Moran_Process import numeric_fixation_probability, simulate, plot_fixation_iteration


def calculate_markov_weights(i, j, G, fitness, active_nodes):
    pass


def update_selfloop(node, markov):
    pass



def add_weights_to_edges(markov, G, fitness,active_nodes):
    pass

def create_star_markov_chain(G, active_leaves,fitness):
    N = len(G.nodes)
    extinction_node = '0,0,0'
    markov = nx.DiGraph()
    markov.add_node(extinction_node)
    markov.add_edge(extinction_node,extinction_node)
    markov[extinction_node][extinction_node]['weight'] = 1
    previous_nodes = [extinction_node]
    for i in range(1,N+1):
        node_list = create_integer_partitions(i,active_leaves)
        for node in node_list:
            markov.add_node(node)
            Graphs.draw_markov_model(markov)
            values = node.split(',')
            x = int(values[0])
            y = int(values[1])
            z = int(values[2])
            for prev_node in previous_nodes:
                values = prev_node.split(',')
                prev_x = int(values[0])
                prev_y = int(values[1])
                prev_z = int(values[2])
                print("node",x,y,z)
                print("prev_node",prev_x,prev_y,prev_z)
                #Backwards edges
                if x > prev_x and z == 0:
                    # Loose mutant in active leaf
                    markov.add_edge(node,prev_node)
                elif y > prev_y and z ==0 :
                    # Loose mutant in passive leaf
                    markov.add_edge(node,prev_node)
                elif z > prev_z and (x+y < N):
                    # Loose mutant in center
                    markov.add_edge(node,prev_node)
                elif x < prev_x and z == 1:
                    # Gain mutant in active leaf
                    markov.add_edge(node,prev_node)
                elif y < prev_y and z == 1:
                    # Gain mutant in passive leaf
                    markov.add_edge(node,prev_node)
                elif z < prev_z and (x+y > 0):
                    # Gain mutant in center
                    markov.add_edge(node,prev_node)




        previous_nodes = node_list

    Graphs.draw_markov_model(markov)
    return markov

def create_integer_partitions(number,active_leaves,N):
    answer = []
    for z in range(2):
        number = number - z
        for x in range(number+1):
            y = number-x
            if x <= active_leaves and y < N-active_leaves:


                node_string = str(x) + "," + str(y) + "," + str(z)
                answer.append(node_string)
    return answer


# Give nodes name corresponding to their variable in the linear system
def rename_nodes(markov):
    pass

def compute_fixation_probability_star(markov, G, active_nodes):
    pass

def star_experiment(G,fitness):
    pass



if __name__ == '__main__':
    multiplier = 1
    graph_size = 6
    active_leaves = 3
    fitness = 2

    # G = Graphs.create_star_graph(graph_size)
    # Graphs.initialize_nodes_as_resident(G,multiplier)
    # markov = create_star_markov_chain(G,active_leaves,fitness)
    # Graphs.draw_graph(G)
    for i in range(1,7):

        answer = create_integer_partitions(i,3,7)
        print(answer)
        print(len(answer))


