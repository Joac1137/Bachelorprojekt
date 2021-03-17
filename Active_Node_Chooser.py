from abc import ABC, abstractmethod

import Graphs
from Moran_Process import *

class Strategy(ABC):
    """
    The Strategy interface that declare operations common to all supported versions of some algorithm.
    """
    @abstractmethod
    def choosing_algorithm(self, k_nodes, graph):
        pass

class Active_Node_Chooser():
    """
    The Active_Node_Chooser defines a class in which you can give k nodes, a graph and a strategy and it finds the active nodes
    """

    def __init__(self, k_nodes, graph,fitness, strategy: Strategy):
        """
        :param k_nodes: Number of nodes to consider
        :param graph: The graph to consider
        :param strategy: The choosing algorithm to consider
        """
        self._k_nodes = k_nodes
        self._graph = graph
        self._fitness = fitness
        self._strategy = strategy


    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy):
        self._strategy = strategy


    def choose_nodes(self):
        """
        Delegates the logical work to the concrete Strategy object
        """
        nodes = self._strategy.choosing_algorithm(self._k_nodes,self._fitness, self._graph)
        return nodes


class Greedy(Strategy):
    """
    Greedily chooses k nodes to become active based upon the numeric fixation probabilities of choosing that node
    """
    def choosing_algorithm(self,k_nodes, fitness, G):
        #Might have to do some sort of rounding
        #Might need to take fitness as parameter
        graph = G.copy()
        nodes = []
        for i in range(k_nodes):
            #The below is just a test that it gives the correct node if the multipliers aren't the same
            #graph.nodes[i]['multiplier'] = 20

            print(graph.nodes(data=True))

            #We only want to choose nodes that are not already active
            non_active_nodes = [x for x in graph.nodes() if graph.nodes[x]['active'] == False]
            print("Non active Nodes", non_active_nodes)

            old_graph = graph.copy()
            active_probability_list = []
            for j in non_active_nodes:
                #Set a node as active and compute the fixation probability
                graph.nodes[j]['active'] = True
                numeric_fixation_prob = numeric_fixation_probability(graph, fitness)
                active_probability_list.append(numeric_fixation_prob)
                graph = old_graph.copy()

            #Get the index of the largest value
            max_index = active_probability_list.index(max(active_probability_list))

            #Find the non active node that corresponded to this largest value
            node_to_make_active = non_active_nodes[max_index]
            print("What node do we pick then? ", node_to_make_active)
            print(active_probability_list)

            #Make the choosen node active
            graph.nodes[node_to_make_active]['active'] = True
            nodes.append(node_to_make_active)
            print(graph.nodes(data=True))
            print("In round ", i+1, " we choose node ", node_to_make_active, " to become active")

        return nodes


class High_node_degree(Strategy):
    """

    """
    def choosing_algorithm(self,k_nodes, fitness, G):
        pass


class Low_node_degree(Strategy):
    """

    """
    def choosing_algorithm(self,k_nodes, fitness, G):
        pass


class High_node_temperature(Strategy):
    """

    """
    def choosing_algorithm(self,k_nodes, fitness, G):
        pass


class Low_node_degree_temperature(Strategy):
    """

    """
    def choosing_algorithm(self,k_nodes, fitness, G):
        pass

class Selfloop(Strategy):
    """

    """
    def choosing_algorithm(self,k_nodes, fitness, G):
        pass


class Centrality(Strategy):
    """

    """
    def choosing_algorithm(self,k_nodes, fitness, G):
        pass

if __name__ == '__main__':
    fitness = 0.1
    multiplier = 1
    graph_size = 3
    eps = 0.0015

    #G = Graphs.create_complete_graph(graph_size)
    G = Graphs.create_star_graph(graph_size)
    Graphs.initialize_nodes_as_resident(G,multiplier)
    Graphs.draw_graph(G)

    chooser = Active_Node_Chooser(2,G,fitness,Greedy())
    nodes = chooser.choose_nodes()

    print("Here is the graph we get back",G.nodes(data=True))
    print("Nodes to activate list", nodes)

    print("\n")

    for i in nodes:
        G.nodes[i]['active'] = True

    numeric_fixation_prob = numeric_fixation_probability(G, fitness)

    iteration_list, fixation_list, simulated_fixation_prob = simulate(3000, G,fitness,numeric_fixation_prob,eps)

    plot_fixation_iteration(iteration_list, fixation_list, numeric_fixation_prob)
    print("Simulated fixation probability = ", simulated_fixation_prob)
    print("Numeric fixation probability = ", numeric_fixation_prob)
    print("Difference = ", abs(simulated_fixation_prob - numeric_fixation_prob))