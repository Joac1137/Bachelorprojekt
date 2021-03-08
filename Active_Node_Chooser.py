from abc import ABC, abstractmethod

import Graphs
from Moran_Process import *

class Strategy(ABC):
    @abstractmethod
    def choosing_algorithm(self, k_nodes, graph):
        pass

class Active_Node_Chooser():

    def __init__(self, k_nodes, graph, strategy: Strategy):
        self._k_nodes = k_nodes
        self._graph = graph
        self._strategy = strategy


    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy):
        self._strategy = strategy


    def choose_nodes(self):
        self._strategy.choosing_algorithm(self._k_nodes,self._graph)

class Greedy(Strategy):
    def choosing_algorithm(self,k_nodes, graph):
        print("Nodes", k_nodes)
        Graphs.draw_graph(graph)

if __name__ == '__main__':
    fitness = 1
    graph_size = 3
    eps = 0.0015

    G = Graphs.create_complete_graph(graph_size)
    Graphs.initialize_nodes_as_resident(G)
    Graphs.draw_graph(G)

    chooser = Active_Node_Chooser(1,G,Greedy())
    chooser.choose_nodes()
    print("\n")


    numeric_fixation_prob = numeric_fixation_probability(G, 1)

    iteration_list, fixation_list, simulated_fixation_prob = simulate(3000, G,fitness,numeric_fixation_prob,eps)

    plot_fixation_iteration(iteration_list, fixation_list, numeric_fixation_prob)
    print("Simulated fixation probability = ", simulated_fixation_prob)
    print("Numeric fixation probability = ", numeric_fixation_prob)
    print("Difference = ", abs(simulated_fixation_prob - numeric_fixation_prob))