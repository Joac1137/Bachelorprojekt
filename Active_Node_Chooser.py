from abc import ABC, abstractmethod

from networkx import betweenness_centrality

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
    Greedily chooses k nodes to become active based upon the numeric fixation probabilities/simulation fixation probability of choosing that node
    """
    def choosing_algorithm(self,k_nodes, fitness, G):
        #Might have to do some sort of rounding
        graph = G.copy()
        nodes = []
        for i in range(k_nodes):
            non_active_nodes = [x for x in graph.nodes() if graph.nodes[x]['active'] == False]
            #print("Non active Nodes", non_active_nodes)

            old_graph = graph.copy()
            active_probability_list = []
            for j in non_active_nodes:
                #Set a node as active and compute the fixation probability
                graph.nodes[j]['active'] = True
                numeric_fixation_prob = numeric_fixation_probability(graph, fitness)
                active_probability_list.append(numeric_fixation_prob)
                graph = old_graph.copy()

            #Round the probabilities
            active_probability_list = [round(x,10) for x in active_probability_list]

            #Get the index of the largest value
            max_index = active_probability_list.index(max(active_probability_list))

            #Find the non active node that corresponded to this largest value
            node_to_make_active = non_active_nodes[max_index]
            #print("What node do we pick then? ", node_to_make_active)
            print(active_probability_list)

            #Make the choosen node active
            graph.nodes[node_to_make_active]['active'] = True
            nodes.append(node_to_make_active)
            #print(graph.nodes(data=True))
            print("In round ", i+1, " we choose node ", node_to_make_active, " to become active")

        return nodes


class High_node_degree(Strategy):
    """
    Chooses k nodes to become active based upon the degree of the nodes. We prefer high node degree
    """
    def choosing_algorithm(self,k_nodes, fitness, G):
        #Might have to do some sort of rounding
        graph = G.copy()
        nodes = []
        for i in range(k_nodes):
            #We only want to choose nodes that are not already active
            non_active_nodes = [x for x in graph.nodes() if graph.nodes[x]['active'] == False]
            #print("Non active Nodes", non_active_nodes)

            old_graph = graph.copy()
            degree_list = []
            for j in non_active_nodes:
                #Find degree of nodes
                centrality_temp = nx.degree_centrality(graph)
                #Get the values of the dict if the key is in non_active_nodes
                centrality = [value for key,value in centrality_temp.items() if key in non_active_nodes]
                degree_list = centrality
                graph = old_graph.copy()

            #Round the probabilities
            degree_list = [round(x,10) for x in degree_list]

            #Get the index of the largest value
            max_index = degree_list.index(max(degree_list))

            #Find the non active node that corresponded to this largest value
            node_to_make_active = non_active_nodes[max_index]
            #print("What node do we pick then? ", node_to_make_active)
            print(degree_list)

            #Make the choosen node active
            graph.nodes[node_to_make_active]['active'] = True
            nodes.append(node_to_make_active)
            #print(graph.nodes(data=True))
            print("In round ", i+1, " we choose node ", node_to_make_active, " to become active")

        return nodes


class Low_node_degree(Strategy):
    """
    Chooses k nodes to become active based upon the degree of the nodes. We prefer low node degree
    """
    def choosing_algorithm(self,k_nodes, fitness, G):
        #Might have to do some sort of rounding
        graph = G.copy()
        nodes = []
        for i in range(k_nodes):
            #We only want to choose nodes that are not already active
            non_active_nodes = [x for x in graph.nodes() if graph.nodes[x]['active'] == False]
            #print("Non active Nodes", non_active_nodes)

            old_graph = graph.copy()
            degree_list = []
            for j in non_active_nodes:
                #Find degree of nodes
                centrality_temp = nx.degree_centrality(graph)
                #Get the values of the dict if the key is in non_active_nodes
                centrality = [value for key,value in centrality_temp.items() if key in non_active_nodes]
                degree_list = centrality
                graph = old_graph.copy()

            #Round the probabilities
            degree_list = [round(x,10) for x in degree_list]

            #Get the index of the largest value
            max_index = degree_list.index(min(degree_list))

            #Find the non active node that corresponded to this largest value
            node_to_make_active = non_active_nodes[max_index]
            #print("What node do we pick then? ", node_to_make_active)
            print(degree_list)

            #Make the choosen node active
            graph.nodes[node_to_make_active]['active'] = True
            nodes.append(node_to_make_active)
            #print(graph.nodes(data=True))
            print("In round ", i+1, " we choose node ", node_to_make_active, " to become active")

        return nodes


class Random(Strategy):
    """
    Chooses k nodes to become active randomly
    """
    def choosing_algorithm(self,k_nodes, fitness, G):
        graph = G.copy()
        nodes = []
        for i in range(k_nodes):
            #We only want to choose nodes that are not already active
            non_active_nodes = [x for x in graph.nodes() if graph.nodes[x]['active'] == False]
            node_to_make_active = random.choice(non_active_nodes)
            #Make the choosen node active
            graph.nodes[node_to_make_active]['active'] = True
            nodes.append(node_to_make_active)
        return nodes

class Centrality(Strategy):
    """
    Chooses k nodes to become active based upon the centrality of the nodes. We prefer high node centrality
    """
    def choosing_algorithm(self,k_nodes, fitness, G):
        #Might have to do some sort of rounding
        graph = G.copy()
        nodes = []
        for i in range(k_nodes):
            #We only want to choose nodes that are not already active
            non_active_nodes = [x for x in graph.nodes() if graph.nodes[x]['active'] == False]

            old_graph = graph.copy()
            for j in non_active_nodes:
                #Find degree of nodes
                centrality_temp = betweenness_centrality(G)

                #Get the values of the dict if the key is in non_active_nodes
                centrality = [value for key,value in centrality_temp.items() if key in non_active_nodes]

                graph = old_graph.copy()

            #Round the probabilities
            centrality = [round(x,10) for x in centrality]

            #Get the index of the largest value
            max_index = centrality.index(max(centrality))

            #Find the non active node that corresponded to this largest value
            node_to_make_active = non_active_nodes[max_index]
            #print("What node do we pick then? ", node_to_make_active)
            print(centrality)

            #Make the choosen node active
            graph.nodes[node_to_make_active]['active'] = True
            nodes.append(node_to_make_active)
            #print(graph.nodes(data=True))
            print("In round ", i+1, " we choose node ", node_to_make_active, " to become active")

        return nodes

class Optimal(Strategy):
    """

    """
    def choosing_algorithm(self,k_nodes, fitness, G):
        pass


class Temperature(Strategy):
    """
    Chooses k nodes to become active based upon the temperature of the nodes. We prefer high node temperature
    """
    def choosing_algorithm(self,k_nodes, fitness, G):
        #Might have to do some sort of rounding
        graph = G.copy()
        nodes = []
        for i in range(k_nodes):
            #We only want to choose nodes that are not already active
            non_active_nodes = [x for x in graph.nodes() if graph.nodes[x]['active'] == False]

            old_graph = graph.copy()
            for j in non_active_nodes:

                temp_list = np.zeros(len(G.nodes()))
                for node1, node2, data in G.edges(data=True):

                    temp_list[node2] += list(data.values())[0]

                #Get the values of the dict if the key is in non_active_nodes
                index = np.where(1 == temp_list)
                temperature = [x for x in temp_list if np.where(x == temp_list)[0][0] in non_active_nodes]

                graph = old_graph.copy()

            #Round the probabilities
            temperature = [round(x,10) for x in temperature]

            #Get the index of the largest value
            max_index = temperature.index(max(temperature))

            #Find the non active node that corresponded to this largest value
            node_to_make_active = non_active_nodes[max_index]
            #print("What node do we pick then? ", node_to_make_active)
            print(temperature)

            #Make the choosen node active
            graph.nodes[node_to_make_active]['active'] = True
            nodes.append(node_to_make_active)
            #print(graph.nodes(data=True))
            print("In round ", i+1, " we choose node ", node_to_make_active, " to become active")

        return nodes


if __name__ == '__main__':
    fitness = 0.1
    multiplier = 1
    graph_size = 3
    eps = 0.0015

    #G = Graphs.create_complete_graph(graph_size)
    #G = Graphs.create_star_graph(graph_size)

    all_graphs_of_size_n = get_all_graphs_of_size_n("6c")
    G = all_graphs_of_size_n[35]



    Graphs.initialize_nodes_as_resident(G,multiplier)
    Graphs.draw_graph(G)

    greedy_chooser = Active_Node_Chooser(2,G,fitness,Greedy())
    greedy_nodes = greedy_chooser.choose_nodes()
    print("Greedy nodes to activate list", greedy_nodes, "\n")


    high_degree_chooser = Active_Node_Chooser(2,G,fitness,High_node_degree())
    high_degree_nodes = high_degree_chooser.choose_nodes()
    print("High Degree nodes to activate list", high_degree_nodes, "\n")

    low_degree_chooser = Active_Node_Chooser(2,G,fitness,Low_node_degree())
    low_degree_nodes = low_degree_chooser.choose_nodes()
    print("Low Degree nodes to activate list", low_degree_nodes, "\n")

    random_chooser = Active_Node_Chooser(2,G,fitness,Random())
    random_nodes = random_chooser.choose_nodes()
    print("Random nodes to activate list", random_nodes, "\n")

    centrality_chooser = Active_Node_Chooser(2,G,fitness,Centrality())
    centrality_nodes = centrality_chooser.choose_nodes()
    print("Centrality nodes to activate list", centrality_nodes, "\n")

    temperature_chooser = Active_Node_Chooser(2,G,fitness,Temperature())
    temperature_nodes = temperature_chooser.choose_nodes()
    print("Temperature nodes to activate list", centrality_nodes, "\n")

    print("\nHere is the graph we get back",G.nodes(data=True))


    print("\n")

    for i in high_degree_nodes:
        G.nodes[i]['active'] = True

    numeric_fixation_prob = numeric_fixation_probability(G, fitness)

    iteration_list, fixation_list, simulated_fixation_prob = simulate(3000, G,fitness,numeric_fixation_prob,eps)

    plot_fixation_iteration(iteration_list, fixation_list, numeric_fixation_prob)
    print("Simulated fixation probability = ", simulated_fixation_prob)
    print("Numeric fixation probability = ", numeric_fixation_prob)
    print("Difference = ", abs(simulated_fixation_prob - numeric_fixation_prob))

    """
        TODO:
            - Optimal
            - Clean might be able to delete a for loop
    """