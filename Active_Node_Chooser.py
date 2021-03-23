import re
from abc import ABC, abstractmethod
from itertools import repeat

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
            if k_nodes != 0:
                non_active_nodes = [x for x in graph.nodes() if graph.nodes[x]['active'] == False]
                #print("Non active Nodes", non_active_nodes)

                old_graph = graph.copy()
                active_probability_list = []
                for j in non_active_nodes:
                    #Set a node as active and compute the fixation probability
                    graph.nodes[j]['active'] = True
                    #numeric_fixation_prob = numeric_fixation_probability(graph, fitness)
                    iteration_list, fixation_list, simulated_fixation_prob = simulate(3000,graph,fitness,0,0.0015,3000)
                    numeric_fixation_prob=simulated_fixation_prob

                    active_probability_list.append(numeric_fixation_prob)
                    graph = old_graph.copy()
                #Round the probabilities
                active_probability_list = [round(x,10) for x in active_probability_list]
                #Get the index of the largest value
                max_index = active_probability_list.index(max(active_probability_list))

                #Find the non active node that corresponded to this largest value
                node_to_make_active = non_active_nodes[max_index]
                #print("What node do we pick then? ", node_to_make_active)
                #print(active_probability_list)

                #Make the choosen node active
                graph.nodes[node_to_make_active]['active'] = True
                nodes.append(node_to_make_active)
                #print(graph.nodes(data=True))
                #print("In round ", i+1 , " we choose node ", node_to_make_active, " to become active")

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
            #print(degree_list)

            #Make the choosen node active
            graph.nodes[node_to_make_active]['active'] = True
            nodes.append(node_to_make_active)
            #print(graph.nodes(data=True))
            #print("In round ", i+1, " we choose node ", node_to_make_active, " to become active")

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
            #print(degree_list)

            #Make the choosen node active
            graph.nodes[node_to_make_active]['active'] = True
            nodes.append(node_to_make_active)

            #print("In round ", i+1, " we choose node ", node_to_make_active, " to become active")

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
            #print(centrality)

            #Make the choosen node active
            graph.nodes[node_to_make_active]['active'] = True
            nodes.append(node_to_make_active)
            #print(graph.nodes(data=True))
            #print("In round ", i+1, " we choose node ", node_to_make_active, " to become active")

        return nodes

class Optimal(Strategy):
    """
    Computes the optimal choice for which k nodes to become active
    """
    def choosing_algorithm(self,k_nodes, fitness, G):
        #Create all pairs of nodes and a graph for that
        if len(G.nodes) > 10:
            print("Do you wanna fucking die?")
            print("Smaller graph please....")
            return 0
        number_of_nodes = len(G.nodes)
        node_list = range(0, number_of_nodes)
        all_pairs = []
        for i in range(1, number_of_nodes + 1):
            all_pairs.append(list(itertools.combinations(node_list, i)))
        markov_model_graph = Graphs.create_markov_model(G, all_pairs, fitness)
        #Graphs.draw_markov_model(markov_model_graph)

        #Get all nodes that contains k or smaller amount of nodes to become active
        all_combinations_of_nodes_of_k_size = []
        for i in markov_model_graph.nodes():
            int_of_node = re.sub("[^0-9]", "", i)
            tuple_of_node = tuple(int_of_node)
            #Asger says we have to pick k nodes :(
            if len(tuple_of_node) == k_nodes:
                all_combinations_of_nodes_of_k_size.append(tuple_of_node)

        # Compute the numerical fixation probability for each of the possible choices of active nodes
        optimal_fixations_probabilities = []
        for i in all_combinations_of_nodes_of_k_size:
            graph = G.copy()
            for j in i:
                graph.nodes[int(j)]['active'] = True
            numeric_fixation_prob = numeric_fixation_probability(graph, fitness)
            optimal_fixations_probabilities.append(numeric_fixation_prob)

        #Choose the active nodes to be the ones that maximizes the fixation probability
        max_index = optimal_fixations_probabilities.index(max(optimal_fixations_probabilities))
        nodes_to_become_active = all_combinations_of_nodes_of_k_size[max_index]

        list_of_nodes = [int(x) for x in nodes_to_become_active]

        return list_of_nodes

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

            temp_list = np.zeros(len(G.nodes()))
            for node1, node2, data in G.edges(data=True):
                temp_list[node1] += list(data.values())[0]
                temp_list[node2] += list(data.values())[0]



            #temperature = [x for x in temp_list if np.where(x == temp_list)[0][0] in non_active_nodes]
            temperature = [temp_list[x] for x in non_active_nodes]

            graph = old_graph.copy()

            #Round the probabilities
            temperature = [round(x,10) for x in temperature]

            #Get the index of the largest value
            max_index = temperature.index(max(temperature))

            #Find the non active node that corresponded to this largest value
            node_to_make_active = non_active_nodes[max_index]
            #print(temperature)

            #Make the choosen node active
            graph.nodes[node_to_make_active]['active'] = True
            nodes.append(node_to_make_active)
            #print("In round ", i+1, " we choose node ", node_to_make_active, " to become active")

        return nodes


if __name__ == '__main__':
    fitness = 0.1
    multiplier = 1
    graph_size = 4
    eps = 0.0015

    #G = Graphs.create_complete_graph(graph_size)
    G = Graphs.create_star_graph(graph_size)

    #all_graphs_of_size_n = get_all_graphs_of_size_n("6c")
    #G = all_graphs_of_size_n[35]



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

    centrality_chooser = Active_Node_Chooser(2,G,fitness,Centrality())
    centrality_nodes = centrality_chooser.choose_nodes()
    print("Centrality nodes to activate list", centrality_nodes, "\n")

    temperature_chooser = Active_Node_Chooser(2,G,fitness,Temperature())
    temperature_nodes = temperature_chooser.choose_nodes()
    print("Temperature nodes to activate list", temperature_nodes, "\n")

    random_chooser = Active_Node_Chooser(2,G,fitness,Random())
    random_nodes = random_chooser.choose_nodes()
    print("Random nodes to activate list", random_nodes, "\n")

    optimal_chooser = Active_Node_Chooser(2,G,fitness,Optimal())
    optimal_nodes = optimal_chooser.choose_nodes()
    print("Optimal nodes to activate list", optimal_nodes, "\n")

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
