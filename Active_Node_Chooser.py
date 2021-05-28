import copy
import re
from abc import ABC, abstractmethod
from itertools import repeat
from queue import PriorityQueue
from operator import itemgetter

from networkx import betweenness_centrality

import Graphs
from Moran_Process import *

class Strategy(ABC):
    """
    The Strategy interface that declare operations common to all supported versions of some algorithm.
    """
    @abstractmethod
    def choosing_algorithm(self, k_nodes, fitness, graph):
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
        graph = G.copy()
        nodes = []

        for i in range(k_nodes):
            print("Greedy chooser ", i, " out of ", k_nodes)
            if k_nodes != 0:
                non_active_nodes = [x for x in graph.nodes() if graph.nodes[x]['active'] == False]
                #print("Non active Nodes", non_active_nodes)

                old_graph = graph.copy()
                active_probability_list = []
                simulated_fixation_prob = 0
                for j in non_active_nodes:
                    #Set a node as active and compute the fixation probability
                    graph.nodes[j]['active'] = True
                    # numeric_fixation_prob = numeric_fixation_probability(graph, fitness)
                    fixation_list, simulated_fixation_prob = simulate(10000,graph,fitness,lowest_acceptable_fitness=simulated_fixation_prob)
                    #fixation_list, simulated_fixation_prob = simulate(10000, graph,fitness)
                    #numeric_fixation_prob=simulated_fixation_prob

                    active_probability_list.append(simulated_fixation_prob)
                    graph = old_graph.copy()
                #Round the probabilities
                active_probability_list = [round(x,5) for x in active_probability_list]
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

class Vertex_Cover(Strategy):

    def choosing_algorithm(self, k_nodes, fitness, graph):
        def f(S):
            sum = 0
            for edge in graph.edges:
                if (edge[0] in S) or (edge[1] in S):
                    sum += 1
            return sum

        nodes = []
        possible_nodes = list(range(len(G.nodes)))
        for i in range(k_nodes):
            max = 0
            maximizing_node = None
            for node in possible_nodes:
                nodes2 = nodes + [node]
                value = f(nodes2)
                if value > max:
                    max = value
                    maximizing_node = node
            nodes.append(maximizing_node)
            possible_nodes.remove(maximizing_node)
        return nodes
class Greedy_Numeric(Strategy):
    """
    Greedily chooses k nodes to become active based upon the numeric fixation probabilities/simulation fixation probability of choosing that node
    """
    def choosing_algorithm(self,k_nodes, fitness, G):
        graph = G.copy()
        nodes = []

        for i in range(k_nodes):
            # print("Greedy chooser ", i, " out of ", k_nodes)
            if k_nodes != 0:
                non_active_nodes = [x for x in graph.nodes() if graph.nodes[x]['active'] == False]
                #print("Non active Nodes", non_active_nodes)

                old_graph = graph.copy()
                active_probability_list = []
                simulated_fixation_prob = 0
                for j in non_active_nodes:
                    #Set a node as active and compute the fixation probability
                    graph.nodes[j]['active'] = True
                    numeric_fixation_prob = numeric_fixation_probability(graph, fitness)
                    simulated_fixation_prob=numeric_fixation_prob

                    active_probability_list.append(simulated_fixation_prob)
                    graph = old_graph.copy()
                #Round the probabilities
                active_probability_list = [round(x,5) for x in active_probability_list]
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


class Lazy_Greedy(Strategy):
    """
    Greedily chooses k nodes to become active based upon the numeric fixation probabilities/simulation fixation probability of choosing that node
    """
    def choosing_algorithm(self,k_nodes, fitness, G):
        graph = G.copy()
        number_of_nodes = len(graph.nodes)
        nodes = []
        baseline_probability = 1/number_of_nodes
        baseline = round(baseline_probability, 5)
        marginal_gain = {}
        for i in range(number_of_nodes):
            graph.nodes[i]['active'] = True
            fixation_list, fixation_prob = simulate(50000,graph,fitness)
            marginal_gain[i] = fixation_prob-baseline
            graph.nodes[i]['active']= False
        print("marginal_gain",marginal_gain)
        node_max = max(marginal_gain.items(), key=itemgetter(1))[0]
        print("initial node max", node_max)
        graph.nodes[node_max]['active'] = True
        nodes.append(node_max)
        fixation_prob_s = marginal_gain[node_max] + baseline
        marginal_gain.pop(node_max)
        print("marginal_gain2",marginal_gain)
        for i in range(1,k_nodes):
            print("nodes", nodes)
            print("Lazy greedy ", i+1 , " out of ", k_nodes)
            node_max = max(marginal_gain.items(), key=itemgetter(1))[0]
            print("new node max", node_max)
            graph.nodes[node_max]['active'] = True
            fixation_list, fixation_prob_s_union = simulate(50000,graph,fitness)
            graph.nodes[node_max]['active'] = False
            print("fixation_prob_s",fixation_prob_s)
            print("fixation_prob_s_union", fixation_prob_s_union)
            marginal_gain[node_max] = fixation_prob_s_union - fixation_prob_s
            print("updated marginal_gain",marginal_gain)
            for key, value in marginal_gain.items():
                if key != node_max:
                    print("reconsider?", key, value)
                    if value > marginal_gain[node_max]:
                        graph.nodes[key]['active'] = True
                        fixation_list, fixation_prob_s_union = simulate(50000,graph,fitness)
                        graph.nodes[key]['active'] = False
                        marginal_gain[key] = fixation_prob_s_union - fixation_prob_s
                        print("updated marginal gain 2", marginal_gain)
                        if marginal_gain[key] > marginal_gain[node_max]:
                            node_max = key
                            print("new node max", node_max)
            print("choice of max node", node_max)
            print("marginal gains when chosen", marginal_gain)
            graph.nodes[node_max]['active'] = True
            nodes.append(node_max)
            fixation_prob_s = marginal_gain[node_max] + fixation_prob_s
            marginal_gain.pop(node_max)


        return nodes

class Lazy_Greedy_Numeric(Strategy):
    """
    Greedily chooses k nodes to become active based upon the numeric fixation probabilities/simulation fixation probability of choosing that node
    """
    def choosing_algorithm(self,k_nodes, fitness, G):
        graph = G.copy()
        number_of_nodes = len(graph.nodes)
        nodes = []
        baseline_probability = 1/number_of_nodes
        baseline = round(baseline_probability, 5)
        marginal_gain = {}
        for i in range(number_of_nodes):
            graph.nodes[i]['active'] = True
            fixation_prob = numeric_fixation_probability(graph,fitness)
            marginal_gain[i] = fixation_prob-baseline
            graph.nodes[i]['active']= False
        #print("marginal_gain",marginal_gain)
        node_max = max(marginal_gain.items(), key=itemgetter(1))[0]
        #print("initial node max", node_max)
        graph.nodes[node_max]['active'] = True
        nodes.append(node_max)
        fixation_prob_s = marginal_gain[node_max] + baseline
        marginal_gain.pop(node_max)
        #print("marginal_gain2",marginal_gain)
        for i in range(1,k_nodes):
            #print("nodes", nodes)
            #print("Lazy greedy ", i+1 , " out of ", k_nodes)
            node_max = max(marginal_gain.items(), key=itemgetter(1))[0]
            #print("new node max", node_max)
            graph.nodes[node_max]['active'] = True
            fixation_prob_s_union = numeric_fixation_probability(graph,fitness)
            graph.nodes[node_max]['active'] = False
            #print("fixation_prob_s",fixation_prob_s)
            #print("fixation_prob_s_union", fixation_prob_s_union)
            marginal_gain[node_max] = fixation_prob_s_union - fixation_prob_s
            #print("updated marginal_gain",marginal_gain)
            for key, value in marginal_gain.items():
                if key != node_max:
                    #print("reconsider?", key, value)
                    if value > marginal_gain[node_max]:
                        graph.nodes[key]['active'] = True
                        fixation_prob_s_union = numeric_fixation_probability(graph,fitness)
                        graph.nodes[key]['active'] = False
                        marginal_gain[key] = fixation_prob_s_union - fixation_prob_s
                        #print("updated marginal gain 2", marginal_gain)
                        if marginal_gain[key] > marginal_gain[node_max]:
                            node_max = key
                            #print("new node max", node_max)
            #print("choice of max node", node_max)
            #print("marginal gains when chosen", marginal_gain)
            graph.nodes[node_max]['active'] = True
            nodes.append(node_max)
            fixation_prob_s = marginal_gain[node_max] + fixation_prob_s
            marginal_gain.pop(node_max)


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
        # if len(G.nodes) > 10:
        #     print("Do you wanna fucking die?")
        #     print("Smaller graph please....")
        #     return 0
        number_of_nodes = len(G.nodes)
        node_list = range(0, number_of_nodes)
        # all_pairs = []
        # for i in range(1, number_of_nodes + 1):
        #     all_pairs.append(list(itertools.combinations(node_list, i)))
        # markov_model_graph = Graphs.create_markov_model(G, all_pairs, fitness)
        # #Graphs.draw_markov_model(markov_model_graph)
        #
        # #Get all nodes that contains k or smaller amount of nodes to become active
        # all_combinations_of_nodes_of_k_size = []
        # for i in markov_model_graph.nodes():
        #     int_of_node = re.sub("[^0-9]", "", i)
        #     tuple_of_node = tuple(int_of_node)
        #     #Asger says we have to pick k nodes :(
        #     if len(tuple_of_node) == k_nodes:
        #         all_combinations_of_nodes_of_k_size.append(tuple_of_node)
        all_combinations_of_nodes_of_k_size = list(itertools.combinations(node_list, k_nodes))
        # Compute the numerical fixation probability for each of the possible choices of active nodes
        optimal_fixations_probabilities = []
        for i in all_combinations_of_nodes_of_k_size:
            graph = G.copy()
            for j in i:
                graph.nodes[int(j)]['active'] = True
            numeric_fixation_prob = numeric_fixation_probability(graph, fitness)
            optimal_fixations_probabilities.append(numeric_fixation_prob)

        #Round the probabilities
        optimal_fixations_probabilities = [round(x,5) for x in optimal_fixations_probabilities]

        #print("Combi ", all_combinations_of_nodes_of_k_size)
        #print("Probs ", optimal_fixations_probabilities)



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

class DualPriorityQueue(PriorityQueue):
    def __init__(self, maxPQ=False):
        PriorityQueue.__init__(self)
        self.reverse = -1 if maxPQ else 1

    def put(self, priority, data):
        PriorityQueue.put(self, (self.reverse * priority, data))

    def get(self, *args, **kwargs):
        priority, data = PriorityQueue.get(self, *args, **kwargs)
        return self.reverse * priority, data

    def get_first(self,*args, **kwargs):
        priority, data = PriorityQueue.get(self, *args, **kwargs)
        DualPriorityQueue.put(self, self.reverse*priority, data)
        return self.reverse * priority, data

    def empty(self):
        return PriorityQueue.empty(self)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            if not self.empty():
                return self.get()  # block=True | default
            else:
                raise StopIteration
        except ValueError:  # the Queue is closed
            raise StopIteration

if __name__ == '__main__':
    fitness = 2
    multiplier = 1
    # graph_size = 3
    # eps = 0.0015
    #
    # #G = Graphs.create_complete_graph(graph_size)
    # G = Graphs.create_star_graph(graph_size)
    # G = Graphs.create_karate_club_graph()
    G = nx.erdos_renyi_graph(100, 0.05, directed=True)
    # all_graphs_of_size_n = get_all_graphs_of_size_n("6c")
    # #35
    # G = all_graphs_of_size_n[42]
    #
    #
    #
    Graphs.initialize_nodes_as_resident(G,multiplier)
    Graphs.draw_graph(G)
    # print(len(G.nodes))
    # fixation_list, baseline_probability = simulate(100000,G,fitness)
    # print(baseline_probability)

    # lazy_greedy_chooser = Active_Node_Chooser(3, G, fitness, Lazy_Greedy())
    # lazy_greedy_nodes = lazy_greedy_chooser.choose_nodes()
    # print("Lazy Greedy nodes to activate list", lazy_greedy_nodes, "\n")
    # print("The graph after node choosen", G.nodes(data = True))

    # greedy_chooser = Active_Node_Chooser(3,G,fitness,Greedy())
    # greedy_nodes = greedy_chooser.choose_nodes()
    # print("Greedy nodes to activate list", greedy_nodes, "\n")
    # print("The graph after node choosen", G.nodes(data = True))


    vertex_cover_chooser = Active_Node_Chooser(50, G, fitness, Vertex_Cover())
    vertex_cover_nodes = vertex_cover_chooser.choose_nodes()
    print("Vertex cover nodes to activate list", vertex_cover_nodes, "\n")

    """    
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
    n = 3000
    fixation_list, simulated_fixation_prob = simulate(n, G,fitness)
    iteration_list = list(range(0, n))
    plot_fixation_iteration(iteration_list, fixation_list, numeric_fixation_prob)
    print("Simulated fixation probability = ", simulated_fixation_prob)
    print("Numeric fixation probability = ", numeric_fixation_prob)
    print("Difference = ", abs(simulated_fixation_prob - numeric_fixation_prob))
    """