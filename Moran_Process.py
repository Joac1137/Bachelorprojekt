from random import uniform, random, sample, randint, choices
import itertools
import Graphs
import matplotlib.pyplot as plt
import random

class Mutant:
    def __init__(self, fitness, id_n='mutant', color='red'):
        self.fitness = fitness
        self.color = color
        self.id_n = id_n

    def __hash__(self):
        return self.id_n

    def __cmp__(self, p):
        if self.id_n < p.id_n:
            return -1
        elif self.id_n == p.id_n:
            return 0
        else:
            return 1

    def __eq__(self, p):
        if p.id_n == self.id_n:
            return True
        else:
            return False

    def __repr__(self):
        return str(self.id_n)


class Resident:
    def __init__(self, fitness, id_n='resident', color='blue'):
        self.id_n = id_n
        self.fitness = fitness
        self.color = color

    def __hash__(self):
        return self.id_n

    def __cmp__(self, p):
        if self.id_n < p.id_n:
            return -1
        elif self.id_n == p.id_n:
            return 0
        else:
            return 1

    def __eq__(self, p):
        if p.id_n == self.id_n:
            return True
        else:
            return False

    def __repr__(self):
        return str(self.id_n)


# Mutate neighbor
def step(G):
    # Get all neighboring nodes and walk on a edge based upon the weights

    # Choose a node based on fitness and the multiplier
    fitness_distribution = list()
    for i in G.nodes():
        # Fitness
        fitness = G.nodes[i]['type'].fitness

        # Multiplier for node
        multiplier = G.nodes[i]['multiplier']

        # Only Mutants should benefit of the multiplier
        if G.nodes[i]['type'].id_n == 'resident':
            multiplier = 1

        fitness_distribution.append(multiplier*fitness)

    #Nodes as a list
    nodes = range(0,len(G.nodes()))

    replicating_node_index = random.choices(nodes,weights = fitness_distribution,k=1)[0]


    # Mutate a neighbor based on the weights of the edges
    # Find all node neighbors
    neighbors = G.edges(replicating_node_index)

    # Get the corresponding weights
    edge_weights = [G.get_edge_data(x,y)['weight'] for x,y in neighbors]
    neighbor_nodes = [y for x,y in neighbors]

    # Choose one edge to walk on
    node_to_mutate = random.choices(neighbor_nodes,weights=edge_weights,k=1)[0]

    G.nodes[node_to_mutate]['type'] = G.nodes[replicating_node_index]['type']
    # Graphs.drawGraph(G)


# Uniformly picks a node to initially mutate
def mutate_a_random_node(G):
    # Generate 'random' node to mutate
    node = randint(0, len(G.nodes()) - 1)
    node_type = Mutant(1)
    G.nodes[node]['type'] = node_type
    # Graphs.drawGraph(G)


# Checks whether or not we have the same color in all nodes of the graph
def have_we_terminated(G):
    first_type = G.nodes[0]['type']
    for i in G.nodes():
        if G.nodes[i]['type'] != first_type:
            return False
    return True


def is_the_first_node_mutant(G):
    if G.nodes[0]['type'].id_n == 'mutant':
        return 1
    return 0

#Plotting iterations and fixation fraction
def plot_fixation_iteration(x,y):
    plt.plot(x,y)

    #Plot expected value for well-mixed graph (0.2) - might need to change based on numeric solution
    plt.axhline(y=0.2,color='r',linestyle='-',label = 'Expected Probability')

    #Name x-axis
    plt.xlabel('Iterations')

    #Name y-axis
    plt.ylabel('Fixation/Iterations')

    #Title
    plt.title('Fixation Fraction as a function of Iterations')
    plt.legend(loc=1, prop={'size': 6})
    plt.show()

# Computes the numerical fixation probability
def numeric_fixation_probability(G):
    if len(G.nodes) > 10:
        print("Do you wanna fucking die?")
        print("Smaller graph please....")
        return 0
    number_of_nodes = len(G.nodes)
    node_list = range(0, number_of_nodes)
    all_pairs = []
    for i in range(1, number_of_nodes+1):
        all_pairs.append(list(itertools.combinations(node_list, i)))
    markov_model_graph = Graphs.create_markov_model(G,all_pairs)
    Graphs.draw_markov_model(markov_model_graph)
    return 0

def simulate(n):
    fixationCounter = 0
    fixationList = list()
    iterationList = list(range(0,n))
    for i in range(1, n+1):
        G = Graphs.createKarateClubGraph()
        mutate_a_random_node(G)
        # Does a Moran Step whenever we do not have the same color in the graph
        while not have_we_terminated(G):
            step(G)
            # Graphs.drawGraph(G)
        fixationCounter += is_the_first_node_mutant(G)
        fixationList.append(fixationCounter/i)
        # Graphs.drawGraph(G)
    #numeric_fixation_probability(G)
    plot_fixation_iteration(iterationList,fixationList)
    return fixationCounter/n

if __name__ == "__main__":
    # fixation_prob = simulate(10)
    # print("Fixation Probability",fixation_prob)
    # G = Graphs.create_star_graph()
    # G = Graphs.createCompleteGraph()
    G = Graphs.createKarateClubGraph()
    numeric_fixation_probability(G)


