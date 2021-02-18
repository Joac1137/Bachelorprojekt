from random import uniform, random, sample, randint
import Graphs


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
def step(G):  # Idk the arg might be wrong
    # Arg 1 -> Might be something different than the graph
    # Arg 2 -> Node that has been chosen for mutation based upon fitness

    # Get all neighboring nodes and walk on a edge based upond the weights

    # Choose a node based on fitness (for now it's just uniformly)
    replicating_node_index = randint(0, len(G.nodes()) - 1)

    # Find all node neighbors
    neighbors = [x for x in G.neighbors(replicating_node_index)]

    # Mutate a neighbor
    choice = randint(0, len(neighbors) - 1)
    node_to_mutate = neighbors[choice]
    # print("replicating",G.nodes[replicating_node_index]['type'].__class__)
    # print("dying",G.nodes[node_to_mutate]['type'].__class__)
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


if __name__ == "__main__":
    n = 10000
    fixationCounter = 0
    for i in range(0, n):
        G = Graphs.createCompleteGraph()
        mutate_a_random_node(G)
        # Does a Moran Step whenever we do not have the same color in the graph
        while not have_we_terminated(G):
            step(G)
        fixationCounter += is_the_first_node_mutant(G)
    print(fixationCounter/n)
