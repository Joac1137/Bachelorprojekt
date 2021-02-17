from random import uniform, random, choice, sample, randint
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
    neighbors = [n for n in G.neighbors(replicating_node_index)]


    # Mutate a neighbor
    choice = randint(0, len(neighbors)-1)
    nodeToMutate = neighbors[choice]
    #print("replicating",G.nodes[replicating_node_index]['type'].__class__)
    #print("dying",G.nodes[nodeToMutate]['type'].__class__)
    G.nodes[nodeToMutate]['type'] = G.nodes[replicating_node_index]['type']
    #Graphs.drawGraph(G)


# Uniformly picks a node to initially mutate
def mutateARandomNode(G):
    # Generate 'random' node to mutate
    node = randint(0, len(G.nodes()) - 1)
    nodeType = Mutant(1)
    G.nodes[node]['type'] = nodeType
    #Graphs.drawGraph(G)

#Checks whether or not we have the same color in all nodes of the graph
def haveWeTerminated(G):
    firstType = G.nodes[0]['type']
    for i in G.nodes():
        if G.nodes[i]['type'] != firstType:
            return False
    return True

def didWeFixate(G):
    firstType = G.nodes[0]['type']
    print("type",firstType.__class__)
    if firstType.id_n == 'mutation':
        return 1
    return 0

if __name__ == "__main__":
    fixationCounter = 0
    for i in range(100):
        G = Graphs.createCompleteGraph()
        mutateARandomNode(G)
        #Does a Moran Step whenever we do not have the same color in the graph
        while(not haveWeTerminated(G)):
            step(G)
        fixationCounter += didWeFixate(G)
    print(fixationCounter/100)