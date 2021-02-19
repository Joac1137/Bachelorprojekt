import networkx as nx
import matplotlib.pyplot as plt
import Moran_Process as mp

def createCompleteGraph():
    G = nx.complete_graph(5)
    G = initializeNodesAsResident(G)
    #drawGraph(G)

    return G

def create_star_graph():
    G = nx.star_graph(5)
    G = initializeNodesAsResident(G)
    #drawGraph(G)
    G.nodes[0]['multiplier'] = 10
    return G

def createKarateClubGraph():
    G = nx.karate_club_graph()
    G = initializeNodesAsResident(G)
    #drawGraph(G)

    return G

def drawGraph(G):
    # The drawn colors are created from whether the node is a resident or a mutant
    colors = [G.nodes[i]['type'].color for i in G.nodes()]

    nx.draw_circular(G, with_labels=True, node_color=colors)
    plt.show()

def initializeNodesAsResident(G):
    #Initialize edge weights to be uniformly distributed
    for node1, node2, data in G.edges(data=True):
        data['weight'] = 1/len(G.adj[node1])

    for i in G.nodes():
        #Initialize node as Resident
        nodeType = mp.Resident(1)
        G.nodes[i]['type'] = nodeType

        #Initialize multiplier to one
        G.nodes[i]['multiplier'] = 1

        #Print edges and their weight
        #print(G.adj[i])

    # Make graph directed
    G = G.to_directed()
    return G

if __name__ == "__main__":
    createCompleteGraph()



