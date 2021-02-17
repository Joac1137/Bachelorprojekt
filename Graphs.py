import networkx as nx
import matplotlib.pyplot as plt
import Moran_Process as mp

def createCompleteGraph():
    G = nx.complete_graph(5)
    G = initializeNodesAsResident(G)

    drawGraph(G)

def createKarateClubGraph():
    G = nx.karate_club_graph()
    drawGraph(G)

def drawGraph(G):
    # The drawn colors are created from whether the node is a resident or a mutant
    colors = [G.nodes[i]['type'].color for i in G.nodes()]

    nx.draw_circular(G, with_labels=True, node_color=colors)
    plt.show()

def initializeNodesAsResident(G):
    for i in G.nodes():
        nodeType = mp.Resident(1)
        G.nodes[i]['type'] = nodeType
    print(list(G.nodes.data()))
    print(G.nodes[0]['type'].color)

    return G

if __name__ == "__main__":
    createCompleteGraph()



