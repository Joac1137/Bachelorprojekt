import copy
import networkx as nx
import matplotlib.pyplot as plt



def draw_beautiful(g):
    # Get positions.
    # Here I use the spectral layout and add a little bit of noise.
    pos = nx.layout.spectral_layout(g)
    pos = nx.spring_layout(g, pos=pos, iterations=50)

    # Create position copies for shadows, and shift shadows
    pos_shadow = copy.deepcopy(pos)
    shift_amount = 0.006
    for idx in pos_shadow:
        pos_shadow[idx][0] += shift_amount
        pos_shadow[idx][1] -= shift_amount

    #~~~~~~~~~~~~
    # Draw graph
    #~~~~~~~~~~~~
    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    nx.draw_networkx_nodes(g, pos_shadow, node_color='k', alpha=0.5)
    nx.draw_networkx_nodes(g, pos, node_color="#3182bd", linewidths=1)
    nx.draw_networkx_edges(g, pos, width=1)
