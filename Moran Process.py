import networkx as nx
import matplotlib.pyplot as plt




if __name__ == "__main__":
   G = nx.Graph()
   G.add_node(1)

   G.add_nodes_from([2,3])
   G.add_nodes_from([
       (4, {"color": "red"}),
       (5, {"color": "green"}),
   ])

   G.add_edge(1,2)
   e = (2,3)
   G.add_edge(*e)

   G.add_edges_from([(1,2),(1,3)])

   print(list(G.nodes))


   H = nx.karate_club_graph()
   print(H)
   print(H.nodes[5]["club"])

   print(H.nodes[9]["club"])

   nx.draw_circular(H, with_labels=True)
   plt.show()




