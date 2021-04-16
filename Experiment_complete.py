import Graphs
import networkx as nx


def calculate_markov_weights(i, j, G, fitness, active_nodes):
    active_mutant_before = int(i.split(',')[0])
    passive_mutant_before = int(i.split(',')[1])
    active_mutant_after = int(j.split(',')[0])
    passive_mutant_after = int(j.split(',')[1])

    total_fitness = active_mutant_before*(1 + fitness) + len(G.nodes()) - active_mutant_before
    weight = 0
    if active_mutant_before < active_mutant_after:
        weight = ((active_mutant_before*(1+fitness)/total_fitness) + (passive_mutant_before/total_fitness))*((len(G.nodes())-(active_mutant_before + passive_mutant_before))/len(G.nodes()))

    elif passive_mutant_before < passive_mutant_after:
        weight = ((active_mutant_before*(1+fitness)/total_fitness) + (passive_mutant_before/total_fitness))*(active_nodes-active_mutant_before)

    elif active_mutant_before > active_mutant_after:
        weight = ((len(G.nodes()) - active_mutant_before - passive_mutant_before)/total_fitness) * (active_mutant_before/len(G.nodes()))

    elif passive_mutant_before > passive_mutant_after:
        weight = ((len(G.nodes()) - active_mutant_before - passive_mutant_before)/total_fitness) * (passive_mutant_before/len(G.nodes()))

    return weight




def add_weights_to_edges(markov, G, fitness,active_nodes):

    for i, j, data in markov.edges(data = True):
        data['weight'] = calculate_markov_weights(i,j,G,fitness,active_nodes)

    return markov


def create_complete_markov_chain(G, active_nodes,fitness):
    extinction_node = '0,0'
    markov = nx.DiGraph()
    markov.add_node(extinction_node)
    markov.add_edge(extinction_node,extinction_node)
    markov[extinction_node][extinction_node]['weight'] = 1

    previous_nodes = [extinction_node]
    for i in range(1,len(G.nodes()) + 1):
        current_nodes = []
        x = i if i <= active_nodes else active_nodes
        y = i - x

        while x >= 0 and y <= len(G.nodes()) - active_nodes:
            node = str(x) + ',' + str(y)
            markov.add_node(node)

            for i in previous_nodes:
                markov.add_edge(node,i)
                markov.add_edge(i,node)
            current_nodes.append(node)
            x = x - 1
            y = y + 1
        previous_nodes = current_nodes

    markov = add_weights_to_edges(markov,G,fitness,active_nodes)
    return markov




if __name__ == '__main__':
    fitness = 2
    multiplier = 1
    graph_size = 3
    active_nodes = 1

    G = Graphs.create_complete_graph(graph_size)
    Graphs.initialize_nodes_as_resident(G,multiplier)
    Graphs.draw_graph(G)

    markov_chain = create_complete_markov_chain(G,active_nodes,fitness)
    Graphs.draw_markov_model(markov_chain)


