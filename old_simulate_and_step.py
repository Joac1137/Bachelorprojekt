def simulate(n, G, fitness_mutant,lowest_acceptable_fitness=0):
    fixation_counter = 0
    fixation_list = list()
    number_of_nodes = len(G.nodes)
    old_graph = G.copy()
    counter = 1
    fix_prop_this_round =0
    start_time = time.time()
    while (counter < n or fix_prop_this_round <= lowest_acceptable_fitness) and not counter > 50000:
        G = old_graph.copy()
        mutate_a_random_node(G, fitness_mutant)
        #print("Whats the fitness ", fitness_mutant)
        #print("The graph ", G.nodes(data=True))
        # Choose a node based on fitness and the multiplier
        fitness_distribution = list()
        for i in G.nodes():
            # The below logic implements the fact that only active nodes can take advantage of their multiplier
            # Fitness
            fitness = G.nodes[i]['type'].fitness
            #print("Fitness ", fitness)
            is_active = G.nodes[i]['active']
            #print("Is active ", is_active)
            # Multiplier for node
            multiplier = 1 if is_active else 0
            fitness_distribution.append(1 + multiplier * fitness)
        #print("Distribution ", fitness_distribution)

        # Does a Moran Step whenever we do not have the same color in the graph
        k = 1
        terminated = False
        while not terminated:
            i, fitness_distribution = step(G,fitness_distribution)
            k += i # Step() now returns the difference in number of mutants after_step - before_step
            terminated = k == number_of_nodes or k == 0
        if k == number_of_nodes:
            fixation_counter += 1
        # while not have_we_terminated(G):
        #     step(G)
        # fixation_counter += is_the_first_node_mutant(G)
        fix_prop_this_round = fixation_counter / counter
        fixation_list.append(fix_prop_this_round)
        counter += 1
    print("Rounds ", counter)
    end_time = time.time()
    print("Simulation took", end_time-start_time, "Seconds")
    return fixation_list, fixation_counter / counter


def step(G,fitness_distribution):
    # Get all neighboring nodes and walk on a edge based upon the weights


    # Nodes as a list
    nodes = range(0, len(G.nodes()))
    replicating_node_index = random.choices(nodes, weights=fitness_distribution, k=1)[0]
    # Mutate a neighbor based on the weights of the edges
    # Find all node neighbors
    neighbors = G.edges(replicating_node_index)
    # Get the corresponding weights
    edge_weights = [G.get_edge_data(x, y)['weight'] for x, y in neighbors]
    neighbor_nodes = [y for x, y in neighbors]
    # Choose one edge to walk on
    #print("Neighbors ", neighbor_nodes)
    node_to_mutate = random.choices(neighbor_nodes, weights=edge_weights, k=1)[0]
    #print("Node ", node_to_mutate)
    if G.nodes[node_to_mutate]['type'] != G.nodes[replicating_node_index]['type']:
        if G.nodes[replicating_node_index]['type'].id_n == 'resident':
            res = -1
        else:
            res = 1
        G.nodes[node_to_mutate]['type'] = G.nodes[replicating_node_index]['type']
        node_fitness = G.nodes[node_to_mutate]['type'].fitness
        is_active = G.nodes[node_to_mutate]['active']
        # Multiplier for node
        multiplier = 1 if is_active else 0
        fitness_distribution[node_to_mutate] = 1 + multiplier * node_fitness
    else:
        res = 0

    return res,fitness_distribution