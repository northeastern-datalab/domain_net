import networkit as nk
import networkx as nx

from timeit import default_timer as timer
from tqdm import tqdm

def betweeness_exact(G, quiet=False):
    '''
    Calculates the exact betweenness for networkit graph `G`. Returns the exact betweenness of the nodes in a list
    in the order of the nodes in `G`

    Arguments
    -------
        G (networkit graph): a networkit graph to be analyzed

    Returns
    -------
    List of exact betweenness scores for each node
    '''
    start = timer()
    # Calculate betweeness centrality for each node
    betweeness = nk.centrality.Betweenness(G, normalized=True).run()
    betweeness_scores = betweeness.scores()

    if not quiet:
        # Estimate of computations per second (betweeness has a |nodes|*|edges| complexity)
        num_computations = G.numberOfEdges() * G.numberOfNodes()

        # Computation done per second
        computations_per_second = "{:4e}".format(num_computations / (timer()-start))
        print('Number of computations (i.e. edges*nodes):', "{:.4e}".format(num_computations))
        print(computations_per_second, 'computations per second\n')

    return betweeness_scores

def betweeness_approximate(G, num_samples=5000, quiet=False):
    '''
    Calculates an approximation the the betweenness scores for each node in networkit graph `G`

    Arguments
    -------
        G (networkit graph): a networkit graph to be analyzed

        num_samples (int): number of samples to be used in the approximate calculation of betweenness

    Returns
    -------
    List of approximate betweenness scores for each node
    '''
    start = timer()
    betweeness = nk.centrality.EstimateBetweenness(G, nSamples = num_samples, parallel=True, normalized=True).run()
    approximate_betweeness_scores = betweeness.scores()

    if not quiet:
        # Estimate betweenness O(m * num_samples) complexity
        num_computations_approx = G.numberOfEdges() * num_samples

        # Computation done per second
        computations_per_second = "{:4e}".format(num_computations_approx / (timer()-start))
        print('Number of computations (i.e. edges*num_samples):', "{:.4e}".format(num_computations_approx))
        print(computations_per_second, 'computations per second\n')

    return approximate_betweeness_scores

def betweenness_for_nodes_in_k_neighborhood(G, nodes, radius, exact=True):
    '''
    Calculate the betweeness centrality for every node in `nodes` in graph `G` by only considering the
    `radius` neighbors of `node`. In other, words the betweeness centrality returned
    is only calculated on the ego graph of `node` with radius `radius`.

    Arguments
    -------
        G (networkX graph): a networkX input graph

        node (list of node identifiers): node that the ego graph is based on

        radius (int): radius considered in the ego graph

    Returns
    -------
    List of BC scores for each node in the `nodes` list based on its radius and ego graph
    '''
    bc_scores_in_k_neighborhood = []

    # Because for many nodes their egonet at a radius may be identical then store the subgraphs we have
    # already examined together with the BC of every node in that subgraph. Each subgraph is defined by the set
    # of nodes it contains. Two subgraphs are identical if they have the same set of nodes, because they are always
    # constructed from the same bigger graph `G`

    # A dictionary keyed by the computed subgraphs to the subgraph id
    # Recall that each subgraph is defined as the set of nodes it is made of.
    # So the key of this dictionary is a set, specifically a frozenset
    computed_subgraphs_to_subgraph_id = {}

    # ID to use for the next newly identified subgraph
    cur_subgraph_id=0

    # Dictionary keyed by the subgraph id to a dictionary of BC scores for each node in the subgraph
    subgraph_id_to_bc_scores_dict = {}

    # The sum of all (number_of_nodes * number_of_edges) from all newly computed ego graphs.
    num_computations = 0

    for cur_node in tqdm(nodes):
        # Get cur_node's ego graph
        ego_graph = nx.ego_graph(G, cur_node, radius=radius)

        # Check if this ego_graph was previously analysed
        if frozenset(ego_graph.nodes) in computed_subgraphs_to_subgraph_id:
            # We already know the BC scores for every node in this
            subgraph_id = computed_subgraphs_to_subgraph_id[frozenset(ego_graph.nodes)]
            bc_score_for_cur_node = subgraph_id_to_bc_scores_dict[subgraph_id][cur_node]
            bc_scores_in_k_neighborhood.append(bc_score_for_cur_node)
        else:
            # Update the computed_subgraphs_to_subgraph_id
            computed_subgraphs_to_subgraph_id[frozenset(ego_graph.nodes)] = cur_subgraph_id

            # Calculate the BC scores for all nodes in the ego_graph using Networkit
            ego_graph_nk = nk.nxadapter.nx2nk(ego_graph)
            nx_to_nk_id_dict = dict((id, int_id) for (id, int_id) in zip(ego_graph.nodes(), range(ego_graph.number_of_nodes())))

            if exact:
                betweeness_scores = betweeness_exact(ego_graph_nk, quiet=True)
            else:
                betweeness_scores = betweeness_approximate(ego_graph_nk, quiet=True)

            # Construct dictionary of node_name to BC score
            bc_dict = {}
            for node_name, nk_node_id in nx_to_nk_id_dict.items():
                bc_dict[node_name] = betweeness_scores[nk_node_id]
            
            subgraph_id_to_bc_scores_dict[cur_subgraph_id] = bc_dict
            bc_scores_in_k_neighborhood.append(bc_dict[cur_node])

            # Increment the cur_subgraph_id
            cur_subgraph_id += 1

            # Update the number of computations
            num_computations += (ego_graph.number_of_nodes() * ego_graph.number_of_edges())

    print('Computed BC for', len(computed_subgraphs_to_subgraph_id), 'subgraphs/egonets')
    print('Number of computations for all egonets is:', num_computations)
    print('Number of computations on the full graph is:', G.number_of_nodes() * G.number_of_edges())
    return bc_scores_in_k_neighborhood

