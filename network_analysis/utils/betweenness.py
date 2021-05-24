import networkit as nk
import networkx as nx
import pandas as pd

from timeit import default_timer as timer
from tqdm import tqdm

def betweeness_exact_df(G, normalized=True, quiet=False):
    '''
    Calculates the exact betweenness for every node in a networkx graph.
    The computation is carried using networkit.

    Arguments
    -------
        G (networkx graph): a networkx graph. Must be a bipartite graph with node types 'cell' and 'attr'

        normalized (bool): specifies if the BC scores need to be normalized

        quiet (bool): specifies if informational print statements are presented  

    Returns
    -------
    A dataframe keyed by each node with a mapping to its respective 
    'node_type' and exact 'betweenness_centrality' score
    '''

    G_nk = nk.nxadapter.nx2nk(G)
    nx_to_nk_id_dict = dict((id, int_id) for (id, int_id) in zip(G.nodes(), range(G.number_of_nodes())))

    # Construct the dataframe placeholder
    df = pd.DataFrame()
    df['node'] = list(nx_to_nk_id_dict.keys())

    # Add node type as a column
    node_types = [G.nodes[node]['type'] for node in df['node'].values]
    df['node_type'] = node_types
    df['betweenness_centrality'] = betweeness_exact(G_nk, normalized=normalized, quiet=quiet)

    return df

def betweenness_approximate_df(G, normalized=True, quiet=False, num_samples=5000, source_target_nodes_list=None, ident=None, seed=0, column_name='approximate_betweenness_centrality'):
    '''
    Calculates the approximate betweenness centrality for every node in a networkx graph.
    The computation is carried using networkit.

    Arguments
    -------
        G (networkx graph): a networkx graph. Must be a bipartite graph with node types 'cell' and 'attr'

        normalized (bool): specifies if the BC scores need to be normalized

        quiet (bool): specifies if informational print statements are presented

        num_samples (int): number of samples to be used in the approximate calculation of betweenness

        source_target_nodes_list (list of int): a list of node IDs to be used as source and target nodes for the approximate BC computation.
        If the list set to None then all nodes in the graph are used as source/target nodes

        ident (list of int): a list of the ident values for each node in the graph.
        The order of the list is the same as the order of the nodes in the graph

        seed (int): seed used for sampling nodes for the approximate BC computation

        column_name (str): name of the dataframe column where the BC scores are stored. By default it is "approximate_betweenness_centrality"

    Returns
    -------
    A dataframe keyed by each node with a mapping to its respective 
    'node_type' and exact 'betweenness_centrality' score
    '''

    G_nk = nk.nxadapter.nx2nk(G)
    nx_to_nk_id_dict = dict((id, int_id) for (id, int_id) in zip(G.nodes(), range(G.number_of_nodes())))

    # Construct the dataframe placeholder
    df = pd.DataFrame()
    df['node'] = list(nx_to_nk_id_dict.keys())

    # Add node type as a column
    node_types = [G.nodes[node]['type'] for node in df['node'].values]
    df['node_type'] = node_types
    df[column_name] = betweeness_approximate(
        G_nk, normalized=normalized, num_samples=num_samples,
        source_target_nodes_list=source_target_nodes_list, ident=ident, seed=seed, quiet=quiet)

    return df


def betweeness_exact(G, normalized=True, quiet=False):
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
    betweeness = nk.centrality.Betweenness(G, normalized=normalized).run()
    betweeness_scores = betweeness.scores()

    if not quiet:
        # Estimate of computations per second (betweeness has a |nodes|*|edges| complexity)
        num_computations = G.numberOfEdges() * G.numberOfNodes()

        # Computation done per second
        computations_per_second = "{:4e}".format(num_computations / (timer()-start))
        print('Number of computations (i.e. edges*nodes):', "{:.4e}".format(num_computations))
        print(computations_per_second, 'computations per second\n')

    return betweeness_scores

def betweeness_approximate(G, num_samples=5000, quiet=False, source_target_nodes_list=None, ident=None, normalized=True, seed=0):
    '''
    Calculates an approximation the the betweenness scores for each node in networkit graph `G`

    Arguments
    -------
        G (networkit graph): a networkit graph to be analyzed

        num_samples (int): number of samples to be used in the approximate calculation of betweenness

        source_target_nodes_list (list of int): a list of node IDs to be used as source and target nodes for the approximate BC computation.
        If the list set to None then all nodes in the graph are used as source/target nodes

        ident (list of int): a list of the ident values for each node in the graph.
        The order of the list is the same as the order of the nodes in the graph

        normalized (bool): specifies if the BC scores need to be normalized

        seed (int): seed used for sampling nodes for the approximate BC computation

    Returns
    -------
    List of approximate betweenness scores for each node
    '''

    if ident is None:
        # If ident is not specified then every node has an ident value 1
        ident = [1] * G.numberOfNodes()

    start = timer()
    betweeness = nk.centrality.EstimateBetweenness(
        G,
        nSamples = num_samples,
        parallel=True,
        normalized=normalized,
        seed=seed,
        sources=source_target_nodes_list,
        targets=source_target_nodes_list,
        ident=ident,
    ).run()
    approximate_betweeness_scores = betweeness.scores()

    if not quiet:
        elapsed_time = timer() - start
        print('Elapsed Time:', elapsed_time, 'seconds')

        # Estimate betweenness O(m * num_samples) complexity
        num_computations_approx = G.numberOfEdges() * num_samples

        # Computation done per second
        computations_per_second = "{:4e}".format(num_computations_approx / elapsed_time)
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

