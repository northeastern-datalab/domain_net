import networkx as nx
import networkit as nk
import pandas as pd

import utils

import pickle
import json
import argparse

from timeit import default_timer as timer
from pathlib import Path

def get_graph_statistics_networkit(G_nx, graph_type, output_dir, computation_mode, num_samples):
    '''
    Returns a pandas dataframe of relevant statistical measurements for the input graph `G`
    Each row corresponds to one node in the graph.

    It uses networkit functions for its calculations

    Arguments
    -------
        G_nx (networkx graph): a networkx graph to be analyzed

        graph_type (str): one of {'bipartite', 'cell_graph'}, specifies the type of graph
        so that the appropriate analysis can be run.

        computation_mode (str): one of {'all', 'exact', 'approximate'} specifies if an approximation
        of the betweenness score is computed or an exact computation is carried. If 'all' then both
        are carried out 
       
    Returns
    -------
    pandas dataframe for the input graph. Each row corresponds to one node and the
    various columns corresponds to the respective measures for that node.
    '''
    print('Input graph has:', G_nx.number_of_nodes(), 'nodes and', G_nx.number_of_edges(), 'edges\n')

    # Convert NetworkX graph into networkit graph
    start = timer()
    print('Converting NetworkX graph into a networkit graph...')
    G = nk.nxadapter.nx2nk(G_nx)
    print('Finished converting NetworkX graph into a networkit graph \nElapsed time:', timer()-start, 'seconds\n')

    # Create dictionaries that map networkx IDs to networkit IDs and vice-versa
    start = timer()
    print('Creating + Saving mapping dictionaries for nodes between NetworkX and networkit...')
    nx_to_nk_id_dict = dict((id, int_id) for (id, int_id) in zip(G_nx.nodes(), range(G_nx.number_of_nodes())))
    nk_to_nx_id_dict = dict((int_id, id) for (id, int_id) in zip(G_nx.nodes(), range(G_nx.number_of_nodes())))

    # Save the two dictionaries in the output_dir
    with open(output_dir + 'nx_to_nk_id_dict.pickle', 'wb') as handle:
        pickle.dump(nx_to_nk_id_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_dir + 'nk_to_nx_id_dict.pickle', 'wb') as handle:
        pickle.dump(nk_to_nx_id_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Finished Creating + Saving mapping dictionaries for nodes between NetworkX and networkit\
        \nElapsed time:', timer()-start, 'seconds\n')

    # Construct the dataframe placeholder
    df = pd.DataFrame()
    df['node'] = nx_to_nk_id_dict.keys()

    # Add node type as a column
    node_types = [G_nx.nodes[node]['type'] for node in df['node'].values]
    df['node_type'] = node_types

    start = timer()
    print('Calculating betweeness scores...')

    if computation_mode == 'all' or computation_mode == 'exact':
        df['betweenness_centrality'] = utils.betweenness.betweeness_exact(G)
    if computation_mode == 'all' or computation_mode == 'approximate':
        # TODO figure out an appropriate sample size
        df['approximate_betweenness_centrality'] = utils.betweenness.betweeness_approximate(G, num_samples=num_samples)

    print('Finished calculating betweeness scores \nElapsed time:', timer()-start, 'seconds\n')


    # TODO: Add clustering coefficients for bipartite graphs (currently networkit doesn't support it for bipartite graphs)

    return df


def get_graph_statistics(G, mode):
    '''
    DEPRECATE: OLD implementation using networkx

    Returns a pandas dataframe of relevant statistical measurements for the input graph `G`
    Each row corresponds to one node in the graph.

    Arguments
    -------
        G (networkx graph): a networkx graph to be analyzed

        mode (str): one of ['bipartite', 'cell_graph'], specifies the type of graph
        so that the appropriate analysis can be run.
       
    Returns
    -------
    pandas dataframe for the input graph. Each row corresponds to one node and the
    various columns corresponds to the respective measures for that node.
    '''
    print('Input graph has:', G.number_of_nodes(), 'nodes and', G.number_of_edges(), 'edges.')

    density = nx.function.density(G)
    print('Density:', density, '\n')

    # Calculate various measures on a per-node level
    if mode == 'cell_graph':
        start = timer()
        print('Calculating betweeness centrality...')
        betweenness_centrality = nx.algorithms.centrality.betweenness_centrality(G)
        print('Finished calculating betweeness centrality \n Elapsed time:', timer()-start, 'seconds\n')

        start = timer()
        print('Calculating local clustering coefficient...')
        local_clustering_coefficient = nx.algorithms.cluster.clustering(G)
        print('Finished calculating local clustering coefficient \n Elapsed time:', timer()-start, 'seconds\n')

        # Construct the dataframe
        df = pd.DataFrame()
        df['node'] = betweenness_centrality.keys()
        df['betweenness_centrality'] = betweenness_centrality.values()
        df['local_clustering_coefficient'] = local_clustering_coefficient.values()

        return df
    elif mode == 'bipartite':
        # Find how many cell nodes only appear in one column (i.e. they have degree of 1)
        cell_nodes = {n for n, d in G.nodes(data=True) if d['type']=='cell'}
        degree_view = G.degree(cell_nodes)

        num_nodes_with_degree_greater_than_one = 0
        for node in cell_nodes:
            if degree_view[node] > 1:
                num_nodes_with_degree_greater_than_one += 1

        print('There are', num_nodes_with_degree_greater_than_one, 'cell nodes with degree greater than one. That is',\
        str(num_nodes_with_degree_greater_than_one / len(cell_nodes) * 100) + '% of all cell nodes.')

        start = timer()
        print('Calculating local clustering coefficient using dot mode...')
        local_clustering_coefficient_dot = nx.algorithms.bipartite.cluster.clustering(G, mode='dot')
        print('Finished calculating local clustering coefficient using dot mode')
        print('Elapsed time:', timer()-start, 'seconds\n')

        start = timer()
        print('Calculating local clustering coefficient using min mode...')
        local_clustering_coefficient_min = nx.algorithms.bipartite.cluster.clustering(G, mode='min')
        print('Finished calculating local clustering coefficient using min mode')
        print('Elapsed time:', timer()-start, 'seconds\n')

        start = timer()
        print('Calculating local clustering coefficient using max mode...')
        local_clustering_coefficient_max = nx.algorithms.bipartite.cluster.clustering(G, mode='max')
        print('Finished calculating local clustering coefficient using max mode')
        print('Elapsed time:', timer()-start, 'seconds\n')

        start = timer()
        cell_nodes = {n for n, d in G.nodes(data=True) if d['type']=='cell'}
        print('Calculating betweeness centrality...')
        betweenness_centrality = nx.algorithms.bipartite.centrality.betweenness_centrality(G, nodes=cell_nodes)
        print('Finished calculating betweeness centrality')
        print('Elapsed time:', timer()-start, 'seconds\n')

        # Construct the dataframe
        df = pd.DataFrame()
        df['node'] = betweenness_centrality.keys()
        df['betweenness_centrality'] = betweenness_centrality.values()
        df['local_clustering_coefficient_dot'] = local_clustering_coefficient_dot.values()
        df['local_clustering_coefficient_min'] = local_clustering_coefficient_min.values()
        df['local_clustering_coefficient_max'] = local_clustering_coefficient_max.values()
        
        return df

def main(args):
    # Load the graph file
    start = timer()
    print('Loading graph file...')
    graph = pickle.load(open(args.graph, 'rb'))
    print('Finished loading graph file \nElapsed time:', timer()-start, 'seconds\n')

    if args.perform_cleaning:
        # Perform cleaning on the graph file
        graph = utils.cleaning.clean_graph(G=graph,
            min_str_length=args.min_str_length,
            remove_numerical_vals=args.remove_numerical_vals
        )

    if args.collapsed_graph:
        # Replace `graph` with its collapsed version
        graph = utils.graph_collapsing.get_collapsed_graph(graph)

        # Save the collapsed graph into the output directory
        nx.write_gpickle(graph, args.output_dir+'collapsed_graph.pickle')
   
    if not args.existing_computation:
        if args.groundtruth_path != None:
            # Get groundtruth
            groundtruth = pickle.load(open(args.groundtruth_path, 'rb'))

            # Get the homographs from the groundtruth
            node_is_homograph_df = utils.groundtruth.get_homographs_from_groundtruth(
                G = graph,
                groundtruth = groundtruth,
                output_dir = args.output_dir
            )

        # Construct the graph stats dataframe
        graph_stats_df = get_graph_statistics_networkit(
            G_nx = graph,
            graph_type=args.mode,
            output_dir=args.output_dir,
            computation_mode=args.betweenness_mode,
            num_samples = args.num_samples
        )
        
        # Save the dataframe to file
        graph_stats_df.to_pickle(args.output_dir + 'graph_stats_df.pickle')

        if args.groundtruth_path != None:
            # Combine the graph_stats_df with the node_is_homograph_df to augment the dataframe with the groundtruth
            graph_stats_df = pd.merge(graph_stats_df, node_is_homograph_df, on='node', how='outer')
            # Save the dataframe to file
            graph_stats_df.to_pickle(args.output_dir + 'graph_stats_with_groundtruth_df.pickle')
    else:
        # Existing computation for graph_stats_df exists so just load it from file
        print('Loading graph_stats_df from file...')
        if args.groundtruth_path != None:
            graph_stats_df = pickle.load(open(args.output_dir + 'graph_stats_with_groundtruth_df.pickle', 'rb'))
            node_is_homograph_df = pickle.load(open(args.output_dir + 'node_is_homograph_df.pickle', 'rb'))
        else:
            graph_stats_df = pickle.load(open(args.output_dir + 'graph_stats_df.pickle', 'rb'))
        print('Finished loading graph_stats_df from file\n')

    # Perform further analysis if specified
    if args.betweenness_in_k_neighborhood:
        min_radius = args.betweenness_in_k_neighborhood[0]
        max_radius = args.betweenness_in_k_neighborhood[1]
        radius_step_size = args.betweenness_in_k_neighborhood[2]

        nodes = graph_stats_df['node']
        for radius in range(min_radius, max_radius+1, radius_step_size):
            start = timer()
            print('Calculating BC in for all nodes for neighborhood with radius', radius)
            graph_stats_df['BC_at_radius_' + str(radius)] = utils.betweenness_for_nodes_in_k_neighborhood(graph, nodes, radius)
            print('Finished calculating BC in for all nodes for neighborhood with radius', radius,
            'Elapsed time:', timer()-start, 'seconds\n')

        # Updated and save the dataframe: graph_stats_df 
        graph_stats_df.to_pickle(args.output_dir + 'graph_stats_df.pickle')


if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Perform network analysis on a given repository of tables \
    and identify potential homograph terms')

    # Output directory where output files and figures are stored
    parser.add_argument('-o', '--output_dir', metavar='output_dir', required=True,
    help='Path to the output directory where output files and figures are stored. \
    Path must terminate with backslash "\\"')

    # Input graph representation of the set of tables
    parser.add_argument('-g', '--graph', metavar='graph', required=True,
    help='Path to the Graph representation of the set of tables')

    # Denotes if we perform a cleaning over the nodes in the input graph
    parser.add_argument('--perform_cleaning', action='store_true', 
    help='Denotes if we perform a cleaning over the nodes in the input graph')

    parser.add_argument('--min_str_length', type=int,
    help='The minimum length of a string, smaller length strings are not considered after cleaning')

    # If specified removes all nodes from the graph with numerical values in their name
    parser.add_argument('--remove_numerical_vals', action='store_true', 
    help='If specified removes all nodes from the graph with numerical values in their name')

    # The graph representation used. One of 'cell_graph' or 'bipartite'
    parser.add_argument('-m', '--mode', default='bipartite', choices=['cell_graph', 'bipartite'],
    help='The graph representation used. One of cell_graph or bipartite')

    # Specifies the path to the ground truth file used in the benchmark computation
    parser.add_argument('-gt', '--groundtruth_path', metavar='groundtruth_path',
    help='Specifies the ground truth file used in the benchmark computation.\
    If not specified the it is assumed that we do not know in ground truth which words are homographs.')

    # Denotes if the graph statistics dataframe was already been built, so it can be skipped.
    parser.add_argument('--existing_computation', action='store_true', 
    help='Denotes if the graph statistics dataframe was already been built, so it can be skipped.')

    # The mode for calculating the betweeness centrality. One of {all, exact, approximate}. If all
    # then we calculate both the exact and approximate betweeness
    parser.add_argument('-bm', '--betweenness_mode', default='approximate', choices=['all', 'exact', 'approximate'],
    help='The mode for calculating the betweeness centrality. One of {all, exact, approximate}.\
     If all then we calculate both the exact and approximate betweeness')

    # Denotes the radius for the betweenness_in_k_neighborhood calculations. Argument is a range of min to max radius and the step.
    parser.add_argument('--betweenness_in_k_neighborhood', nargs=3, type=int, metavar=('min_radius', 'max_radius', 'step'),
    help='If specified we compute the betweenness_in_k_neighborhood in the specified radius range.')

    # If specified, then use a collapsed version of the input graph to perform the analysis
    parser.add_argument('--collapsed_graph', action='store_true',
    help='If specified, then use a collapsed version of the input graph to perform the analysis')   

    # Number of nodes to sample to approximately calculate the betweeness 
    # Used only with conjunction with betweenness_mode set to approximate. 
    parser.add_argument('--num_samples', type=int, 
    help='Number of nodes to sample to approximately calculate the betweeness \
    Used only with conjunction with betweenness_mode set to approximate.')

    # Parse the arguments
    args = parser.parse_args()

    # Check for argument consistency
    if args.perform_cleaning and (not args.min_str_length and not args.remove_numerical_vals):
        parser.error('To perform you must specify at least one of min_str_length or remove_numerical_vals arguments')
    
    print('##### ----- Running network_analysis/main.py with the following parameters ----- #####\n')

    print('Output directory:', args.output_dir)
    print('Graph path:', args.graph)
    print('Ground Truth path:', args.groundtruth_path)
    print('Existing computation:', args.existing_computation)
    print('Betweenness mode:', args.betweenness_mode)
    if args.betweenness_in_k_neighborhood:
        print('Betweenness in k neighborhood in the range:',  args.betweenness_in_k_neighborhood[0],
        '-', args.betweenness_in_k_neighborhood[1], 'with step size:', args.betweenness_in_k_neighborhood[2])
    if args.betweenness_mode == 'approximate':
        print('Number of samples:', args.num_samples)
    if args.collapsed_graph:
        print('Using collapsed graph for analysis')
    print()
    if args.perform_cleaning:
        print('Cleaning is set: ON')
        if args.min_str_length:
            print('Minimum string length:', args.min_str_length)
        if args.remove_numerical_vals:
            print('Removing numerical valued nodes')
    else:
        print('Cleaning is set: OFF')
    print('\n\n')

    # Create the output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save the input arguments in the output_dir
    with open(args.output_dir + 'args.json', 'w') as fp:
        json.dump(vars(args), fp, sort_keys=True, indent=4)

    main(args)