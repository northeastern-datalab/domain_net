import networkx as nx
import networkit as nk
import pandas as pd
import random
import sys
import math

import utils

import pickle
import json
import argparse

from timeit import default_timer as timer
from pathlib import Path

def get_source_target_nodes_list(G, source_target_nodes):
    '''
    Arguments
    -------
        G (networkx graph): a networkx graph to be analyzed

        source_target_nodes (str): one of {'all', 'cell_nodes', 'attribute_nodes'} specifying what type of nodes are used 
        as source/target nodes in the BC computation.
       
    Returns
    -------
    A list of node IDs to be used as source and target nodes for the approximate BC computation.
    If the returned list is set to None then all nodes in the graph are used as source/target nodes
    '''
    source_target_nodes_list = None
    if source_target_nodes == 'all':
        return source_target_nodes_list
    else:
        # Find appropriate networkit node IDs for cell or attr nodes
        nx_to_nk_id_dict = dict((id, int_id) for (id, int_id) in zip(G.nodes(), range(G.number_of_nodes())))

        if source_target_nodes == 'cell_nodes':
            cell_nodes = [x for x,y in G.nodes(data=True) if y['type']=='cell']
            source_target_nodes_list = [nx_to_nk_id_dict[node] for node in cell_nodes]
        elif source_target_nodes == 'attribute_nodes':
            attr_nodes = [x for x,y in G.nodes(data=True) if y['type']=='attr']
            source_target_nodes_list = [nx_to_nk_id_dict[node] for node in attr_nodes]

        return source_target_nodes_list

def get_num_samples(G, sampling_percentage):
    '''
    Arguments
    -------
        G (networkx graph): a networkx graph to be analyzed

        sampling_percentage (float): percentage of graph nodes sampled for approximate betweenness centrality
       
    Returns
    -------
    Returns an integer amount of nodes to be sampled from graph G. The returned integer must be between
    1 and G.number_of_nodes()
    '''
    num_samples = math.ceil(G.number_of_nodes() * (sampling_percentage/100))
    if num_samples < 1:
        num_samples = 1
    if num_samples > G.number_of_nodes():
        num_samples = G.number_of_nodes()

    return num_samples




def get_graph_statistics_networkit(G_nx, output_dir, computation_mode, num_samples, sampling_percentage, source_target_nodes, seed, node_compression):
    '''
    Returns a pandas dataframe of relevant statistical measurements for the input graph `G`
    Each row corresponds to one node in the graph.

    It uses networkit functions for its calculations

    Arguments
    -------
        G_nx (networkx graph): a networkx graph to be analyzed

        output_dir (str): output directory for the  networkx to networkit node ID dictionaries

        computation_mode (str): one of {'all', 'exact', 'approximate'} specifies if an approximation
        of the betweenness score is computed or an exact computation is carried. If 'all' then both
        are carried out

        num_samples (int): number of samples nodes used for approximate betweenness centrality.
        If not specified then sampling_proportion must be specified

        sampling_percentage (float): percentage of graph nodes sampled for approximate betweenness centrality.
        If not specified then num_samples must be specified 

        source_target_nodes (str): one of {'all', 'cell_nodes', 'attribute_nodes'} specifying what type of nodes are used 
        as source/target nodes in the BC computation.

        seed (int): seed used for sampling nodes for approximate betweenness centrality

        node_compression (boolean): if true then find set of cell nodes that co-occur with the same set of attribute nodes
        and compress these cell nodes into a single compressed node
       
    Returns
    -------
    pandas dataframe for the input graph. Each row corresponds to one node and the
    various columns corresponds to the respective measures for that node.
    '''
    print('Input graph has:', G_nx.number_of_nodes(), 'nodes and', G_nx.number_of_edges(), 'edges\n')

    # Run BC using the compressed graph
    if node_compression:
        G_nx_compressed, compressed_node_to_orig_nodes = utils.graph_collapsing.get_compressed_graph(G_nx)
        print("\nCompressed graph has", G_nx_compressed.number_of_nodes(), 'nodes and', G_nx_compressed.number_of_edges(), 'edges.')

        # Get the ident_dict for the compressed graph
        ident_dict = utils.graph_collapsing.get_ident_dict(G_nx_compressed, compressed_node_to_orig_nodes)
        print("There are", len(compressed_node_to_orig_nodes), 'compressed nodes')

        # Choose the number of samples over the compressed graph
        if sampling_percentage:
            num_samples = get_num_samples(G_nx_compressed, sampling_percentage)
            print("Sampling", num_samples, "nodes from the compressed graph\n")

        source_target_nodes_list = get_source_target_nodes_list(G_nx, source_target_nodes)

        df = utils.betweenness.betweenness_approximate_df(
            G=G_nx_compressed,
            normalized=True,
            quiet=False,
            num_samples=num_samples,
            source_target_nodes_list=source_target_nodes_list,
            ident=list(ident_dict.values()),
            seed=seed
        )
    else:
        # Use the full graph to run BC
        if sampling_percentage:
            num_samples = get_num_samples(G_nx, sampling_percentage)

        source_target_nodes_list = get_source_target_nodes_list(G_nx, source_target_nodes)

        df = pd.DataFrame()

        if computation_mode == "exact" or computation_mode == "all":
            print("Computing exact BC scores...")
            df = utils.betweenness.betweenness_approximate_df(
                G=G_nx,
                normalized=True,
                quiet=False,
                num_samples=G_nx.number_of_nodes(),
                source_target_nodes_list=source_target_nodes_list,
                seed=seed,
                column_name='betweenness_centrality'
            )
        if computation_mode == 'approximate' or computation_mode == 'all':
            print("Computing approximate BC scores using", num_samples, "sampled nodes...")
            df_approx = utils.betweenness.betweenness_approximate_df(
                G=G_nx,
                normalized=True,
                quiet=False,
                num_samples=num_samples,
                source_target_nodes_list=source_target_nodes_list,
                seed=seed,
                column_name='approximate_betweenness_centrality'
            )

            if df.empty:
                df = df_approx
            else:
                df['approximate_betweenness_centrality'] = df_approx['approximate_betweenness_centrality']

    print(df)
    return df 
    
    
    # # Convert NetworkX graph into networkit graph
    # start = timer()
    # print('Converting NetworkX graph into a networkit graph...')
    # G = nk.nxadapter.nx2nk(G_nx)
    # print('Finished converting NetworkX graph into a networkit graph \nElapsed time:', timer()-start, 'seconds\n')

    # # Create dictionaries that map networkx IDs to networkit IDs and vice-versa
    # start = timer()
    # print('Creating + Saving mapping dictionaries for nodes between NetworkX and networkit...')
    # nx_to_nk_id_dict = dict((id, int_id) for (id, int_id) in zip(G_nx.nodes(), range(G_nx.number_of_nodes())))
    # nk_to_nx_id_dict = dict((int_id, id) for (id, int_id) in zip(G_nx.nodes(), range(G_nx.number_of_nodes())))

    # # Save the two dictionaries in the output_dir
    # with open(output_dir + 'nx_to_nk_id_dict.pickle', 'wb') as handle:
    #     pickle.dump(nx_to_nk_id_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(output_dir + 'nk_to_nx_id_dict.pickle', 'wb') as handle:
    #     pickle.dump(nk_to_nx_id_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print('Finished Creating + Saving mapping dictionaries for nodes between NetworkX and networkit\
    #     \nElapsed time:', timer()-start, 'seconds\n')

    # # Construct the dataframe placeholder
    # df = pd.DataFrame()
    # df['node'] = list(nx_to_nk_id_dict.keys())

    # # Add node type as a column
    # node_types = [G_nx.nodes[node]['type'] for node in df['node'].values]
    # df['node_type'] = node_types

    # start = timer()
    # print('Calculating betweeness scores...')

    # if computation_mode == 'all' or computation_mode == 'exact':
    #     # df['betweenness_centrality'] = utils.betweenness.betweeness_exact(G)
    #     # NOTE: Exact BC is computed by running the approximate algorithm and setting the num_samples equal to the number of nodes in the graph
    #     df['approximate_betweenness_centrality'] = utils.betweenness.betweeness_approximate(G, num_samples=G.numberOfNodes())
    # if computation_mode == 'all' or computation_mode == 'approximate':
    #     # Get the appropriate list of source_target_nodes
    #     if source_target_nodes == 'cell_nodes':
    #         cell_nodes = [x for x,y in G_nx.nodes(data=True) if y['type']=='cell']
    #         source_target_nodes_list = [nx_to_nk_id_dict[node] for node in cell_nodes]
    #     elif source_target_nodes == 'attribute_nodes':
    #         attr_nodes = [x for x,y in G_nx.nodes(data=True) if y['type']=='attr']
    #         source_target_nodes_list = [nx_to_nk_id_dict[node] for node in attr_nodes]
    #     else:
    #         source_target_nodes_list = None

    #     if (source_target_nodes_list is not None and num_samples > len(source_target_nodes_list)):
    #         raise ValueError('The number of samples cannot be less than the number of source/target nodes specified.')

    #     df['approximate_betweenness_centrality'] = utils.betweenness.betweeness_approximate(G, num_samples=num_samples, source_target_nodes_list=source_target_nodes_list, seed=seed)

    # print('Finished calculating betweeness scores \nElapsed time:', timer()-start, 'seconds\n')

    # # TODO: Add clustering coefficients for bipartite graphs (currently networkit doesn't support it for bipartite graphs)

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
            output_dir=args.output_dir,
            computation_mode=args.betweenness_mode,
            num_samples = args.num_samples,
            sampling_percentage=args.sampling_percentage,
            source_target_nodes = args.betweenness_source_target_nodes,
            seed = args.seed,
            node_compression=args.node_compression
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

    # Specifies the types of nodes used as source/target nodes when computing BC. For `all` mode, all the nodes are used, for `cell_nodes`
    # mode only cell nodes can be used as source/target nodes and similarly for attribute_nodes. 
    parser.add_argument('-bstn', '--betweenness_source_target_nodes', default='all', choices=['all', 'cell_nodes', 'attribute_nodes'],
    help='Specifies the types of nodes used as source/target nodes when computing BC. For `all` mode, all the nodes are used, for `cell_nodes`\
    mode only cell nodes can be used as source/target nodes and similarly for attribute_nodes.')

    # Seed used for the random sampling used by the approximate betweenness centrality algorithm
    parser.add_argument('--seed', metavar='seed', type=int,
    help='Seed used for the random sampling used by the approximate betweenness centrality algorithm')

    # Denotes the radius for the betweenness_in_k_neighborhood calculations. Argument is a range of min to max radius and the step.
    parser.add_argument('--betweenness_in_k_neighborhood', nargs=3, type=int, metavar=('min_radius', 'max_radius', 'step'),
    help='If specified we compute the betweenness_in_k_neighborhood in the specified radius range.')

    # If specified, then use a collapsed version of the input graph to perform the analysis
    parser.add_argument('--collapsed_graph', action='store_true',
    help='If specified, then use a collapsed version of the input graph to perform the analysis')   

    # If specified, then find set of cell nodes that co-occur with the same set of attribute nodes 
    # and compress these cell nodes into a single compressed node 
    parser.add_argument('--node_compression', action='store_true',
    help='If specified, then find set of cell nodes that co-occur with the same set of attribute nodes\
    and compress these cell nodes into a single compressed node')

    # Number of nodes to sample to approximately calculate the betweeness 
    # Used only with conjunction with betweenness_mode set to approximate. 
    parser.add_argument('--num_samples', type=int, 
    help='Number of nodes to sample to approximately calculate the betweeness \
    Used only with conjunction with betweenness_mode set to approximate.')

    # Specifies the percentage of the graph's nodes used for sampling when computing approximate BC
    parser.add_argument('--sampling_percentage', type=float,
    help="Specifies the percentage of the graph's nodes used for sampling when computing approximate BC. \
        Must be a value in the range (0, 100]")

    # Parse the arguments
    args = parser.parse_args()

    # Check for argument consistency
    if args.perform_cleaning and (not args.min_str_length and not args.remove_numerical_vals):
        parser.error('To perform you must specify at least one of min_str_length or remove_numerical_vals arguments')
    if args.num_samples and args.sampling_percentage:
        parser.error('Specify either num_samples or sampling_proportion not both.')
    if args.sampling_percentage and (args.sampling_percentage <= 0 or args.sampling_percentage > 100):
        parser.error('sampling_percentage must be in the range (0, 100]')
    if args.node_compression and not args.sampling_percentage:
        parser.error('When using node_compression, specify a sampling_percentage instead of num_samples')
    
    print('##### ----- Running network_analysis/main.py with the following parameters ----- #####\n')

    print('Output directory:', args.output_dir)
    print('Graph path:', args.graph)
    print('Ground Truth path:', args.groundtruth_path)
    print('Existing computation:', args.existing_computation)
    print('Betweenness mode:', args.betweenness_mode)
    print('Source/Target nodes used for BC computation are of type:', args.betweenness_source_target_nodes)
    if args.betweenness_in_k_neighborhood:
        print('Betweenness in k neighborhood in the range:',  args.betweenness_in_k_neighborhood[0],
        '-', args.betweenness_in_k_neighborhood[1], 'with step size:', args.betweenness_in_k_neighborhood[2])
    if args.num_samples:
        print('Number of samples:', args.num_samples)
    if args.sampling_percentage:
        print('Sampling', args.sampling_percentage, ' percent of the nodes')
    if args.collapsed_graph:
        print('Using collapsed graph for analysis')
    if args.node_compression:
        print('Using node compression for BC computation')
    print()
    if args.perform_cleaning:
        print('Cleaning is set: ON')
        if args.min_str_length:
            print('Minimum string length:', args.min_str_length)
        if args.remove_numerical_vals:
            print('Removing numerical valued nodes')
    else:
        print('Cleaning is set: OFF')
    
    if args.seed:   
        print('User specified seed:', args.seed)
        # Set the seed
        random.seed(args.seed)
    else:
        # Generate a random seed if not specified
        args.seed = random.randrange(2**32)
        random.seed(args.seed)
        print('No seed specified, picking one at random. Seed chosen is:', args.seed)
    print('\n\n')

    # Create the output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save the input arguments in the output_dir
    with open(args.output_dir + 'args.json', 'w') as fp:
        json.dump(vars(args), fp, sort_keys=True, indent=4)

    main(args)