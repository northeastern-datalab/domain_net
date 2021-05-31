import networkx as nx
import pandas as pd
import random

import argparse
import pickle
import json
import utils

from timeit import default_timer as timer
from pathlib import Path
from tqdm import tqdm

def get_initial_lists(df, top_perc=10.0, bottom_perc=10.0):
    '''
    Returns two lists. The cell values in the `top_perc` percentage ranks are marked as homographs
    and the cell values in the `bottom_perc` percentage ranks that are are marked as unambiguous values

    Arguments
    -------
        df (pandas dataframe): dataframe that includes a 'dense_rank' column with the rank for each cell node

        top_perc (float): top percentage of ranks that are marked as homographs 

        bottom (float): bottom percentage of ranks that are marked as unambiguous values
       
    Returns
    -------
    marked_homographs list, marked_unambiguous_values list
    '''
    num_unique_ranks = df['dense_rank'].nunique()
    homograph_rank_threshold = (top_perc/100) * num_unique_ranks
    unambiguous_rank_threshold = num_unique_ranks - ((bottom_perc/100) * num_unique_ranks)

    marked_homographs = df[df['dense_rank'] <= homograph_rank_threshold]['node'].tolist()
    marked_unambiguous_values = df[df['dense_rank'] >= unambiguous_rank_threshold]['node'].tolist()

    return marked_homographs, marked_unambiguous_values

def process_df(df, G):
    '''
    Processes the input dataframe so that it only contains cell nodes with degree greater than 1.
    The returned dataframe also includes a `dense_rank` column with the nodes ranked by their BC scores

    Arguments
    -------
        df (pandas dataframe): dataframe with BC for each node in the graph 

        G (networkx graph): Input graph corresponding to the dataframe

        bottom (float): bottom percentage of ranks that are marked as unambiguous values
       
    Returns
    -------
    Updated dataframe
    '''
    # Filter down to only cell nodes with degree greater than 1
    df = df[df['node_type'] == 'cell']
    cell_nodes = df['node'].tolist()
    nodes_with_degree_greater_than_1 = [n for n in cell_nodes if G.degree[n] > 1]
    df = df.loc[df['node'].isin(nodes_with_degree_greater_than_1)]
    print('There are', len(nodes_with_degree_greater_than_1), 'cell nodes with a degree greater than 1')

    # Perform dense ranking on the BC column for all remaining nodes
    df['dense_rank'] = df['betweenness_centrality'].rank(method='dense', ascending=False)
    df.sort_values(by='betweenness_centrality', ascending=False, inplace=True)

    return df

def type_propagation(df, G, marked_homographs, marked_unambiguous_values):
    '''
    Arguments
    -------
        df (pandas dataframe): dataframe with BC for each node in the graph 

        G (networkx graph): Input graph

        marked_homographs (list of strings): List of cell nodes marked as homographs
        
        marked_unambiguous_values (list of strings): List of cell nodes marked as unambiguous values
       
    Returns
    -------
    Nothing
    '''

    attr_nodes = [n for n, d in G.nodes(data=True) if d['type']=='attr']

    # Each attribute is initialized to map to node type -1 (i.e. uninitialized) 
    attr_to_type = {n: -1 for n in attr_nodes}
    
    # Propagate unambiguous values
    for val in marked_unambiguous_values:
        attrs_of_val = utils.graph_helpers.get_attribute_of_instance(G, val)

        # Check if the types all attrs are already the same
        if (False):
            pass
        else:
            for attr in attrs_of_val:
                if attr_to_type[attr] < 0:
                    pass


def main(args): 
    # Load the graph file
    start = timer()
    print('Loading graph file...')
    graph = pickle.load(open(args.graph, 'rb'))
    print('Finished loading graph file \nElapsed time:', timer()-start, 'seconds\n')

    # Load dataframe and filter it
    df = pickle.load(open(args.dataframe, 'rb'))
    df = process_df(df, graph)

    # Get initial lists of homographs and unambiguous nodes by extacting the top and bottom nodes in the BC rankings
    # TODO: allow for custom homograph and unambiguous values lists
    marked_homographs, marked_unambiguous_values = get_initial_lists(
        df=df, 
        top_perc=10.0,
        bottom_perc=40.0
    )
    print('For initialization:', len(marked_homographs), 'cell nodes marked as homographs and', 
        len(marked_unambiguous_values), 'cell nodes marked as unambiguous values.')

    # Perform the Propagation
    type_propagation(df, graph, marked_homographs, marked_unambiguous_values)


    

if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Perform semantic type propagation over a bipartite graph \
        given a list of candidate homographs and candidate unambiguous values')

    # Output directory where output files and figures are stored
    parser.add_argument('-o', '--output_dir', metavar='output_dir', required=True,
    help='Path to the output directory where output files and figures are stored. \
    Path must terminate with backslash "\\"')

    # Path to the Graph representation of the set of tables
    parser.add_argument('-g', '--graph', metavar='graph', required=True,
    help='Path to the Graph representation of the set of tables')

    # Path to the Pandas dataframe with the respective BC scores for each node in the graph
    parser.add_argument('-df', '--dataframe', metavar='dataframe', required=True,
    help='Path to the Pandas dataframe with the respective BC scores for each node in the graph')

    # Seed used for the random sampling used by the approximate betweenness centrality algorithm
    parser.add_argument('--seed', metavar='seed', type=int,
    help='Seed used for the random sampling used by the approximate betweenness centrality algorithm')

    # Parse the arguments
    args = parser.parse_args()

    # Check for argument consistency
    
    print('##### ----- Running network_analysis/semantic_type_propagation.py with the following parameters ----- #####\n')

    print('Output directory:', args.output_dir)
    print('Graph path:', args.graph)
    print('DataFrame path:', args.dataframe)
    
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