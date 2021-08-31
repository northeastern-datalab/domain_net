import networkx as nx
import pandas as pd
import random

import argparse
import pickle
import json
import utils
import itertools

from timeit import default_timer as timer
from pathlib import Path
from tqdm import tqdm

def process_df(df, G):
    '''
    Processes the input dataframe so that it only contains cell nodes with degree greater than 1.
    The returned dataframe also includes a `dense_rank` column with the nodes ranked by their BC scores

    Arguments
    -------
        df (pandas dataframe): dataframe with BC for each node in the graph 

        G (networkx graph): Input graph corresponding to the dataframe
       
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

    num_unique_ranks = df['dense_rank'].nunique()
    print('There are', num_unique_ranks, 'unique ranks based on BC.')

    return df

def get_marked_nodes_from_file(file_path):
    '''
    Given the JSON file path that specifies what nodes are marked return
    them in a list
    '''

    # The selected nodes are specified by provided JSON file
    with open(file_path) as json_file:
        json_dict = json.load(json_file)

        # Ensure single quotes are properly converted from escaped json string to python string
        parsed_input_nodes_list = []
        for val in json_dict["input_nodes"]:
            parsed_input_nodes_list.append(val.replace("\\'", "\'"))
        json_dict["input_nodes"] = parsed_input_nodes_list

    return json_dict["input_nodes"]

def get_jaccard(node1, node2, G):
    node1_nodes = set(utils.get_instances_for_attribute(G, node1))
    node2_nodes = set(utils.get_instances_for_attribute(G, node2))

    intersection = node1_nodes & node2_nodes
    union = node1_nodes | node2_nodes
    return len(intersection)/len(union)


def get_measure(node, G, pairwise_measure='jaccard'):
    '''
    Returns a dictionary that maps a pair of attributes to its score.
    The key is: 'attr1' + '__' + 'attr2' where attr1 < attr2 when sorted lexicographically
    '''
    
    # Get a list of the attribute nodes connected to `node`
    attrs = sorted(utils.graph_helpers.get_attribute_of_instance(G, node))

    # Get a list of the attribute pairs for which the measure needs to be computed 
    attr_pairs = list(itertools.combinations(attrs, 2))
    
    # Maps a pair to its score 
    pair_to_measure = {}

    for pair in attr_pairs:
        if pairwise_measure == 'jaccard':
            score = get_jaccard(pair[0], pair[1], G)
            key_str = pair[0] + '__' + pair[1]
            pair_to_measure[key_str] = score
    
    return pair_to_measure


def get_pairwise_measures(nodes, G, output_dir, pairwise_measure='jaccard'):
    '''
    Given a list of nodes compute the pairwise_measures between the attribute nodes for each of the `nodes` specified
    '''
    node_to_measures = {}

    for node in nodes:
        node_to_measures[node] = get_measure(node, G, pairwise_measure)
    
    # Save the node_to_measures as a json in the output_dir
    with open(args.output_dir + 'node_to_measures.json', 'w') as fp:
        json.dump(node_to_measures, fp, sort_keys=True, indent=4)


def main(args):
    # Load the graph file
    print('Loading graph file...')
    graph = pickle.load(open(args.graph, 'rb'))
    print('Finished loading graph file')
    num_attr_nodes = sum(1 for n, d in graph.nodes(data=True) if d['type']=='attr')
    num_cell_nodes = sum(1 for n, d in graph.nodes(data=True) if d['type']=='cell')
    print("Input graph has", num_attr_nodes, 'attribute nodes and', num_cell_nodes, 'cell nodes.\n')

    # Load dataframe and filter it
    df = pickle.load(open(args.dataframe, 'rb'))
    df = process_df(df, graph)

    # Load the input nodes
    input_nodes = get_marked_nodes_from_file(file_path=args.input_nodes)

    # Compute the pairwise measure
    get_pairwise_measures(
        nodes=input_nodes,
        G=graph,
        output_dir=args.output_dir,
        pairwise_measure=args.pairwise_measure
    )

if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Estimate the number of meanings of a value \
    given a list of candidate homographs')

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

    parser.add_argument('--input_nodes', metavar='input_nodes',
    help='Path to the JSON file that specifies the input nodes')

    parser.add_argument('--pairwise_measure', choices=['jaccard', 'unionability'], default='jaccard',
    help='The pairwise measure used to compare two columns')

    # Parse the arguments
    args = parser.parse_args()

    # Check for argument consistency
    
    print('##### ----- Running network_analysis/semantic_type_propagation.py with the following parameters ----- #####\n')

    print('Output directory:', args.output_dir)
    print('Graph path:', args.graph)
    print('DataFrame path:', args.dataframe)
    print('Input Nodes Path:', args.input_nodes)
    print('Pairwise Measure:', args.pairwise_measure)
   
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