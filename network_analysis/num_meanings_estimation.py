import networkx as nx
import pandas as pd
import numpy as np
import random

import argparse
import pickle
import json
import utils
import itertools

from timeit import default_timer as timer
from pathlib import Path
from tqdm import tqdm

import os
import glob
import subprocess

import csv

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

def get_unionability(node1, node2, all_att_unionability_df, table_unionability_df):
    '''
    Method 1: Search for all rows where (query_table=='node1' AND candidate_table=='node2') or (query_table=='node2' AND candidate_table=='node1')  

    Method 2: Extract unionability score from the table_unionability_df, if a queried pair is not found assume a unionability score of 0
    '''
    relevant_df_1 = all_att_unionability_df[(all_att_unionability_df['query_table'] == node1) & (all_att_unionability_df['candidate_table'] == node2)]
    relevant_df_2 = all_att_unionability_df[(all_att_unionability_df['query_table'] == node2) & (all_att_unionability_df['candidate_table'] == node1)]
    combined_df = relevant_df_1.append(relevant_df_2, ignore_index=True)
    
    # TODO: Figure a better way to extract a score in case of many options
    score = combined_df['score'].mean()

    # Make use of the table_unionability_df
    relevant_df_1 = table_unionability_df[(table_unionability_df['query_table'] == node1) & (table_unionability_df['candidate_table'] == node2)]
    relevant_df_2 = table_unionability_df[(table_unionability_df['query_table'] == node2) & (table_unionability_df['candidate_table'] == node1)]
    combined_df = relevant_df_1.append(relevant_df_2, ignore_index=True)
    
    if combined_df.empty:
        score = 0
    else:
        score = combined_df['score'].max()

    return score

def symmetrize(a):
    """
    Return a symmetrized version of NumPy array a.

    Values 0 are replaced by the array value at the symmetric
    position (with respect to the diagonal), i.e. if a_ij = 0,
    then the returned array a' is such that a'_ij = a_ji.

    Diagonal values are left untouched.

    a -- square NumPy array, such that a_ij = 0 or a_ji = 0, 
    for i != j.
    """
    return a + a.T - np.diag(a.diagonal())

def populate_unionability_output_dir(attrs, G, unionability_output_path):
    '''
    Populates the specified 'unionability_output_path' with a csv file for each attribute in 'attrs'.
    The csv file is populated with the cell nodes appearing in each attribute
    '''
    # Remove all existing files in the 'unionability_output_path' directory
    files = glob.glob(unionability_output_path+'*')
    for f in files:
        os.remove(f)

    # Create a csv file for each attribute
    for attr in attrs:
        column_name = G.nodes[attr]['column_name']
        vals = utils.graph_helpers.get_instances_for_attribute(G, attr)
        vals = [column_name] + vals
        
        file = open(unionability_output_path+attr, 'w')
        csv_writter = csv.writer(file)

        # writing the data into the file
        for val in vals:
            csv_writter.writerow([val])
        file.close()

def is_file_empty(file_path):
    return os.stat(file_path).st_size == 0

def get_measure(node, G, pairwise_measure='jaccard', unionability_output_path=None):
    '''
    Processes the input dataframe so that it only contains cell nodes with degree greater than 1.
    The returned dataframe also includes a `dense_rank` column with the nodes ranked by their BC scores

    Arguments
    -------
        node (str): the input node for which we want to compute pairwise measure between its connected attribute nodes 

        G (networkx graph): Input graph

        pairwise_measure (str): the measure computed between two attributes (must be one of ['jaccard', 'unionability'])
       
    Returns
    -------
    Returns 3 items:
    
    1) A dictionary that maps a pair of attributes to its score where the key is: 'attr1' + '__' + 'attr2' where attr1 < attr2 when sorted lexicographically

    2) A numpy array of the pairwise measures represented as a matrix (the matrix is square with dimensions equal to the number of attribute nodes of the specified `node`)

    3) A mapping of the index in the numpy array to the attribute node it corresponds to
    '''
    
    # Get a list of the attribute nodes connected to `node`
    attrs = sorted(utils.graph_helpers.get_attribute_of_instance(G, node))

    # Mapping of index in `attrs` list to the attribute
    idx_to_node = {i:attrs[i] for i in range(len(attrs))}

    # Get a list of the attribute pairs for which the measure needs to be computed 
    attr_pairs = list(itertools.combinations(attrs, 2))
    
    # Maps a pair to its score 
    pair_to_measure = {}

    if pairwise_measure == 'unionability':
        print('\n\n\n########### Running Unionability Procedure for node:', node, '###########')

        # Populate the files that will be used to compute TUS 
        populate_unionability_output_dir(attrs = attrs, G=G, unionability_output_path=unionability_output_path)

        # Modify the opendata.list file (with the list of new files)
        file = open('../TABLE_UNION_OUTPUT/opendata.list', 'w')
        for val in attrs:
            file.write(val+'\n')
        file.close()
        
        # Run the TUS_sem_file.py file from the shell script
        subprocess.call(['sh', './num_meanings_TUS_subprocess.sh'])

        # Read the CSV file into a dataframe (Check if the csv files are populated, if not then populate the dataframes as empty)
        if is_file_empty('../TABLE_UNION_OUTPUT/all_att_unionability.csv'):
            all_att_unionability_df = pd.DataFrame(columns=['query_table', 'candidate_table', 'score'])
        else:
            all_att_unionability_df = pd.read_csv('../TABLE_UNION_OUTPUT/all_att_unionability.csv')

        if is_file_empty('../TABLE_UNION_OUTPUT/ctable_unionability_percentile.csv'):
            table_unionability_df = pd.DataFrame(columns=['query_table', 'candidate_table', 'score'])
        else:
            table_unionability_df = pd.read_csv('../TABLE_UNION_OUTPUT/ctable_unionability_percentile.csv')

    for pair in attr_pairs:
        if pairwise_measure == 'jaccard':
            score = get_jaccard(pair[0], pair[1], G)
        elif pairwise_measure == 'unionability':
            score = get_unionability(pair[0], pair[1], all_att_unionability_df, table_unionability_df)
        key_str = pair[0] + '__' + pair[1]
        pair_to_measure[key_str] = score
    

    # Construct the pairwise measures matrix 
    pairwise_measures_matrix = np.zeros(shape=(len(attrs), len(attrs)))
    np.fill_diagonal(pairwise_measures_matrix, 1)

    for i in range(0, len(attrs)-1):
        for j in range(i, len(attrs)):
            if i==j:
                # Do nothing for the diagonal terms
                continue
            else:
                pair = idx_to_node[i] + '__' + idx_to_node[j]
                score = pair_to_measure[pair]
                pairwise_measures_matrix[i][j] = score
    # The matrix is symmetric so populate the values in the bottom left triangle
    pairwise_measures_matrix = symmetrize(pairwise_measures_matrix)

    return pair_to_measure, pairwise_measures_matrix, idx_to_node


def get_pairwise_measures(nodes, G, output_dir, pairwise_measure='jaccard', unionability_output_path=None):
    '''
    Given a list of nodes compute the pairwise_measures between the attribute nodes for each of the `nodes` specified
    '''
    node_to_measures = {}
    Path(output_dir + 'matrices/').mkdir(parents=True, exist_ok=True)

    for node in tqdm(nodes):
        node_to_measures[node], matrix, idx_to_node = get_measure(node, G, pairwise_measure, unionability_output_path=unionability_output_path)

        # Save the 'matrix' numpy array and the 'index_to_node' dictionary under the output_dir/matrices/ directory
        np.save(output_dir+'matrices/'+node+'.npy', matrix)
        with open(output_dir+'matrices/'+node+'_idx_to_node.pickle', 'wb') as handle:
            pickle.dump(idx_to_node, handle)
    
    # Save the node_to_measures as a json in the output_dir
    with open(output_dir + 'node_to_measures.json', 'w') as fp:
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
        pairwise_measure=args.pairwise_measure,
        unionability_output_path = args.unionability_output_path
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

    parser.add_argument('--unionability_output_path', help='Path to the directory where the csv files needed for the TUS computation are stored')

    # Parse the arguments
    args = parser.parse_args()

    # Check for argument consistency
    
    print('##### ----- Running network_analysis/semantic_type_propagation.py with the following parameters ----- #####\n')

    print('Output directory:', args.output_dir)
    print('Graph path:', args.graph)
    print('DataFrame path:', args.dataframe)
    print('Input Nodes Path:', args.input_nodes)
    print('Pairwise Measure:', args.pairwise_measure)

    if args.unionability_output_path:
        print('Unionability output files directory:', args.unionability_output_path)
   
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