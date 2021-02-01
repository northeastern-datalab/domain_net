import networkx as nx
import pandas as pd

import pickle
import argparse

from tqdm import tqdm
from timeit import default_timer as timer

def get_attributes_of_instance(G, instance_node):
    '''
    Given a graph `G` and an `instance_node` from the graph return its corresponding set of attribute nodes
    '''
    attribute_nodes = []
    for neighbor in G[instance_node]:
        if G.nodes[neighbor]['type'] == 'attr':
            attribute_nodes.append(neighbor)
    return attribute_nodes
        
def get_instances_for_attribute(G, attribute_node):
    '''
    Given a graph `G` and an `instance_node` from the graph find its cell nodes
    '''
    instances_nodes = []
    for neighbor in G[attribute_node]:
        if G.nodes[neighbor]['type'] == 'cell':
            instances_nodes.append(neighbor)
    return instances_nodes

def get_num_meanings_of_homograph(homograph, filename_column_unionable_pairs_dict, G):
    '''
    Return the number of meanings of a given homograph
    '''

    attrs = get_attributes_of_instance(G, homograph)

    # Get filename column tuples that we test for
    filename_column_tuples = []
    for attr in attrs:
        column_name = G.nodes[attr]['column_name']
        file_name = G.nodes[attr]['filename']
        filename_column_tuples.append((file_name, column_name))
    

    # print('There are', len(attrs), 'attributes the homograph connects to')
    sets_of_unionable_vals_set = set([])
    for tup in filename_column_tuples:
        sets_of_unionable_vals_set.add(frozenset(filename_column_unionable_pairs_dict[tup]))

    return len(sets_of_unionable_vals_set)



def main(args):

    start = timer()
    print('Loading input files')
    filename_column_unionable_pairs_dict = pickle.load(open(args.unionable_pairs_dict, 'rb'))
    G = pickle.load(open(args.graph, 'rb'))
    df = pickle.load(open(args.dataframe, 'rb'))
    print('Finished loading input files \nElapsed time:', timer()-start, 'seconds\n')

    # Get list of homographs
    homographs_list = df[df['is_homograph'] == True]['node'].values

    # Dictionary of each homograph to its respective number of meanings
    homograph_to_num_meanings_dict = {}
    print('Identifying number of meanings for each homograph...')
    for homograph in tqdm(homographs_list):
        homograph_to_num_meanings_dict[homograph] = get_num_meanings_of_homograph(homograph, filename_column_unionable_pairs_dict, G)
    
    print('\nSaving homograph to number of meanings dictionary...')
    with open(args.output_dir+'homograph_to_num_meanings_dict.pickle', 'wb') as handle:
        pickle.dump(homograph_to_num_meanings_dict, handle)

    print(homograph_to_num_meanings_dict)


if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Find the number of meanings for homographs in the TUS benchmark based on groundtruth')

    # Path to the input graph
    parser.add_argument('-g', '--graph', metavar='graph', required=True,
    help='Path to the input graph')

    # Path to the filename_column_unionable_pairs dictionary
    parser.add_argument('--unionable_pairs_dict', metavar='unionable_pairs_dict', required=True,
    help='Path to the filename_column_unionable_pairs dictionary')

    # Path to the graph_stats_with_groundtruth_df dataframe
    parser.add_argument('-df', '--dataframe', metavar='df', required=True,
    help='Path to the graph_stats_with_groundtruth_df dataframe')

    # Output directory for the homograph to number of meanings dictionary
    parser.add_argument('-od', '--output_dir', metavar='df', required=True,
    help='Output directory for the homograph to number of meanings dictionary')

    # Parse the arguments
    args = parser.parse_args()

    print('##### ----- num_meanings.py with the following parameters ----- #####\n')

    print('Graph path:', args.graph)
    print('Unionable pairs dictionary path:', args.unionable_pairs_dict)
    print('Graph stats with groundtruth dataframe path:', args.dataframe)
    print('Output directory:', args.output_dir)
    print('\n\n')

    # graph_path = '../EMBEDDINGS/combined_graphs_output/csvfiles_small_clean/bipartite/bipartite.graph'
    # filename_column_unionable_pairs_dict_path = 'output/csvfiles_small/filename_column_tuple_to_unionable_pairs_dict.pickle'
    # df_path = 'output/csvfiles_small/graph_stats_with_groundtruth_df.pickle'

    # start = timer()
    # print('Loading input files')
    # filename_column_unionable_pairs_dict = pickle.load(open(filename_column_unionable_pairs_dict_path, 'rb'))
    # G = pickle.load(open(graph_path, 'rb'))
    # df = pickle.load(open(df_path, 'rb'))
    # print('Finished loading input files \nElapsed time:', timer()-start, 'seconds\n')

    # # Convert to dictionary of sets
    # filename_column_unionable_pairs_dict_set = {}
    # for key, val in filename_column_unionable_pairs_dict.items():
    #     filename_column_unionable_pairs_dict_set[key] = set(val)


    main(args)