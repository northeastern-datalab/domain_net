import pandas as pd
import pickle

from collections import Counter

from tqdm import tqdm
from timeit import default_timer as timer

def get_filename_column_tuple_to_unionable_pairs_dict(G, groundtruth):
    '''
    Given the groundtruth get for each (filename, column_name) get a set
    of (filename, column_name) tuples that are unionable with the key.
    
    Arguments
    -------
    G (Networkx graph): A Networkx graph for which the `groundtruth` corresponds to

    groundtruth (dict): Dictionary of (key: filename, value: dictionary_per_file)
    dictionary_per_file is a dictionary of (key: another_filename, list_of_column_pairs)
    Each column_pair in list_of_column_pairs is a pair/tuple of columns from two tables that are unionable

    Returns
    -------
    Dictionary of (filename, column_name) tuple to the set of (filename, column_name) tuples that it is unionable with
    '''
    start = timer()
    print('Constructing filename_column_tuple_to_unionable_pairs_dict...')
    
    filename_column_tuple_to_unionable_pairs_dict = {}

    # Loop through all files in the groundtruth
    for filename in tqdm(groundtruth):
        for other_filename in groundtruth[filename]:
            for column_pair in groundtruth[filename][other_filename]:

                # Check if (filename, column_pair[0]) is in the dictionary (if not initialize it)
                if (filename, column_pair[0]) not in filename_column_tuple_to_unionable_pairs_dict:
                    filename_column_tuple_to_unionable_pairs_dict[((filename, column_pair[0]))] = set()
                else:
                    filename_column_tuple_to_unionable_pairs_dict[((filename, column_pair[0]))].add((other_filename, column_pair[1]))

    print('Finished Constructing filename_column_tuple_to_unionable_pairs_dict\
        \nElapsed time:', timer()-start, 'seconds\n')

    return filename_column_tuple_to_unionable_pairs_dict

def decide_if_homograph_for_node(node, G, filename_column_tuple_to_unionable_pairs_dict):
    '''
    Returns True or False if the specified `node` is a homograph or not based on the 
    `filename_column_tuple_to_unionable_pairs_dict`, and True or False if there
    was a (filename, column_name) tuple that was not present in the 
    `filename_column_tuple_to_unionable_pairs_dict`.

    The node must be a cell node and not an attribute node
    '''

    # Ensure node is a cell node
    assert G.nodes[node]['type'] == 'cell', 'The node must be of type cell.'

    # Find all (filename, column_name) tuples associated with the specified node.
    # This is the same as finding all the attribute nodes connected to a cell value node in a bipartite graph
    neighbors = list(G.neighbors(node))

    # Set to true if we encounter at least one filename_column_name_tuple that isn't in the filename_column_tuple_to_unionable_pairs_dict
    missing_key = False

    # If there is only one neighbor we automatically say it is an identical value
    if len(neighbors) == 1:
        return False, missing_key
    
    filename_column_name_tuples = []
    for neighbor in neighbors:
        filename_column_name_tuple = (G.nodes[neighbor]['filename'], G.nodes[neighbor]['column_name'])
        filename_column_name_tuples.append(filename_column_name_tuple)

    # Filter the filename_column_name_tuples list so that all elements are keys in 
    # the filename_column_tuple_to_unionable_pairs_dict dictionary
    filename_column_name_tuples_filtered = [tup for tup in filename_column_name_tuples if tup in filename_column_tuple_to_unionable_pairs_dict]

    if len(filename_column_name_tuples_filtered) < len(filename_column_name_tuples):
        missing_key = True

    # A node is an identical value if the sets corresponding to all filename_column_name_tuples_filtered
    # are identical. Otherwise it is a homograph
    for filename_column_name_tuple in filename_column_name_tuples_filtered:
        # Ensure that all (filename, column_name) tuples are keys in filename_column_tuple_to_unionable_pairs_dict
        assert filename_column_name_tuple in filename_column_tuple_to_unionable_pairs_dict
        assert filename_column_name_tuples_filtered[0] in filename_column_tuple_to_unionable_pairs_dict

        # Compare the first set to all the others
        if filename_column_tuple_to_unionable_pairs_dict[filename_column_name_tuples_filtered[0]] !=\
        filename_column_tuple_to_unionable_pairs_dict[filename_column_name_tuple]:
            return True, missing_key
    
    # It is an identical value if all comparisons of the sets were identical
    return False, missing_key

def get_homographs_from_groundtruth(G, groundtruth, output_dir):
    '''
    Given the input graph and the groundtruth provided by the benchmark identify which values are homographs
    and which aren't based on if they share the same unionable pairs.

    Returns
    -------
    Pandas Dataframe with a row for each node specifying if that node is a homograph and if it was associated with
    a (filename, column_name) tuple that was missing from the ground truth
    '''
    # Construct a dictionary of (filename, column_name) tuple to the set of (filename, column_name) tuples that it is unionable with
    filename_column_tuple_to_unionable_pairs_dict = get_filename_column_tuple_to_unionable_pairs_dict(
        G = G,
        groundtruth = groundtruth
    )

    # Save the filename_column_tuple_to_unionable_pairs_dict dictionary
    with open(output_dir + 'filename_column_tuple_to_unionable_pairs_dict.pickle', 'wb') as handle:
        pickle.dump(filename_column_tuple_to_unionable_pairs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Determine if homograph or not for each cell node in the graph
    cell_nodes = [n for n, d in G.nodes(data=True) if d['type']=='cell']

    node_is_homograph_dict = {}
    node_has_missing_key = {}
    start = timer()
    print('Identifying homographs from all cell values...')
    for node in tqdm(cell_nodes):
        node_is_homograph_dict[node], node_has_missing_key[node] = decide_if_homograph_for_node(node, G, filename_column_tuple_to_unionable_pairs_dict)
    print('Finished Identifying homographs from all cell values \nElapsed time:', timer()-start, 'seconds\n')

    homograph_and_missing_key = []
    homograph_and_no_missing_key = []
    identical_and_missing_Key = []
    identical_and_no_missing_key = []
    for node in cell_nodes:
        if node_is_homograph_dict[node] and node_has_missing_key[node]:
            homograph_and_missing_key.append(node)
        elif node_is_homograph_dict[node] and not node_has_missing_key[node]:
            homograph_and_no_missing_key.append(node)
        elif not node_is_homograph_dict[node] and node_has_missing_key[node]:
            identical_and_missing_Key.append(node)
        elif not node_is_homograph_dict[node] and not node_has_missing_key[node]:
            identical_and_no_missing_key.append(node)

    print('There are:', len(homograph_and_missing_key), 'homographs with missing key')
    print('There are:', len(homograph_and_no_missing_key), 'homographs without missing key')
    print('There are:', len(identical_and_missing_Key), 'identical words with missing key')
    print('There are:', len(identical_and_no_missing_key), 'identical words without missing key')

    df = pd.DataFrame(columns=['node', 'is_homograph', 'has_missing_key'])
    df['node'] = node_is_homograph_dict.keys()
    df['is_homograph'] = df['node'].map(node_is_homograph_dict)
    df['has_missing_key'] = df['node'].map(node_has_missing_key)
    
    # Save the dataframe to file
    df.to_pickle(output_dir + 'node_is_homograph_df.pickle')

    return df


# Groundtruth based on column name helper functions. Such as in the Synthetic Example
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

def classify_nodes_from_column_name(g):
    '''
    Returns a dictionary keyed by each cell value and mapping to either 'homograph' or 'unambiguous value'

    Cell value nodes with degree 1 are automatically classified as unambiguous values
    '''
    cell_value_to_class_dict = {}

    for node in g.nodes:
        if g.nodes[node]['type'] == 'cell':
            if g.degree[node] > 1:
                # Get the set of attr nodes connected to node
                attribute_nodes = get_attributes_of_instance(g, node)

                # If all attribute nodes have the same column_name then `node` is an unambiguous value
                # Otherwise it is a homograph
                is_homograph = False
                set_of_column_names = set()
                for attribute_node in attribute_nodes:
                    set_of_column_names.add(g.nodes[attribute_node]['column_name'])
                    if len(set_of_column_names) > 1:
                        is_homograph = True
                        break
                
                if is_homograph:
                    cell_value_to_class_dict[node] = 'homograph'
                else:
                    cell_value_to_class_dict[node] = 'unambiguous value'

            else:
                cell_value_to_class_dict[node] = 'unambiguous value'
    return cell_value_to_class_dict