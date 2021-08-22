import networkx as nx
import pandas as pd
import random

import argparse
import pickle
import json
import utils
import operator
import itertools
import copy

from timeit import default_timer as timer
from pathlib import Path
from tqdm import tqdm

def get_marked_nodes(df, top_perc=10.0, bottom_perc=10.0):
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

def same_types(attrs, attr_to_type):
    '''
    Given a list of `attrs` find if they all map to the same type.
    If they all map to an uninitialized value (i.e. -1) then return False.
    '''
    types = {attr_to_type[attr] for attr in attrs}
    
    if (len(types) > 1):
        return False
    else:
        # There is a single type, check if it is uninitialized (i.e. a negative number)
        if (list(types)[0] < 0):
            return False
        else:
            # Only a single type and it is initialized
            return True

def get_valid_assignments(attr_of_hom_to_type, next_available_type):
    '''
    Given a set of attribute nodes corresponding to a marked homograph,
    return the list of all valid type assignments for each attribute such that
    for each assignment there are at least two different types used

    Arguments
    -------
        attr_of_hom_to_type (dict): Maps the attribute nodes of the homograph to their assigned type 
    '''
    # Find the set of types that can be assigned to attributes that have not been assigned a type yet
    assignable_types = set()
    num_of_attrs_without_type = 0
    attrs_awaiting_assignment = []
    for attr in attr_of_hom_to_type:
        if attr_of_hom_to_type[attr] >= 0:
            assignable_types.add(attr_of_hom_to_type[attr])
        else:
            num_of_attrs_without_type += 1
            attrs_awaiting_assignment.append(attr)

    if num_of_attrs_without_type > 0:
        # Need to assign a type to at least one attribute node

        # If `num_of_attrs_without_type` is very large it may be wise to put a hard-cap to limit exponential growth of possibilities (currently set to 3) 
        num_of_attrs_without_type = min(num_of_attrs_without_type, 3)

        # Add `num_of_attrs_without_type` new types in the possible assignable types
        assignable_types.update(range(next_available_type, next_available_type+num_of_attrs_without_type))

        # Get all possible type assignments for `attrs_awaiting_assignment`
        type_assignments = list(itertools.product(assignable_types, repeat=num_of_attrs_without_type))
        
        valid_assignments_list = []
        for assignment in type_assignments:
            attr_to_type_temp_dict = copy.deepcopy(attr_of_hom_to_type)
            for i in range(len(assignment)):
                attr_to_type_temp_dict[attrs_awaiting_assignment[i]] = assignment[i]
            
            # Check if the assignment is valid (it is valid if there are at least two different types in the `attr_to_type_temp_dict`)
            if len(set(attr_to_type_temp_dict.values())) > 1:
                valid_assignments_list.append(attr_to_type_temp_dict)

        return valid_assignments_list
    else:
        # All attributes have already been assigned so just return the current assignment
        return [attr_of_hom_to_type]

def process_homograph(hom, G, attr_to_type, next_available_type):
    '''
    Arguments
    -------
        hom (string): Cell value node marked as a homograph 

        G (networkx graph): Input graph

        attr_to_type (dict): Maps each attribute node to its assigned type. Maps to -1 if it hasn't been assigned yet

        next_available_type (int): Next freely available type to assign to an attribute 
       
    Returns
    -------
    Returns the updated `attr_to_type` dictionary and `next_available_type` value
    '''
    attrs_of_hom = utils.graph_helpers.get_attribute_of_instance(G, hom)
    attr_of_hom_to_type = {attr:attr_to_type[attr] for attr in attrs_of_hom}

    # Find valid type assignments for each attribute in `attr_of_hom_to_type`
    valid_assignments = get_valid_assignments(attr_of_hom_to_type, next_available_type)

    # Sort the assignments based on the number of unique types used. Less types ranks earlier (lower index) 
    valid_assignments = sorted(valid_assignments, key=lambda k: len(set(k.values()))) 

    # Choose the first assignment in the `valid_assignments` list and update `attr_to_type` and `next_available_type`
    for attr in valid_assignments[0]:
        attr_to_type[attr] = valid_assignments[0][attr]

    next_available_type = max({attr_to_type[attr] for attr in attr_to_type}) + 1

    return attr_to_type, next_available_type

def constraint_violations_check(G, marked_homographs, marked_unambiguous_values, attr_to_type):
    '''
    Check if any of the below two constraint violations are observed

    Constraint 1: Each marked unambiguous value node must map to attribute nodes of the same type
    Constraint 2: Each marked homograph node must map to a set of attribute nodes that have at least 2 different types

    Arguments
    -------
        G (networkx graph): Input graph

        marked_homographs (list of strings): List of cell nodes marked as homographs
        
        marked_unambiguous_values (list of strings): List of cell nodes marked as unambiguous values

        attr_to_type (dict): Mapping of each attribute node to a type (set to -1 if hasn't been assigned yet) 
       
    Returns
    -------
    Nothing
    '''

    # Check for constraint 1 violations
    num_constraint_1_violations = 0
    for val in marked_unambiguous_values:
        attrs_of_val = utils.graph_helpers.get_attribute_of_instance(G, val)
        if (not same_types(attrs_of_val, attr_to_type)):
            num_constraint_1_violations += 1

    # Check for coinstint 2 violations
    num_constraint_2_violations = 0
    for hom in marked_homographs:
        attrs_of_hom = utils.graph_helpers.get_attribute_of_instance(G, hom)
        num_types = len({attr_to_type[attr] for attr in attrs_of_hom})
        if num_types < 2:
            print("Constraint-2 violated from cell node:", hom)
            num_constraint_2_violations += 1      

    print("\nThere are in total:", num_constraint_1_violations, "Constraint-1 Violations and", num_constraint_2_violations, "Constraint-2 Violations.\n")

def infer_cell_node_type(G, marked_homographs, marked_unambiguous_values, attr_to_type):
    '''
    Given the current assignment of types for the attribute nodes find cell nodes not in the
    `marked_homographs` and `marked_unambiguous_values` lists that can inferred as unambiguous or homographs

    Arguments
    -------
        G (networkx graph): Input graph

        marked_homographs (list of strings): List of cell nodes marked as homographs
        
        marked_unambiguous_values (list of strings): List of cell nodes marked as unambiguous values

        attr_to_type (dict): Mapping of each attribute node to a type (set to -1 if hasn't been assigned yet) 
       
    Returns
    -------
    Two lists
    1) The nodes inferred as homographs
    2) The nodes inferred as unambiguous values
    '''

    cell_nodes = set([n for n, d in G.nodes(data=True) if d['type']=='cell'])
    cell_nodes = set([n for n in cell_nodes if G.degree[n] > 1])

    cell_nodes_to_examine = cell_nodes - (set(marked_homographs) | set(marked_unambiguous_values))

    print("There are", len(cell_nodes_to_examine), 'cell nodes that can be examined to determine if they are homographs or not.')

    nodes_inferred_as_homographs = []
    nodes_inferred_as_unambiguous = []
    nodes_that_cannot_be_inferred = []
    for node in cell_nodes_to_examine:
        attrs_of_node = utils.graph_helpers.get_attribute_of_instance(G, node)
        types = {attr_to_type[attr] for attr in attrs_of_node}

        if -1 in types:
            # If one of the attributes is uninitialized then don't make a prediction
            nodes_that_cannot_be_inferred.append(node)
        elif len(types) == 1:
            nodes_inferred_as_unambiguous.append(node)
        else:
            nodes_inferred_as_homographs.append(node)

    print("Inferred", len(nodes_inferred_as_unambiguous), 'cell nodes as unambiguous')
    print("Inferred", len(nodes_inferred_as_homographs), 'cell nodes as homographs')
    print("There are", len(nodes_that_cannot_be_inferred), 'nodes that cannot be inferred given the current type assignments of attributes')

    return nodes_inferred_as_homographs, nodes_inferred_as_unambiguous


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
    Returns a mapping of each attribute node to its assigned type
    '''
    attr_nodes = [n for n, d in G.nodes(data=True) if d['type']=='attr']

    # Each attribute is initialized to map to node type -1 (i.e. uninitialized) 
    attr_to_type = {n: -1 for n in attr_nodes}

    next_available_type = 1     # Next available type ID to assign for a new attribute type
    
    ######----- Propagate unambiguous values -----######
    print("Propagating marked unambiguous values...")
    for val in tqdm(marked_unambiguous_values):
        attrs_of_val = utils.graph_helpers.get_attribute_of_instance(G, val)

        # Check if all attributes of the current value already have the same type
        if (same_types(attrs_of_val, attr_to_type)):
            # No need to change anything
            continue
        else:
            # Assign a type to each attribute in `attrs_of_val` if there isn't one available
            cur_max_type = max({attr_to_type[attr] for attr in attrs_of_val})
            if (cur_max_type > 0):
                # Assign every attribute in `attrs_of_val` to `cur_max_type`
                for attr in attrs_of_val:
                    attr_to_type[attr]=cur_max_type 
            else:
                # Assign every attribute in `attrs_of_val` to `next_available_type` and increment it
                for attr in attrs_of_val:
                    attr_to_type[attr]=next_available_type
                next_available_type+=1

    ######----- Propagate homographs -----######

    # Map each marked homograph to the number of attribute nodes it is connected to and sort the dictionary by value (low to high)
    marked_homograph_to_num_attrs_dict = {hom: len(utils.graph_helpers.get_attribute_of_instance(G, hom)) for hom in marked_homographs}
    marked_homograph_to_num_attrs_dict = {k: v for k, v in sorted(marked_homograph_to_num_attrs_dict.items(), key=lambda item: item[1])}
    print("Propagating marked homographs...")
    for hom in tqdm(marked_homograph_to_num_attrs_dict.keys()):
        attrs_of_hom = utils.graph_helpers.get_attribute_of_instance(G, hom)
        
        if len(attrs_of_hom) == 2:
            # Only two attributes connected to the current homograph so assign a different type to each one
            if (attr_to_type[attrs_of_hom[0]] == attr_to_type[attrs_of_hom[1]]):
                # The two attributes have the same type (change one of them, or both they are uninitialized)
                if attr_to_type[attrs_of_hom[0]] < 0:
                    # Types for both attributes are not initialized
                    attr_to_type[attrs_of_hom[0]]=next_available_type
                    next_available_type += 1
                    attr_to_type[attrs_of_hom[1]]=next_available_type
                    next_available_type += 1
                else:
                    # Types for the two attributes are the same and initialized
                    # Do nothing, but this means there will be a homograph constraint violation
                    pass
            else:
                # The two attributes have different types (if one an attribute has an uninitialized type, initialize it)
                if attr_to_type[attrs_of_hom[0]] < 0:
                    attr_to_type[attrs_of_hom[0]]=next_available_type
                    next_available_type += 1
                elif attr_to_type[attrs_of_hom[1]] < 0:
                    attr_to_type[attrs_of_hom[1]]=next_available_type
                    next_available_type += 1
                else:
                    # Do nothing, the are already of different types and initialized
                    pass
        else:
            # Current approach chooses a valid assignment with the least number of unique types (no exploration)
            # TODO: Perform some sort of random assignment of types minimizing constraint violations and number of unique types  
            attr_to_type, next_available_type = process_homograph(hom, G, attr_to_type, next_available_type)

    # Check for constraint violations
    constraint_violations_check(G, marked_homographs, marked_unambiguous_values, attr_to_type)

    # Check how many attributes have been assigned a type, and how many attributes there are in total
    print("There are", sum(1 for attr in attr_to_type if attr_to_type[attr] > 0), "attribute nodes that have been assigned a type")
    print("A total of", len(set(attr_to_type.values())), 'unique types were assigned to all attributes\n')

    # Check how many cell nodes (not in the marked_homographs and marked_unambiguous values) can be inferred as identical or homographs
    inferred_homographs, inferred_unambiguous = infer_cell_node_type(G, marked_homographs, marked_unambiguous_values, attr_to_type)

    # Evaluate the inferred results
    print("Inferred homographs precision:", get_precision(df, inferred_homographs, is_homograph=True))
    print("Inferred unambiguous precision:", get_precision(df, inferred_unambiguous, is_homograph=False))
     
    # TODO: Loop over remaining cell nodes not in `marked_homographs` and `marked_unambiguous_values` and propagate semantic types based on current information

    return attr_to_type

def get_precision(df, node_list, is_homograph=True):
    '''
    Given a list of nodes that are marked either as homographs or non-homographs find how many out of them are truly homographs or not
    '''
    df_subset = df[df['node'].isin(node_list)]
    precision = df_subset['is_homograph'].value_counts()[is_homograph] / len(node_list)
    return precision

def get_attribute_types_of_cell_node(G, cell_node, attr_to_type):
    '''
    Given a cell node, find all the attribute nodes it is connected to and return a dictionary
    keyed by the connected attribute nodes and mapping to their assigned type
    '''
    attrs_of_cell_node = utils.graph_helpers.get_attribute_of_instance(G, cell_node)
    attr_types = {attr:attr_to_type[attr] for attr in attrs_of_cell_node}
    return attr_types

def get_marked_nodes_from_file(file_path, df, G):
    '''
    Given the JSON file path that specifies what nodes are marked return
    the `marked_homographs` and the `marked_unambiguous_values`.

    If the `marked_unambiguous_values` are not specified in the JSON then select the bottom `X` percent
    nodes based on BC that are neighbors of the `marked_homographs`

    Arguments
    -------
        file_path (str): a string to the file_path that specifies the marked nodes 

        df (pandas dataframe): pandas dataframe with the BC score for each node in the graph

        G (networkx graph): The graph representation of the input dataset 

    Returns
    -------
    Returns two lists, a list of the marked homographs and a list of the 
    marked unambiguous values 
    '''

    print('\nExtracting the marked nodes from file...')

    # The selected nodes are specified by provided JSON file
    with open(file_path) as json_file:
        json_dict = json.load(json_file)

    marked_homographs = json_dict['marked_homographs']
    assert len(marked_homographs) > 0, 'There should be at least one value in the marked_homographs list'

    if ('marked_unambiguous_values' not in json_dict) or (len(json_dict['marked_unambiguous_values']) == 0):
        # Find neighboring cell nodes for each `marked_homograph`
        marked_unambiguous_values = []
        for hom in marked_homographs:
            cell_node_neighbors = utils.graph_helpers.get_cell_node_neighbors(G, hom)

            # Limit the df to only those nodes and extract the bottom `X` percent of them based on their BC scores
            df_tmp = df[df['node'].isin(cell_node_neighbors)]
            df_tmp = process_df(df_tmp, G)
            _, unambiguous_values = get_marked_nodes(df_tmp, bottom_perc=10)
            marked_unambiguous_values += unambiguous_values
    else:
        marked_unambiguous_values = json_dict['marked_unambiguous_values']

    print('Finished extracting the marked nodes from file\n')
    return marked_homographs, marked_unambiguous_values

def main(args): 
    # Load the graph file
    start = timer()
    print('Loading graph file...')
    graph = pickle.load(open(args.graph, 'rb'))
    print('Finished loading graph file \nElapsed time:', timer()-start, 'seconds')
    num_attr_nodes = sum(1 for n, d in graph.nodes(data=True) if d['type']=='attr')
    num_cell_nodes = sum(1 for n, d in graph.nodes(data=True) if d['type']=='cell')
    print("Input graph has", num_attr_nodes, 'attribute nodes and', num_cell_nodes, 'cell nodes.\n')

    # Load dataframe and filter it
    df = pickle.load(open(args.dataframe, 'rb'))
    df = process_df(df, graph)

    # Select the marked homographs and unambiguous values
    if args.input_nodes:
        marked_homographs, marked_unambiguous_values = get_marked_nodes_from_file(args.input_nodes, df, graph)
    else:
        # Get initial lists of homographs and unambiguous nodes by extacting the top and bottom nodes in the BC rankings
        marked_homographs, marked_unambiguous_values = get_marked_nodes(
            df=df, 
            top_perc=15.0,
            bottom_perc=40.0
        )

    print('For initialization:', len(marked_homographs), 'cell nodes marked as homographs and', 
        len(marked_unambiguous_values), 'cell nodes marked as unambiguous values.')
    print('Marked Homographs precision:', get_precision(df, marked_homographs, is_homograph=True))
    print('Marked Unambiguous values precision:', get_precision(df, marked_unambiguous_values, is_homograph=False), '\n')

    exit()

    # Perform the Propagation
    attr_to_type = type_propagation(df, graph, marked_homographs, marked_unambiguous_values)

    for attr in attr_to_type:
        if attr_to_type[attr] > 0:
            print(attr, attr_to_type[attr])

    # Save attr_to_type dict to the output_dir
    with open(args.output_dir + 'attr_to_type.pickle', 'wb') as handle:
        pickle.dump(attr_to_type, handle, protocol=pickle.HIGHEST_PROTOCOL)


    

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

    parser.add_argument('--input_nodes', metavar='input_nodes',
    help='Path to the JSON file that specifies the input homograph and/or unambiguous values to be used for propagation. \
    If a file is not provided then the top and bottom cell nodes ranked by their BC scores are selected as the \
    marked homographs and unambiguous values respectively.')

    # parser.add_argument('--mode', metavar='mode', choices=['single', 'multiple'], default='single',
    # help='Specifies if we want to find the number ')

    # Parse the arguments
    args = parser.parse_args()

    # Check for argument consistency
    
    print('##### ----- Running network_analysis/semantic_type_propagation.py with the following parameters ----- #####\n')

    print('Output directory:', args.output_dir)
    print('Graph path:', args.graph)
    print('DataFrame path:', args.dataframe)
    print('Input Nodes Path:', args.input_nodes)
    
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