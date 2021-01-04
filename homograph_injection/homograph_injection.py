'''
The following file injects artificial homographs into a repository of tables
by replacing 2 or more different strings with a single different string (i.e. the injected homograph) 
'''

import pandas as pd
import numpy as np
import networkx as nx
import random

import os
import sys

import argparse
import json
import pickle

from timeit import default_timer as timer
from tqdm import tqdm
from pathlib import Path

import utils

def select_values(num_values_per_homograph, selected_values, value_stats_dict):
    '''
    Select `num_values_per_homograph` that have different header column name sets from
    each other. The selected values must not be in already present in the `selected_values` list
    '''
    selected_values_for_cur_homograph = []
    column_headers_set_list = [] # This list will be populated with sets of column headers present in the selected_values
    start = timer()
    while len(selected_values_for_cur_homograph) < num_values_per_homograph:
        elapsed_time = timer() - start
        if (elapsed_time > 3):
            # This happens rarely but it is possible that we are stuck in an infinite loop with no more valid
            # values to select. In that case just re-select from values we already know  
            print('Cannot find more valid values for replacement...\nSelect at random from unused values in value_stats_dict')
            unused_vals = set(list(value_stats_dict.keys())) - set(selected_values)
            unused_vals = list(unused_vals)
            for i in range(num_values_per_homograph):
                selected_values_for_cur_homograph.append(unused_vals[i])
            break

        cur_val = random.choice(list(value_stats_dict.keys()))
        cur_val_column_headers = value_stats_dict[cur_val]['column_names']
        cur_val_has_distinct_column_headers = True

        # Ensure that the set of column headers for cur_val is distinct from each other selected values
        for header_set in column_headers_set_list:
            if cur_val_column_headers == header_set:
                # Skip the cur_val as it has the exact column headers with other column
                cur_val_has_distinct_column_headers = False
                break
        if cur_val_has_distinct_column_headers and cur_val not in selected_values:        
            selected_values_for_cur_homograph.append(cur_val)
            column_headers_set_list.append(cur_val_column_headers)

    return selected_values_for_cur_homograph

def assign_values_for_homograph(num_homographs, num_values_per_homograph, input_dir, g, value_stats_dict, filter_dict):
    '''
    Assigns values for each homograph and returns them in a dictionary

    Arguments
    -------
        num_homographs (int): the number of injected homographs

        num_values_per_homograph (int): the number of values injected for each homograph

        input_dir (str): path to the input directory from which the values are selected

        g (networkx graph): graph representation of the input directory

        value_stats_dict (dict): dictionary mapping each value to its respective statistical measures

        filter_dict (dict): dictionary mapping value filtering arguments to their specified values
       
    Returns 
    -------
        - homograph_to_values_dict (dict): a dictionary keyed by the injected homographs and mapping to the values replaced
    '''

    # Perform filtering by removing values from the value_stats_dict
    if filter_dict:
        value_stats_dict = utils.filter.filter_values(value_stats_dict, filter_dict)

    # Create a list of the names of the injected homographs
    injected_homographs = ['InjectedHomograph' + str(i) for i in range(1, num_homographs + 1)]
    
    # Dictionary of each homograph to the list of values it replaces 
    homograph_to_values_dict = {}

    # Select the values to be replaced
    print('\nSelecting values for replacement...')
    selected_values = []
    for injected_homograph in tqdm(injected_homographs):
        selected_values_for_cur_homograph = select_values(num_values_per_homograph, selected_values, value_stats_dict)
        homograph_to_values_dict[injected_homograph] = selected_values_for_cur_homograph
        for val in selected_values_for_cur_homograph:
            selected_values.append(val)
    print('There are', len(set(selected_values)), 'unique values selected for replacement')
    print('Finished selecting values for replacement\n')

    return homograph_to_values_dict

def inject_homographs(homograph_to_values_dict, value_stats_dict, input_dir, output_dir, metadata_output_dir):
    '''
    Given a homograph to values dictionary, replace all values in the input directory with
    the specified homograph and write them into the output directory.

    Arguments
    -------
        homograph_to_values_dict (dict): A dictionary keyed by the injected homographs mapping
        to the list of values to be replaced by it.

        value_stats_dict (dict): a dictionary keyed by cell value and maps to stat measures of that value

        input_dir (str): path to the input directory

        output_dir (str): path to the output directory where the homograph injected files are saved

        metadata_output_dir (str): path to the output directory for the metadata
       
    Returns
    -------
    Nothing
    '''

    # Dictionary to keep track of statistics for each value replaced
    replaced_values_stats = {}

    # Get a sorted list of files to scan over and inject the specified homograph
    input_files_list = sorted(os.listdir(input_dir))

    # Loop through all the files, replace the values accordingly and write them as csv files into the output_dir
    print('Injecting homographs ...')
    input_files_progress_bar = tqdm(input_files_list)
    start = timer()
    for filename in input_files_progress_bar:
        input_files_progress_bar.set_description('Processing file: ' + filename)
        df = pd.read_csv(input_dir + filename)

        # Convert to numpy array to get statistics info for replaced values
        df_as_numpy_arr = df.to_numpy()
        for homograph in homograph_to_values_dict:
            for value in homograph_to_values_dict[homograph]:
                count = np.count_nonzero(df_as_numpy_arr == value)
                if count > 0:
                    if value not in replaced_values_stats:
                        replaced_values_stats[value] = {'count': 0, 'num_files_with_value': 0, 'replaced_with': homograph,
                                                        'cardinality': value_stats_dict[value]['cardinality']}
                    replaced_values_stats[value]['count'] += count
                    replaced_values_stats[value]['num_files_with_value'] += 1

            # Replace the values accordingly
            df.replace(to_replace=homograph_to_values_dict[homograph], value=homograph, inplace=True)
        
        # Save updated dataframe to file
        df.to_csv(output_dir + filename, index=False)
    print('Finished injecting homographs \nElapsed time:', timer()-start, 'seconds\n')

    print('Saving homograph to values mapping and statistics of replaced values in', metadata_output_dir, 'directory')    
    # Save the replaced_values_stats dictionary as a json in the metadata folder
    with open(metadata_output_dir + 'replaced_values_stats.json', 'w') as fp:
        json.dump(replaced_values_stats, fp, indent=4)

    # Save the homograph_to_values_dict dictionary as a json in the metadata folder
    with open(metadata_output_dir + 'injected_homographs.json', 'w') as fp:
        json.dump(homograph_to_values_dict, fp, sort_keys=True, indent=4)

def main(args):
    print('Reading input graph...')
    # Load the graph representation of the input directory
    g = pickle.load(open(args.graph, 'rb'))
    print('Input graph has', g.number_of_nodes(), 'nodes and', g.number_of_edges(), 'edges\n')

    # Get statistics for each value in the input directory
    if not args.value_stats_dict:
        print('No value_stats_dict provided, building it now...')
        value_stats_dict = utils.graph_helpers.get_per_value_stats_from_graph(g)
        # Save value_stats_dict in root folder
        with open('value_stats_dict.pickle', 'wb') as f:
            pickle.dump(value_stats_dict, f)
    else:
        print('Loading value_stats_dict from file value_stats_dict.pickle\n')
        # Load the value stats_dict from file
        value_stats_dict = pickle.load(open(args.value_stats_dict, 'rb'))

    if args.injection_mode == 'manual':
        # Load the homographs_file that specifies what values will be replaced with a specified injected homograph
        with open(args.homographs_file) as json_file:
            homograph_to_values_dict = json.load(json_file)
    elif args.injection_mode == 'random':
        # Initialize the filter_dict, if appropriate 
        filter_dict = utils.filter.build_filter_dict(
            filter=args.filter,
            min_str_length=args.min_str_length,
            max_str_length=args.max_str_length,
            min_cardinality=args.min_cardinality,
            max_cardinality=args.max_cardinality,
            remove_numerical_vals=args.remove_numerical_vals
        )

        # Assign the values to replace for each injected homograph
        homograph_to_values_dict = assign_values_for_homograph(
            num_homographs=args.num_injected_homographs,
            num_values_per_homograph=args.num_values_replaced_per_homograph,
            input_dir=args.input_dir,
            g=g,
            value_stats_dict = value_stats_dict,
            filter_dict = filter_dict
        )
    
    # Inject the homographs by replacing values as specified in homograph_to_values_dict
    inject_homographs(
        homograph_to_values_dict=homograph_to_values_dict,
        value_stats_dict=value_stats_dict,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        metadata_output_dir=args.metadata_output_dir
    )

if __name__ == "__main__":

    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Artificially inject homographs')

    # Input directory where input files are stored
    parser.add_argument('-id', '--input_dir', metavar='input_dir', required=True,
    help='Directory of input csv files that will be used to inject the homographs. \
    Path must terminate with backslash "\\"')

    # Output directory where the homograph injected files are stored
    parser.add_argument('-od', '--output_dir', metavar='output_dir', required=True,
    help='Directory of where the output csv files will be stored that have the injected homographs included. \
    Path must terminate with backslash "\\"')

    # Path to the bipartite graph representation of data specified in the input directory
    parser.add_argument('-g', '--graph', metavar='graph', required=True,
    help='Path to the bipartite graph representation of data specified in the input directory')

    # Path to the value statistics from the graph dictionary, if not specified it will have to be built and saved in the root directory
    parser.add_argument('--value_stats_dict', metavar='value_stats_dict',
    help='Path to the value statistics from the graph dictionary,\
    if not specified it will have to be built and saved in the root directory')

    # Denotes if we perform filtering over the values selected to be replaced
    parser.add_argument('--filter', action='store_true', 
    help='Denotes if we perform filtering over the values selected to be replaced')

    # The minimum length of a string for a value to be chosen for replacement, smaller length strings are not considered
    parser.add_argument('--min_str_length', type=int,
    help='The minimum length of a string for a value to be chosen for replacement, smaller length strings are not considered')

    # The maximum length of a string for a value to be chosen for replacement, longer length strings are not considered
    parser.add_argument('--max_str_length', type=int,
    help='The maximum length of a string for a value to be chosen for replacement, longer length strings are not considered')

    # The minimum cardinality of a string value to be chosen for replacement
    parser.add_argument('--min_cardinality', type=int,
    help='The minimum cardinality of a string value to be chosen for replacement')

    # The maximum cardinality of a string value to be chosen for replacement
    parser.add_argument('--max_cardinality', type=int,
    help='The maximum cardinality of a string value to be chosen for replacement')

    # If specified removes all nodes from the graph with numerical values in their name
    parser.add_argument('--remove_numerical_vals', action='store_true', 
    help='If specified removes all nodes from the graph with numerical values in their name')

    # Output directory where the metadata of the injected homographs are stored
    parser.add_argument('-mod', '--metadata_output_dir', metavar='--metadata_output_dir',
    help='Output directory where the metadata of the injected homographs are stored')

    parser.add_argument('-im', '--injection_mode', choices=['manual', 'random'], metavar='injection_mode', required=True,
    help='Specifies the mode that injected homographs are inserted in the data. ')

    # json file that specifies the list of values that will be replaced by the specified injected homograph value
    parser.add_argument('-hf', '--homographs_file', metavar='homographs_file',
    help='json file that specifies the list of values that will be replaced by the specified injected homograph value. \
    To be used only when --injection_mode is set to manual.')

    # Number of homographs injected into the dataset
    parser.add_argument('-nih', '--num_injected_homographs', metavar='num_injected_homographs', type=int,
    help='Number of homographs injected into the dataset')

    # Number of values replaced for each injected homograph
    parser.add_argument('-nvr', '--num_values_replaced_per_homograph', metavar='num_values_replaced_per_homograph',
    help='Number of values replaced for each injected homograph', type=int)

    # Seed used for random selection of values to replace with injected homograph
    parser.add_argument('--seed', metavar='seed', type=int,
    help='Seed used for random selection of values to replace with injected homograph')

    # Parse the arguments
    args = parser.parse_args()

    # Check for argument consistency
    if args.injection_mode == 'manual' and args.homographs_file is None:
        parser.error('Must specify a homographs file for manual injection mode.')
    if args.injection_mode == 'random' and (args.num_injected_homographs is None 
    or args.num_values_replaced_per_homograph is None):
        parser.error('Must specify num_injected_homographs and num_values_replaced_per_homograph for random injection mode.')
    if args.injection_mode == 'random' and args.num_values_replaced_per_homograph <= 1:
        parser.error('At least 2 values must be replaced for each homograph, otherwise the injected homograph is \
        not really a homograph.')

    if not args.metadata_output_dir:
        # If metadata output dir is not specified then set it to be 
        # the same name as the innermost directory of output_dir
        args.metadata_output_dir = 'metadata/' + os.path.basename(os.path.dirname(args.output_dir)) + '/'

    print('\n##### ----- Running homograph_injection.py with the following parameters ----- #####\n')

    print('Input directory: ' + args.input_dir)
    print('Output directory: ' + args.output_dir)
    print('Input graph path:', args.graph)
    print('Metadata output directory:', args.metadata_output_dir)
    print('Injection mode: ' + args.injection_mode)

    if args.value_stats_dict:
        print('Value stats dict:', args.value_stats_dict)
    if args.homographs_file:
        print('Homographs file: ' + args.homographs_file)
    if args.num_injected_homographs and args.num_values_replaced_per_homograph:
        print('Number of injected homographs:', args.num_injected_homographs)
        print('Number of values replaced per homograph:', args.num_values_replaced_per_homograph)
    if args.seed:
        print('User specified seed:', args.seed)
        # Set the seed
        random.seed(args.seed)
    else:
        # Generate a random seed if not specified
        args.seed = random.randrange(sys.maxsize)
        random.seed(args.seed)
        print('No seed specified, picking one at random. Seed chosen is:', args.seed)
    print()
    if args.filter:
        print('Filtering is set: ON')
        if args.min_str_length != None:
            print('Minimum string length:', args.min_str_length)
        if args.max_str_length != None:
            print('Maximum string length:', args.max_str_length)
        if args.min_cardinality != None:
            print('Minimum cardinality:', args.min_cardinality)
        if args.max_cardinality != None:
            print('Maximum cardinality:', args.max_cardinality)
        if args.remove_numerical_vals != None:
            print('Removing numerical valued nodes')
    else:
        print('Filtering is set: OFF')
    print('\n\n')

    # Create the output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Create metadata directory if it doesn't exist
    Path(args.metadata_output_dir).mkdir(parents=True, exist_ok=True)

    # Save the input arguments in the metadata folder
    with open(args.metadata_output_dir + 'args.json', 'w') as fp:
        json.dump(vars(args), fp, sort_keys=True, indent=4)

    main(args)