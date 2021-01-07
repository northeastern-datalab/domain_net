import networkx as nx
import pandas as pd
import numpy as np

import pickle
import json
import argparse

from timeit import default_timer as timer
from tqdm import tqdm
from pathlib import Path

def update_info_dict(f, info_dict):
    '''
    Appends a row for every injected homograph in the `info_dict`

    `f` is path to the current `rank_delta_stats.json` file
    '''
    rank_delta_stats_dict = json.load(open(f))

    # Get metadata files
    metadata_dir = rank_delta_stats_dict['metadata_dir']
    args_file = json.load(open(metadata_dir + 'args.json'))
    injected_homographs_file = json.load(open(metadata_dir + 'injected_homographs.json'))
    replaced_values_stats_file = json.load(open(metadata_dir + 'replaced_values_stats.json'))
    num_values_replaced_per_homograph = args_file['num_values_replaced_per_homograph']

    for homograph, homograph_dict in rank_delta_stats_dict['injected_homographs'].items():
        info_dict['num_values_replaced_per_homograph'].append(num_values_replaced_per_homograph)
        info_dict['homograph_rank'].append(homograph_dict['rank'])
        info_dict['homograph_rank_percentile'].append(homograph_dict['rank_percentile'])
        info_dict['min_cardinality'].append(args_file['min_cardinality'])
        info_dict['max_cardinality'].append(args_file['max_cardinality'])
        info_dict['homograph_betweenness'].append(homograph_dict['approximate_betweenness_centrality'])

        # Get replaced value stats
        replaced_vals_ranks = []
        replaced_vals_percentile_ranks = []
        replaced_vals_num_files_with_value = []
        replaced_vals_cardinality = []
        replaced_vals_betweenness = []
        for replaced_val in injected_homographs_file[homograph]:
            replaced_vals_ranks.append(rank_delta_stats_dict['replaced_values'][replaced_val]['rank'])
            replaced_vals_percentile_ranks.append(rank_delta_stats_dict['replaced_values'][replaced_val]['rank_percentile'])
            replaced_vals_num_files_with_value.append(replaced_values_stats_file[replaced_val]['num_files_with_value'])
            replaced_vals_cardinality.append(replaced_values_stats_file[replaced_val]['cardinality'])
            replaced_vals_betweenness.append(rank_delta_stats_dict['replaced_values'][replaced_val]['approximate_betweenness_centrality'])

        info_dict['replaced_values_avg_rank'].append(np.mean(replaced_vals_ranks))
        info_dict['replaced_values_avg_rank_percentile'].append(np.mean(replaced_vals_percentile_ranks))
        info_dict['replaced_values_avg_num_files_with_value'].append(np.mean(replaced_vals_num_files_with_value))
        info_dict['replaced_values_avg_cardinality'].append(np.mean(replaced_vals_cardinality))
        info_dict['replaced_values_avg_betweenness'].append(np.mean(replaced_vals_betweenness))

def main(args):
    rank_delta_stats_file_list = []
    # Get all rank_detla_stats.json files from directory
    for i in Path(args.input_dir).glob('**/*'):
        if i.is_file() and i.name=='rank_delta_stats.json':
            rank_delta_stats_file_list.append(i.absolute())


    info_dict = {'num_values_replaced_per_homograph': [], 'homograph_rank': [],
    'homograph_rank_percentile': [], 'homograph_betweenness': [], 'replaced_values_avg_rank': [],
    'replaced_values_avg_rank_percentile': [], 'replaced_values_avg_num_files_with_value': [],
    'replaced_values_avg_cardinality': [], 'replaced_values_avg_betweenness': [],
    'min_cardinality': [], 'max_cardinality': []}

    # Access rank info for each file
    print('Populating info_dict...')
    for f in tqdm(rank_delta_stats_file_list):
        update_info_dict(f, info_dict)

    # Save info_dict as a panda dataframe
    df = pd.DataFrame(info_dict)
    save_path = args.output_dir + "injected_homograph_df.pickle"
    print(save_path)
    df.to_pickle(save_path)

if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Evaluation the rankings of the injected homographs on aggregate level')

    # Path to the directory with injected homograph evaluations with different parameters
    parser.add_argument('-id', '--input_dir', metavar='input_dir', required=True,
    help='Path to the directory with injected homograph evaluations with different parameters')

    # Output path for dataframe
    parser.add_argument('-od', '--output_dir', metavar='output_dir', required=True,
    help='Output directory for the dataframe')

    # Parse the arguments
    args = parser.parse_args()

    print('\n##### ----- Running homograph_injection_aggregate_evaluation with the following parameters ----- #####\n')

    print('Input directory:', args.input_dir)
    print('Output directory:', args.output_dir)
    print('\n\n')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)