import networkx as nx
import pandas as pd
import numpy as np

import pickle
import json
import argparse

from timeit import default_timer as timer
from tqdm import tqdm
from pathlib import Path

def get_injected_homograph_rank_stats(df, df_injected, metadata_dir, output_dir):
    '''
    Compute the average rank difference between the injected homograph and the values it replaced
    '''
    # Dictionary to be populated with statistical measures from the rank analysis 
    stats = {'injected_homographs': {}, 'replaced_values': {}}

    with open(metadata_dir+'injected_homographs.json') as json_file: 
        injected_homographs_dict = json.load(json_file)

    stats['metadata_dir'] = metadata_dir

    # Calculate the rank delta i.e. (homograph_rank - replaced_val)
    rank_deltas = []
    rank_percentile_deltas = []

    print('Calculating statistics for each injected homograph...')
    for homograph in tqdm(injected_homographs_dict):
        if homograph in df_injected['node'].to_numpy():
            homograph_rank = df_injected.loc[df_injected['node'] == str(homograph)]['rank'].values[0]
            homograph_rank_percentile = df_injected.loc[df_injected['node'] == str(homograph)]['rank_percentile'].values[0]
            homograph_betweenness_score = df_injected.loc[df_injected['node'] == str(homograph)]['approximate_betweenness_centrality'].values[0]
            stats['injected_homographs'][homograph] = {
                'rank': homograph_rank,
                'rank_percentile': homograph_rank_percentile,
                'approximate_betweenness_centrality': homograph_betweenness_score
            }
            for replaced_val in injected_homographs_dict[homograph]:
                original_rank = df.loc[df['node'] == str(replaced_val)]['rank'].values[0]
                original_rank_percentile = df.loc[df['node'] == str(replaced_val)]['rank_percentile'].values[0]
                original_betweenness_score = df.loc[df['node'] == str(replaced_val)]['approximate_betweenness_centrality'].values[0]
                stats['replaced_values'][replaced_val] = {
                    'rank': original_rank,
                    'rank_percentile': original_rank_percentile,
                    'approximate_betweenness_centrality': original_betweenness_score
                }
                rank_deltas.append(homograph_rank - original_rank)
                rank_percentile_deltas.append(homograph_rank_percentile - original_rank_percentile)
    
    stats['average_rank_delta'] = np.average(rank_deltas)
    stats['average_rank_delta_normalized'] = stats['average_rank_delta']/df_injected['rank'].count()
    stats['average_rank_percentile_delta'] = np.average(rank_percentile_deltas)
    stats['average_rank_of_replaced_values'] = np.average([val['rank'] for val in stats['replaced_values'].values()])
    stats['average_rank_of_injected_homographs'] = np.average([val['rank'] for val in stats['injected_homographs'].values()])
    stats['average_rank_percentile_of_replaced_values'] = np.average([val['rank_percentile'] for val in stats['replaced_values'].values()])
    stats['average_rank_percentile_of_injected_homographs'] = np.average([val['rank_percentile'] for val in stats['injected_homographs'].values()])
    print('\nFinished calculating statistics for each injected homograph')

    # Save the the stats into a json file
    with open(output_dir + 'rank_delta_stats.json', 'w') as fp:
        json.dump(stats, fp, indent=4)

def main(args):
    # Read original and injected betweeness scores dataframes
    df = pickle.load(open(args.betweenness_scores_df, 'rb'))
    df_injected = pickle.load(open(args.betweenness_scores_df_injected, 'rb'))

    # Remove all attribute nodes from the dataframes. We only want to analyze nodes of type cell
    df = df[df['node_type'] == 'cell']
    df_injected = df_injected[df_injected['node_type'] == 'cell']

    # Compute the betweenness rankings in descending order (rank1 = highest betweenness)
    df['rank'] = df['approximate_betweenness_centrality'].rank(ascending=False)
    df['rank_percentile'] = df['approximate_betweenness_centrality'].rank(ascending=False, pct=True)
    df_injected['rank'] = df_injected['approximate_betweenness_centrality'].rank(ascending=False)
    df_injected['rank_percentile'] = df_injected['approximate_betweenness_centrality'].rank(ascending=False, pct=True)

    # Calculate rank changes between the original and injected datasets
    get_injected_homograph_rank_stats(df, df_injected, args.metadata_dir, args.output_dir)


if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Evaluation the rankings of the injected homographs')

    # Path to the dataframe storing the betweeness scores of the injected dataset
    parser.add_argument('-bsdi', '--betweenness_scores_df_injected', metavar='betweenness_scores_df_injected', required=True,
    help='Path to the dataframe storing the betweeness scores of the injected dataset')

    # Path to the dataframe storing the betweeness scores of the non-injected dataset
    parser.add_argument('-bsd', '--betweenness_scores_df', metavar='betweenness_scores_df', required=True,
    help='Path to the dataframe storing the  betweeness scores')

    # Directory of the homograph injection metadata
    parser.add_argument('-md', '--metadata_dir', metavar='metadata_dir', required=True,
    help='Directory of the homograph injection metadata')

    # Output directory where the statistics are stared at
    parser.add_argument('-od', '--output_dir', metavar='output_dir', required=True,
    help='Output directory where the statistics are stared at')

    # Parse the arguments
    args = parser.parse_args()

    print('\n##### ----- Running homograph_injection_evaluation with the following parameters ----- #####\n')

    print('Betweenness scores dataframe:', args.betweenness_scores_df)
    print('Betweenness scores dataframe injected:', args.betweenness_scores_df_injected)
    print('Metadata directory:', args.metadata_dir)
    print('Output directory', args.output_dir)
    print('\n\n')

    main(args)