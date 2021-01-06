'''
Create the precision recall figures for the TUS experiment
'''

import numpy as np
import pandas as pd
import pickle
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import itertools as it
sns.set(rc={'figure.figsize':(16,10)})
sns.set(font_scale=2.4)
sns.set_style("whitegrid")

from timeit import default_timer as timer

from pathlib import Path

def calculate_measures(df, num_true_homographs):
    '''
    Calculates and adds columns precision, recall, f1_score and average_precision_at_rank in the dataframe
    for each node.

    num_true_homographs is an integer specifying the number of true homographs 
    in the dataframe based on the ground truth 
    '''
    num_homographs_seen_so_far = 0
    precision_list = []
    recall_list = []
    f1_list = []
    
    average_precision_running_sum = 0
    average_precision_at_rank_list = []

    # Calculate top-k precision/recall/f1-scores in a running fashion (start from k=1 all the way to the largest possible k)
    for k, cur_node_is_homograph in zip(range(1, df.shape[0] + 1), df['is_homograph']):
        if cur_node_is_homograph:
            num_homographs_seen_so_far += 1
        
        precision_list.append(num_homographs_seen_so_far / k)
        recall_list.append(num_homographs_seen_so_far / num_true_homographs)
        f1_list.append((2*precision_list[-1]*recall_list[-1]) / (precision_list[-1]+recall_list[-1]))

        # Calculate the Average Precision at k
        if cur_node_is_homograph:
            average_precision_running_sum += precision_list[-1]
        NF = min(k, num_true_homographs)
        average_precision_at_rank_list.append(average_precision_running_sum / NF)

    df.loc[:, 'precision'] = precision_list
    df.loc[:, 'recall'] = recall_list
    df.loc[:, 'f1_score'] = f1_list
    df.loc[:, 'average_precision_at_rank'] = average_precision_at_rank_list
    return df


def main(df_path, save_dir):
    df = pickle.load(open(df_path, 'rb'))

    # Remove attribute nodes from the dataframe
    df = df[df['node_type'] == 'cell']

    # Sort from highest to lowest betweenness centrality scores
    df = df.sort_values(by=['approximate_betweenness_centrality'], ascending=False)
    # Add a rank column for each node (based on betweenness centrality score)
    df.loc[:,'rank'] = list(range(1, df.shape[0] + 1))

    # Calculate the statistical measures
    num_true_homographs = df['is_homograph'].value_counts()[True]
    df = calculate_measures(df, num_true_homographs)

    print('Average Precision:', df['average_precision_at_rank'].values[-1])
    
    # Summary of results at the cutoff point
    print('The are', num_true_homographs, 'homographs based on the ground truth.\n')
    measures_at_cut_off = df[df['rank'] == num_true_homographs]
    print('At k =', num_true_homographs, 'the metrics are:')
    print('Precision:', measures_at_cut_off['precision'].values[0])
    print('Recall:', measures_at_cut_off['recall'].values[0])
    print('F1-Score:', measures_at_cut_off['f1_score'].values[0])
    print('Average Precision:', measures_at_cut_off['average_precision_at_rank'].values[0])

    # Melt the dictionary for easy use with seaborn
    df_melt = pd.melt(df, id_vars=['rank'], value_vars=['precision', 'recall', 'f1_score', 'average_precision_at_rank'], var_name='measure')
    

    ########## ------- Make full line chart ------- ##########
    start = timer()
    print('\nCreating full figure...')
    ax = sns.lineplot(data=df_melt, x="rank", y="value", hue='measure', style='measure', linewidth=3.2)
    ax.set(xlabel='k', ylabel='')
    ax.grid(alpha=0.4)

    plt.axvline(x=num_true_homographs, color='black', linestyle='--')
    plt.text(num_true_homographs + 5000, 0.90,'Number of true homographs cut-off line', fontsize=26)

    plt.legend(title='', loc='upper right', labels=['precision', 'recall', 'f1_score', 'AP@k'])

    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig(save_dir+'TUS_topk_full.pdf')
    fig.savefig(save_dir+'TUS_topk_full.svg')
    print('Finished creating full figure', timer()-start, 'seconds\n\n')
    plt.clf()

    ########## ------- Make up to num_true_homographs figure ------- ##########
    start = timer()
    print('\nCreating up to k figure...')
    # Update dataframe 
    df_melt = df_melt[df_melt['rank'] <= num_true_homographs]
    ax = sns.lineplot(data=df_melt, x="rank", y="value", hue='measure', style='measure', linewidth=3.2)
    ax.set(xlabel='k', ylabel='')
    ax.grid(alpha=0.4)

    plt.legend(title='', loc='upper right', labels=['precision', 'recall', 'f1_score', 'AP@k'])

    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig(save_dir+'TUS_topk_up_to_real_homographs_k.pdf')
    fig.savefig(save_dir+'TUS_topk_up_to_real_homographs_k.svg')
    print('Finished creating up to k figure. Elapsed time:', timer()-start, 'seconds\n\n')

    ########## ------- Make up to k=500 figure ------- ##########
    start = timer()
    print('\nCreating up to k=500 figure...')
    # Update dataframe 
    df_melt = df_melt[df_melt['rank'] <= 500]
    ax = sns.lineplot(data=df_melt, x="rank", y="value", hue='measure', style='measure', linewidth=3.2)
    ax.set(xlabel='k', ylabel='')
    ax.grid(alpha=0.4)

    plt.legend(title='', loc='center right', labels=['precision', 'recall', 'f1_score', 'AP@k'])

    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig(save_dir+'TUS_topk_up_to_k=500.pdf')
    fig.savefig(save_dir+'TUS_topk_up_to_k=500.svg')
    print('Finished creating up to k=500 figure. Elapsed time:', timer()-start, 'seconds\n\n')


if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Construct the top-k evaluation figures for the TUS benchmark')

    # Path to the TUS benchmark BC dataframe
    parser.add_argument('--df_path', metavar='df_path', required=True,
    help='Path to the TUS benchmark BC dataframe')

    # Path to the output directory of the figures
    parser.add_argument('--save_dir', metavar='save_dir', required=True,
    help='Path to the output directory of the figures')

    # Parse the arguments
    args = parser.parse_args()

    # df_path = 'output/TUS/graph_stats_with_groundtruth_df.pickle'
    # save_dir = 'figures/TUS/'

    # Create output directory for figures
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    main(args.df_path, args.save_dir)