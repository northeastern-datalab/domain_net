import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from pathlib import Path

def calculate_measures(df, num_true_homographs):
    '''
    Calculates and adds columns precision, recall and f1_score in the dataframe
    for each node.

    num_true_homographs is an integer specifying the number of true homographs 
    in the dataframe based on the ground truth 
    '''
    num_homographs_seen_so_far = 0
    precision_list = []
    recall_list = []
    f1_list = []

    # Calculate top-k precision/recall/f1-scores in a running fashion (start from k=1 all the way to the largest possible k)
    for k, cur_node_is_homograph in zip(range(1, df.shape[0] + 1), df['is_homograph']):
        if cur_node_is_homograph:
            num_homographs_seen_so_far += 1
        
        precision_list.append(num_homographs_seen_so_far / k)
        recall_list.append(num_homographs_seen_so_far / num_true_homographs)
        f1_list.append((2*precision_list[-1]*recall_list[-1]) / (precision_list[-1]+recall_list[-1]))

    df.loc[:, 'precision'] = precision_list
    df.loc[:, 'recall'] = recall_list
    df.loc[:, 'f1_score'] = f1_list
    return df

def topk_line_chart(df):
    '''
    Plots the top-k precision/recall/f1_scores given a dataframe
    with rank in the x-axis
    '''
    plt.clf()
    plt.plot('rank', 'precision', data=df)
    plt.plot('rank', 'recall', data=df)
    plt.plot('rank', 'f1_score', data=df)
    plt.ylabel('Measure', fontsize=22)
    plt.xlabel('k', fontsize=22)
    plt.title('Top-k Measures', fontsize=22)
    plt.legend(fontsize=22)

    return plt.figure(1)

def top_k_graphs(df, output_dir):
    '''
    Given a dataframe with existing ground truth perform a top-k evaluation
    based on precision/recall and f1-score metrics
    '''
    figures_dir = output_dir+'figures/'
    Path(output_dir+'figures/').mkdir(parents=True, exist_ok=True)

    # Remove attribute nodes from the dataframe
    df = df[df['node_type'] == 'cell']

    # Sort from highest to lowest betweenness centrality scores
    df = df.sort_values(by=['approximate_betweenness_centrality'], ascending=False)
    # Add a rank column for each node (based on betweenness centrality score)
    df.loc[:,'rank'] = list(range(1, df.shape[0] + 1))

    # Calculate the statistical measures
    num_true_homographs = df['is_homograph'].value_counts()[True]
    df = calculate_measures(df, num_true_homographs)

    # Summary of results at the cutoff point
    print('The are', num_true_homographs, 'homographs based on the ground truth.')
    measures_at_cut_off = df[df['rank'] == num_true_homographs]
    print('At k =', num_true_homographs, 'the metrics are:')
    print('Precision:', measures_at_cut_off['precision'].values[0])
    print('Recall:', measures_at_cut_off['recall'].values[0])
    print('F1-Score:', measures_at_cut_off['f1_score'].values[0])

    # Set-up some matplotlib parameters
    matplotlib.rcParams['figure.figsize'] = (20.0, 12.0)
    matplotlib.rcParams["axes.grid"] = False
    matplotlib.rc('xtick', labelsize=18) 
    matplotlib.rc('ytick', labelsize=18)

    # Draw the full top-k graph 
    fig = topk_line_chart(df)
    plt.axvline(x=num_true_homographs, color='black', linestyle='--')
    plt.text(num_true_homographs,0.90,'Number of true homographs cut-off line', fontsize=20)
    fig.savefig(figures_dir+'topk_full.pdf', bbox_inches='tight')
    plt.clf()

    # Draw the top-k graph up to k=num_true_homographs
    fig = topk_line_chart(df.head(num_true_homographs))
    fig.savefig(figures_dir+'topk_up_to_real_homographs_k.pdf', bbox_inches='tight')


def calculate_eval_measures_from_gt_column(df, measure, gt_column_name, num_true_homographs):
    '''
    Calculates and adds columns precision_`measure`, recall_`measure`, f1_score_`measure`
    for the specified `measure` given groundtruth provided by `gt_column_name`.

    The measure is evaluated in descending fashion. Higher the value of the measure the more likely
    it is assumed to be a homograph

    Arguments
    -------
    df (pandas dataframe): a dataframe where each row corresponds to one node
    with a groundtruth label specified by `gt_column_name` and a measure `measure` 

    measure (str): column name in the dataframe that specifies the measure to evaluate for

    gt_column_name (str): column name in the dataframe that specifies the is_homograph groundtruth (True or False)

    num_true_homographs (int): an integer specifying the number of true homographs 
    in the dataframe based on the ground truth 

    Returns
    -------
    The input dataframe with the added evaluation columns
    '''
    num_homographs_seen_so_far = 0
    precision_list = []
    recall_list = []
    f1_list = []

    # Sort the dataframe by the specified measure (high->low)
    df = df.sort_values(by=[measure], ascending=False)
    df.loc[:,measure+'_rank'] = list(range(1, df.shape[0] + 1))
    df[measure+'_dense_rank'] = df[measure].rank(method='dense', ascending=False)

    # Calculate top-k precision/recall/f1-scores in a running fashion (start from k=1 all the way to the largest possible k)
    for k, cur_node_is_homograph in zip(range(1, df.shape[0] + 1), df[gt_column_name]):
        if cur_node_is_homograph:
            num_homographs_seen_so_far += 1
        
        precision_list.append(num_homographs_seen_so_far / k)
        recall_list.append(num_homographs_seen_so_far / num_true_homographs)

        f1_score = (2*precision_list[-1]*recall_list[-1]) / (precision_list[-1]+recall_list[-1])
        f1_list.append(f1_score)

    df.loc[:, measure+'_precision'] = precision_list
    df.loc[:, measure+'_recall'] = recall_list
    df.loc[:, measure+'_f1_score'] = f1_list

    # Remove NaN values from F1-score
    df[measure+'_f1_score'] = df[measure+'_f1_score'].fillna(value=0)
    return df