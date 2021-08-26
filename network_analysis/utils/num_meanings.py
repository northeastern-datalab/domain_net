import pandas as pd
import numpy as np

from .graph_helpers import get_cell_node_column_names

def get_num_meanings_groundtruth(df, G):
    '''
    Given a dataframe with the 'is_homograph' groundtruth, find the groundtruth number of meanings
    for each graph

    Returns an updated dataframe with the new column 'num_meanings_groundtruth'
    '''

    df['num_meanings_groundtruth'] = np.nan

    # Assign the groundtruth number of meanings for each homograph in the dataframe
    for idx, row in df[df['is_homograph'] == True].iterrows():
        df.loc[idx, 'num_meanings_groundtruth'] = len(get_cell_node_column_names(G, row['node']))

    return df

def process_num_meanings_df(df, out_dict):
    '''
    Returns an updated dataframe with the following new columns:
    
    'num_meanings': corresponds to the number of meanings predicted for each input marked_homograph as specified in the 'out_dict'
    'is_num_meanings_correct': is a boolean column that signifies if the prediction was correct
    'num_marked_unambiguous_vals': number of nodes selected as unambiguous values
    'marked_unambiguous_values_precision': the precision of the values marked as unambiguous 

    The 'out_dict' is obtained by running the 'semantic_type_propagation' pipeline
    '''

    # Update the dataframe to include the number of meanings inferred by type propagation
    df['num_meanings'] = np.nan
    df['num_marked_unambiguous_vals'] = np.nan
    df['marked_unambiguous_values_precision'] = np.nan

    for marked_homograph in out_dict['marked_homographs']:
        num_meanings = len(set(out_dict['marked_homographs'][marked_homograph]['attr_to_type'].values()))
        df.loc[df['node'] == marked_homograph, 'num_meanings'] = num_meanings
        df.loc[df['node'] == marked_homograph, 'num_marked_unambiguous_vals'] = len(out_dict['marked_homographs'][marked_homograph]['marked_unambiguous_values'])
        df.loc[df['node'] == marked_homograph, 'marked_unambiguous_values_precision'] = out_dict['marked_homographs'][marked_homograph]['marked_unambiguous_values_precision']

    df['is_num_meanings_correct'] = df['num_meanings'] == df['num_meanings_groundtruth']

    return df

def get_num_meanings_precision(df, nodes):
    '''
    Evaluate the precision for the predicted number of meanings for the specified `nodes`.
    '''

    df_tmp = df[df['node'].isin(nodes)]

    precision = df_tmp['is_num_meanings_correct'].value_counts()[True] / len(df_tmp.index)
    return precision