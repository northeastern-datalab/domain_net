import pandas as pd
import numpy as np
import networkx as nx

from tqdm import tqdm

from .graph_helpers import get_cell_node_column_names
from .graph_helpers import get_attribute_of_instance

def get_num_connected_components(marked_unambiguous_vals, G):
    '''
    Given a list of the marked unambiguous values and the graph, find the number of connected components
    in the induced subgraph from the marked_unambiguous_values and their connected attribute nodes
    '''
    # Attribute nodes connected to the marked unambiguous values
    attrs = set()
    for val in marked_unambiguous_vals:
        attrs |= set(get_attribute_of_instance(G, val))

    # Extract the subgraph
    sub_G = G.subgraph(marked_unambiguous_vals + list(attrs)).copy()
    num_connected_components = nx.algorithms.components.number_connected_components(sub_G)
    return num_connected_components

def get_avg_degree_of_uv(marked_unambiguous_vals, G):
    '''
    Returns the average degree of the unambiguous values
    '''
    degrees = []
    for val in marked_unambiguous_vals:
        degrees.append(G.degree(val))
    return np.mean(degrees)


def get_num_meanings_groundtruth(df, G):
    '''
    Given a dataframe with the 'is_homograph' groundtruth, find the groundtruth number of meanings
    for each graph

    Returns an updated dataframe with the new column 'num_meanings_groundtruth'
    '''

    df['num_meanings_groundtruth'] = 1

    df_with_homographs = df[df['is_homograph'] == True]

    # Assign the groundtruth number of meanings for each homograph in the dataframe
    for idx, row in tqdm(df_with_homographs.iterrows(), total=df_with_homographs.shape[0]):
        df.loc[idx, 'num_meanings_groundtruth'] = len(get_cell_node_column_names(G, row['node']))

    return df

def process_num_meanings_df(df, out_dict, G):
    '''
    Returns an updated dataframe with the following new columns:
    
    'num_meanings': corresponds to the number of meanings predicted for each input marked_homograph as specified in the 'out_dict'
    'is_num_meanings_correct': is a boolean column that signifies if the prediction was correct
    'num_marked_unambiguous_vals': number of nodes selected as unambiguous values
    'marked_unambiguous_values_precision': the precision of the values marked as unambiguous
    'num_connected_components': the number of connected components in the subgraph composed by the marked_unambiguous_values and their connected attribute nodes 
    'marked_hom_degree': the degree of the currently marked homograph (i.e., the number of attribute nodes it is connected to)
    'marked_uv_avg_degree': the average degree of the currently marked unambiguous values 

    The 'out_dict' is obtained by running the 'semantic_type_propagation' pipeline
    '''

    # Update the dataframe to include the number of meanings inferred by type propagation
    df['num_meanings'] = np.nan
    df['num_marked_unambiguous_vals'] = np.nan
    df['marked_unambiguous_values_precision'] = np.nan
    df['num_connected_components'] = np.nan
    df['marked_hom_degree'] = np.nan
    df['marked_uv_avg_degree'] = np.nan

    for marked_homograph in tqdm(out_dict['marked_homographs']):
        num_meanings = len(set(out_dict['marked_homographs'][marked_homograph]['attr_to_type'].values()))
        df.loc[df['node'] == marked_homograph, 'num_meanings'] = num_meanings
        df.loc[df['node'] == marked_homograph, 'num_marked_unambiguous_vals'] = len(out_dict['marked_homographs'][marked_homograph]['marked_unambiguous_values'])
        df.loc[df['node'] == marked_homograph, 'marked_unambiguous_values_precision'] = out_dict['marked_homographs'][marked_homograph]['marked_unambiguous_values_precision']

        df.loc[df['node'] == marked_homograph, 'marked_hom_degree'] = G.degree(marked_homograph)
        df.loc[df['node'] == marked_homograph, 'marked_uv_avg_degree'] = get_avg_degree_of_uv(out_dict['marked_homographs'][marked_homograph]['marked_unambiguous_values'], G)

        # Compute the number of connected components
        df.loc[df['node'] == marked_homograph, 'num_connected_components'] = get_num_connected_components(out_dict['marked_homographs'][marked_homograph]['marked_unambiguous_values'], G)

    df['is_num_meanings_correct'] = df['num_meanings'] == df['num_meanings_groundtruth']
    df['is_num_components_correct'] = df['num_connected_components'] == df['num_meanings_groundtruth']

    return df

def get_num_meanings_precision(df, nodes):
    '''
    Evaluate the precision for the predicted number of meanings for the specified `nodes`.
    '''

    df_tmp = df[df['node'].isin(nodes)]

    precision = df_tmp['is_num_meanings_correct'].value_counts()[True] / len(df_tmp.index)
    return precision