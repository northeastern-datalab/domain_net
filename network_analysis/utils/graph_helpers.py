import multiprocessing
import networkx as nx

from tqdm import tqdm
from joblib import Parallel, delayed


def get_attribute_of_instance(G, instance_node):
    '''
    Given a graph `G` and an `instance_node` from the graph return its corresponding set of attribute nodes
    '''
    attribute_nodes = []
    for neighbor in G[instance_node]:
        if G.nodes[neighbor]['type'] == 'attr':
            attribute_nodes.append(neighbor)
    return attribute_nodes

def get_neighbors_of_instance(G, instance_node, type='attr'):
    '''
    Given a graph `G` and an `instance_node` from the graph return its direct neighbors of type=`type`
    '''
    neighbor_nodes = []
    for neighbor in G[instance_node]:
        if G.nodes[neighbor]['type'] == type:
            neighbor_nodes.append(neighbor)
    return neighbor_nodes

        
def get_instances_for_attribute(G, attribute_node):
    '''
    Given a graph `G` and an `instance_node` from the graph find its cell nodes
    '''
    instances_nodes = []
    for neighbor in G[attribute_node]:
        if G.nodes[neighbor]['type'] == 'cell':
            instances_nodes.append(neighbor)
    return instances_nodes


def get_cardinality_of_homograph(G, val):
    '''
    Given a graph `G` return the cardinality of cell value val
    '''
    attribute_nodes = get_attribute_of_instance(G, val)
    cell_vals = set()
    for attr in attribute_nodes:
        cell_vals |= set(get_instances_for_attribute(G, attr))
    return len(cell_vals)

def get_cell_node_neighbors(G, cell_node):
    '''
    Given a graph `G` and a `cell_node` return all its cell node neighbors.
    The cell node neighbors of a node are all the cell nodes connected that are connected to its attribute nodes
    '''
    attribute_nodes = get_attribute_of_instance(G, cell_node)
    cell_vals = set()
    for attr in attribute_nodes:
        cell_vals |= set(get_instances_for_attribute(G, attr))
    return list(cell_vals)

def get_row_wise_neighbors(G, cell_node):
    '''
    Given a graph `G` and a `cell_node` return all its row-wise cell node neighbors.
    The row-wise cell node neighbors of a node are all the cell nodes that are connected to its row nodes 
    '''
    row_nodes = get_neighbors_of_instance(G, cell_node, 'row')
    cell_vals = set()
    for row_node in row_nodes:
        cell_vals |= set(get_instances_for_attribute(G, row_node))
    return list(cell_vals)

def get_cell_node_column_names(G, cell_node):
    '''
    Given a graph `G` and a `cell_node` return the unique column names of all its connected attribute nodes

    Note: This function only works for graphs `G` where attribute nodes have a 'column_name' type (i.e. synthetic benchmark graphs) 
    '''
    attribute_nodes = get_attribute_of_instance(G, cell_node)
    column_names = []
    for attr in attribute_nodes:
        column_names.append(G.nodes[attr]['column_name'])
    return list(set(column_names))

def get_cell_node_column_names_frequency(G, cell_node):
    '''
    Given a graph `G` and a `cell_node` return a dictionary keyed by each unique column name to its frequency in the the graph

    Note: This function only works for graphs `G` where attribute nodes have a 'column_name' type (i.e. synthetic benchmark graphs) 
    '''
    attribute_nodes = get_attribute_of_instance(G, cell_node)
    column_names_freq_dict = {}
    for attr in attribute_nodes:
        col_name = G.nodes[attr]['column_name']
        if col_name in column_names_freq_dict:
            column_names_freq_dict[col_name] += 1
        else:
            column_names_freq_dict[col_name] = 1

    return column_names_freq_dict

def get_cell_node_file_names(G, cell_node):
    '''
    Given a graph `G` and a `cell_node` return the unique filenames from its attribute names (i.e., the table files)

    Note: This function only works for graphs `G` where attribute nodes have a 'column_name' type 
    '''
    attribute_nodes = get_attribute_of_instance(G, cell_node)
    file_names = []
    for attr in attribute_nodes:
        file_names.append(attr.split('_')[-1])
    return list(set(file_names))