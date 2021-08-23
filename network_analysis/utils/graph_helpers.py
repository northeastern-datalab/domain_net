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
