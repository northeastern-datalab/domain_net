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


def get_stats_for_node(node, g):
    '''
    Get the cardinality and column_names for a given node
    '''
    # Get a list of attribute nodes that cur_node is present in
    attr_nodes = get_attribute_of_instance(g, node)
    unique_vals = set()
    column_names = set()
    for attr in attr_nodes:
        vals = get_instances_for_attribute(g, attr)
        unique_vals.update(vals)
        column_names.add(g.nodes[attr]['column_name'])
    
    return {node: {'cardinality': len(unique_vals), 'column_names': column_names}}

def get_per_value_stats_from_graph(g):
    '''
    Return a dictionary keyed by each cell value and maps to the
    list of column names from each dictionary as well as the number of 
    values it co-occurs with (i.e. the domain size of the value) 
    '''
    # Get a list of cell nodes
    cell_nodes = [x for x,y in g.nodes(data=True) if y['type']=='cell']

    # Dictionary to build and return
    value_stats_dict = {}

    print('Building value_stats_dict for cell nodes')

    for cur_node in tqdm(cell_nodes):
        # Get a list of attribute nodes that cur_node is present in
        attr_nodes = get_attribute_of_instance(g, cur_node)
        unique_vals = set()
        column_names = set()
        for attr in attr_nodes:
            vals = get_instances_for_attribute(g, attr)
            unique_vals.update(vals)
            column_names.add(g.nodes[attr]['column_name'])
        value_stats_dict[cur_node] = {'cardinality': len(unique_vals), 'column_names': column_names}

    return value_stats_dict