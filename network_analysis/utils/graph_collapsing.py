import networkx as nx
import utils
import copy

from tqdm import tqdm 

def get_collapsed_graph(G):
    '''
    Given a graph return its collapsed version

    A collapsed graph has all co-occuring cell nodes with degree one
    to be represented by one node.
    
    Arguments
    -------
    G (Networkx graph): A Networkx graph of the full graph

    Returns
    -------
    Returns the collapsed graph as a networkx graph, returns a mapping 
    '''

    print('\nThe original graph has:', G.number_of_nodes(), 'nodes and', G.number_of_edges(), 'edges.')

    # Get list of attribute nodes
    attr_nodes = [x for x,y in G.nodes(data=True) if y['type']=='attr']

    num_collapsed_nodes = 0

    for attr_node in tqdm(attr_nodes):
        # Get list of cell value neighbors of attr that have a degree one
        neighbors_of_degree_1 = [n for n in G.neighbors(attr_node) if G.degree[n] == 1]
        if len(neighbors_of_degree_1) > 0:
            # All the nodes in the `neighbors_of_degree_1` list should be deleted and replaced by one collapsed node
            for node in neighbors_of_degree_1:
                G.remove_node(node)
            
            # Add the collapsed node
            collapsed_node_name = attr_node+'_collapsed_node' 
            G.add_node(collapsed_node_name, type='cell', cell_type='collapsed', weight=len(neighbors_of_degree_1))

            # Add edge between collapsed node and the attr_node
            # TODO: Consider adding a weight
            G.add_edge(attr_node, collapsed_node_name, weight=len(neighbors_of_degree_1))

            num_collapsed_nodes += 1
    
    print('The collapsed graph has:', G.number_of_nodes(), 'nodes and', G.number_of_edges(), 'edges.')
    print('There are', num_collapsed_nodes, 'collapsed nodes in the new graph.\n')
    return G

def get_compressed_graph(G):
    '''
    Given the bipartite graph representation (i.e. attribute and cell nodes) of the data
    return a compressed version of the graph where sets of cell nodes that occur with the same
    set of attribute nodes are compressed into a single node
    
    Arguments
    -------
    G (Networkx graph): The bipartite graph representation of the data as a Networkx graph

    Returns
    -------
    Returns two objects
    1) The compressed graph as a networkx graph and a mapping of each compressed node
    to the set of original nodes it is composed of. 
    '''
    G = copy.deepcopy(G)

    # Dictionary mapping a set of attribute nodes to the set of cell nodes that are directly connected
    # to each attribute in the key set.
    attr_nodes_set_to_cell_nodes = {}

    # Loop over each cell node to populate the `attr_nodes_set_to_cell_nodes` dictionary
    cell_nodes = [x for x,y in G.nodes(data=True) if y['type']=='cell']
    for cell in cell_nodes:
        cur_attr_nodes = frozenset(utils.graph_helpers.get_attribute_of_instance(G, cell))
        if cur_attr_nodes not in attr_nodes_set_to_cell_nodes:
            attr_nodes_set_to_cell_nodes[cur_attr_nodes] = set([cell])
        else:
            attr_nodes_set_to_cell_nodes[cur_attr_nodes].add(cell)
    
    # Maps a compressed node to the nodes that it is composed of
    compressed_node_to_original_nodes = {}
    compressed_node_id = 0

    # Compress cell nodes if an attr_nodes_set maps to more than one cell node
    for attr_nodes_set in attr_nodes_set_to_cell_nodes:
        if len(attr_nodes_set_to_cell_nodes[attr_nodes_set]) > 1:
            cur_cell_nodes = list(attr_nodes_set_to_cell_nodes[attr_nodes_set])
            # Ensure the newly introduced compressed_node does not already exist in G
            while 'compressed_node_' + str(compressed_node_id) in G:
                compressed_node_id += 1

            # Use the first 'node' in cur_cell_nodes as the compressed node and remove all other nodes from the graph
            rename_mapping = {cur_cell_nodes[0]: 'compressed_node_' + str(compressed_node_id)}
            G = nx.relabel_nodes(G, rename_mapping)
            G.remove_nodes_from(cur_cell_nodes[1:])

            compressed_node_to_original_nodes['compressed_node_' + str(compressed_node_id)] = cur_cell_nodes
    
    
    return G, compressed_node_to_original_nodes