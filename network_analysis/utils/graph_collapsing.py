import networkx as nx
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