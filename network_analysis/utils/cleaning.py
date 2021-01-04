import networkx as nx

from timeit import default_timer as timer


def is_number_tryexcept(s):
    """ 
    Returns True if string `s` is a number.

    Taken from: https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float
    """
    try:
        float(s)
        return True
    except ValueError:
        return False

def clean_graph(G, min_str_length, remove_numerical_vals):
    '''
    Cleans the graph `G` by removing nodes based on the input arguments

    It is possible that one of `min_str_length` or `remove_numerical_vals` are
    set to None so in that case they are ignored

    Arguments
    -------
        G (networkx graph): the networkx graph to be cleaned

        min_str_length (int): the minimum string length of a node, everything shorter
        will be removed. Note that this ignores numerical nodes.

        remove_numerical_vals (boolean): specifies if nodes of numerical values are to be removed 
       
    Returns
    -------
    Returns a networkx graph
    '''

    nodes_for_removal = []
    numerical_nodes_removed = 0
    short_string_nodes_removed = 0

    print('Performing cleaning on the graph with', nx.number_of_nodes(G), 'nodes and', nx.number_of_edges(G), 'edges.')
    start = timer()
    for node in G.nodes:
        if is_number_tryexcept(node):
            if remove_numerical_vals:
                nodes_for_removal.append(node)
                numerical_nodes_removed += 1
        else:
            if min_str_length and len(node) < min_str_length:
                nodes_for_removal.append(node)
                short_string_nodes_removed += 1
    # Remove the nodes
    G.remove_nodes_from(nodes_for_removal)
    print('Finished cleaning on the graph. \nElapsed time:', timer()-start, 'seconds')

    print('Removed', numerical_nodes_removed, 'nodes with numerical values and',
        short_string_nodes_removed, 'nodes with short strings.')
    print('Cleaned graph has', nx.number_of_nodes(G), 'nodes and', nx.number_of_edges(G), 'edges.\n')
    
    return G
