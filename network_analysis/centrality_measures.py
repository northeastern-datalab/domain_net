import networkx as nx
import networkit as nk
import pandas as pd
import scipy

import utils

import pickle
import argparse

from timeit import default_timer as timer
from tqdm import tqdm

def add_cardinality_column(df, G):
    '''
    Add a cardinality column in the dataframe that corresponds to the cardinality of each cell value in the dataframe

    Assumption: `df` only has cell nodes
    '''
    card_dict = {}
    for val in tqdm(df['node'].values):
        card_dict[val] = utils.graph_helpers.get_cardinality_of_homograph(G, val)
    df['cardinality'] = df['node'].map(card_dict)

def compute_centrality_measures(G_nx, df, centrality_measures):
    '''
    Returns a pandas dataframe of relevant statistical measurements for the input graph `G`
    Each row corresponds to one node in the graph.

    It uses networkit functions for its calculations

    Arguments
    -------
        G_nx (networkx graph): a networkx graph to be analyzed

        df (dataframe): dataframe of each node with its corresponding approximate BC score

        centrality_measures (list of str): list of the centrality measures to be computed
       
    Returns
    -------
    The df pandas dataframe with a new column for each new centrality measure
    '''
    print('Input graph has:', G_nx.number_of_nodes(), 'nodes and', G_nx.number_of_edges(), 'edges\n')

    # Convert NetworkX graph into networkit graph
    start = timer()
    print('Converting NetworkX graph into a networkit graph...')
    G = nk.nxadapter.nx2nk(G_nx)
    print('Finished converting NetworkX graph into a networkit graph \nElapsed time:', timer()-start, 'seconds\n')

    # Compute the centrality measures 
    for measure in centrality_measures:
        start = timer()
        print('Calculating', measure,'measure...')

        if measure == 'katz':
            # Set the alpha value to 1/lambda_max - epsilon where lambda_max is the largest
            # eigenvalue of the adjacency matrix of the graph
            
            epsilon = 1e-6
            print('Computing the adjacency matrix')
            adj_matrix = nk.algebraic.adjacencyMatrix(G)
            print('Finished computing the adjacency matrix')

            eigvals, _ = scipy.sparse.linalg.eigs(adj_matrix, k=1)
            alpha = (1 / eigvals[0].real) - epsilon
            print('alpha is set to', alpha)

            katz_centrality_scores = nk.centrality.KatzCentrality(G=G, alpha=alpha, beta=0.1, tol=1e-8).run().scores()
            df[measure] = katz_centrality_scores
        if measure == 'harmonic_closeness':
            harmonic_closeness_scores = nk.centrality.HarmonicCloseness(G=G, normalized=True).run().scores()
            df[measure] = harmonic_closeness_scores
        if measure == 'pagerank':
            pagerank_scores = nk.centrality.PageRank(G, damp=0.85, tol=1e-12).run().scores()
            df[measure] = pagerank_scores        

        print('Finished computing', measure,'measure.\nElapsed time:', timer()-start)
    
    add_cardinality_column(df, G_nx)

    return df

def main(args):
    # Load the graph file
    start = timer()
    print('Loading graph file...')
    graph = pickle.load(open(args.graph, 'rb'))
    print('Finished loading graph file \nElapsed time:', timer()-start, 'seconds\n')

    # Load the dataframe
    start = timer()
    print('Loading dataframe...')
    df = pickle.load(open(args.dataframe, 'rb'))
    print('Finished loading dataframe \nElapsed time:', timer()-start, 'seconds\n')

    # Calculate the centrality measures
    df = compute_centrality_measures(G_nx=graph, df=df, centrality_measures=args.centrality_measures)

    # save the updated dataframe with the newly calculated measures 
    df.to_pickle(args.dataframe)


if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Compute various other centrality measures (other than BC) over the graph')

    # Input graph representation of the set of tables
    parser.add_argument('-g', '--graph', metavar='graph', required=True,
    help='Path to the Graph representation of the set of tables')

    # Path to the dataframe of all the nodes in the graph with their respective BC score
    # The specified dataframe will be updated to include the newly calculated scores.
    parser.add_argument('-df', '--dataframe', metavar='dataframe', required=True,
    help='Path to the dataframe of all the nodes in the graph with their respective BC score. \
    The specified dataframe will be updated to include the newly calculated scores.')

    # A list of centrality measures to be computed for each node in the graph.
    # The list can contain any subset of the following: {katz, harmonic_closeness, pagerank}
    parser.add_argument('-cm', '--centrality_measures', metavar='centrality_measures',  nargs='+', default=[],
    help='A list of centrality measures to be computed for each node in the graph. The list can contain any subset of the \
    following: \{katz, harmonic_closeness, pagerank\}')

    # Parse the arguments
    args = parser.parse_args()

    allowed_centrality_measures = ['katz', 'harmonic_closeness', 'pagerank']

    for measure in args.centrality_measures:
        if measure not in allowed_centrality_measures:
            parser.error(str(measure) + ' is not in the list of allowed centrality measures.')

    
    print('##### ----- Running network_analysis/centrality_measures.py with the following parameters ----- #####\n')

    # print('Output directory:', args.output_dir)
    print('Graph path:', args.graph)
    print('Dataframe path:', args.dataframe)
    print('Centrality Measures:', args.centrality_measures)
    print('\n\n')

    main(args)