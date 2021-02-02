import pandas as pd
import os
import sys
import networkx as nx
import numpy as np
import pickle

# Import graph generators
from graph_generators.BipartiteGraph import BipartiteGraph
from graph_generators.CellValuesOnlyGraph import CellValuesOnlyGraph 

from pathlib import Path
import argparse

def run_pipeline_on_mode(mode, args):
    '''
    Runs the full graph construction to embeddings pipeline on a given graph construction mode
    '''
    # Read input_data directory and generate the appropriate graph for each table
    dirpath = args.input_dir
    input_data_file_type = args.input_data_file_type
    same_capitalization = args.same_capitalization
    files = list(map(lambda x: dirpath + '/' + x,  sorted(os.listdir(dirpath))))

    G = None

    # Construct the graph representation
    if mode == "bipartite":
        G = BipartiteGraph(files, input_data_file_type, same_capitalization)
    if mode == "cell_values_only":
        G = CellValuesOnlyGraph(files, input_data_file_type)

    print('Created graph has', G.G.number_of_nodes(), 'nodes and', G.G.number_of_edges(), 'edges.\n')
    if mode == "bipartite":
        cell_nodes = [x for x,y in G.G.nodes(data=True) if y['type']=='cell']
        attr_nodes = [x for x,y in G.G.nodes(data=True) if y['type']=='attr']
        print('There are', len(cell_nodes), 'nodes of type cell and', len(attr_nodes), 'nodes of type attribute.\n')
 
    # Create a file (and directory as necessary) to save the combined graph
    combined_graph_path = args.output_dir + mode
    Path(combined_graph_path).mkdir(parents=True, exist_ok=True)
    print('Writting graph to file...')
    with open(combined_graph_path + '/' + mode + '.graph', 'wb') as gfp:
        pickle.dump(G.G, gfp)
    print('Finished writting graph to file.')
   
def main(args):

    run_pipeline_on_mode(mode=args.graph_type, args=args)

if __name__ == '__main__':

    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Construct graph representationfrom input data')

    # Input directory where input files are stored
    parser.add_argument('-id', '--input_dir', metavar='input_dir', required=True,
    help='Directory of input csv files. Path must terminate with backslash "\\"')

    # Output directory where the homograph injected files are stored
    parser.add_argument('-od', '--output_dir', metavar='output_dir', required=True,
    help='Directory of where constructed graph is stored. Path must terminate with backslash "\\"')

    # File format of the input raw data (i.e. the tables). One of {csv, tsv}
    parser.add_argument('-idft', '--input_data_file_type', choices=['csv', 'tsv'], default='csv',
    metavar='input_data_file_type', required=True,
    help='File format of the input raw data (i.e. the tables). One of {csv, tsv}')

    # Specifies the type of the constructed graph
    parser.add_argument('-gt', '--graph_type', choices=['cell_values_only', 'bipartite'], default='bipartite',
    metavar='graph_type', required=True, help='Specifies the type of the constructed graph')

    # If specified, ensures that all strings in a table use the same capitalization (i.e. the are all uppercase letters)
    parser.add_argument('--same_capitalization', action='store_true',
    help='If specified, ensures that all strings in a table use the same capitalization (i.e. the are all uppercase letters)')   

    # Parse the arguments
    args = parser.parse_args()

    print('\n##### ----- Running main.py with the following parameters ----- #####\n')

    print('Input directory:', args.input_dir)
    print('Output directory:', args.output_dir)
    print('Input data file type:', args.input_data_file_type)
    print('Graph type:', args.graph_type)
    if args.same_capitalization:
        print('Ensuring same capitalization for all strings')
    print('\n\n\n')

    main(args)