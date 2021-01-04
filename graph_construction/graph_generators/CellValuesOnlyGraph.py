import networkx as nx
import numpy as np
import pandas as pd
import math
import os
import pickle
import itertools

from tqdm import tqdm


'''
This is the simplest graph construction possible.
We only keep track of only the cell values across all tables using set-semantics
and connect a cell value with all the cell values found in the same column
'''


class CellValuesOnlyGraph:

    tables = []
    G = None
    num_cell_nodes = 0
    input_data_file_type = 'csv'

    def __init__(self, tables, input_data_file_type):
        self.tables = tables
        self.input_data_file_type = input_data_file_type
        self.G = nx.Graph()
        self.fill_graph()
    
    def get_graph_from_table(self, df, table, idx_rows):

        G = nx.Graph()
        
        # Create cell nodes (Convert all cell values to strings)
        G.add_nodes_from(list(map(lambda x: str(x), df.values.flatten())), type='cell')

        # Add edges between all cell values in a given column 
        for c in df.columns:
            cell_values = list(map(lambda x: str(x), df[c].values))
            edges = list(itertools.permutations(cell_values, 2)) 
            G.add_edges_from(edges)

        return G

    def fill_graph(self):

        n_rows = 0
        for i in tqdm(range(0, len(self.tables))):           
            if self.input_data_file_type == 'csv':
                df = pd.read_csv(self.tables[i])
            elif self.input_data_file_type == 'tsv':
                df = pd.read_csv(self.tables[i], sep='\t')
            else:
                raise ValueError('input_data_file_type must be one of: csv or tsv')
            idx_rows = range(n_rows, n_rows+len(df))
            n_rows = n_rows + len(df)
            g = self.get_graph_from_table(df, 'table_' + str(i), idx_rows)
            self.G.add_nodes_from(g.nodes(data=True))
            self.G.add_edges_from(g.edges)

        # Remove all self-loops from the graph
        self.G.remove_edges_from(nx.selfloop_edges(self.G))

        if np.nan in self.G.nodes():
            self.G.remove_node(np.nan)
        if '' in self.G.nodes():
            self.G.remove_node('')
        if 'nan' in self.G.nodes():
            self.G.remove_node('nan')
        
        node_to_type_dict = nx.get_node_attributes(self.G, 'type')
        for cell in self.G.nodes():
            if node_to_type_dict[cell] == 'cell' or node_to_type_dict[cell] == 'global_cell':
                self.num_cell_nodes += 1