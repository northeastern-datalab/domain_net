import networkx as nx
import numpy as np
import pandas as pd
import math
import os
import pickle
import itertools

from tqdm import tqdm

class BipartiteGraph:

    tables = []
    G = None
    num_cell_nodes = 0
    input_data_file_type = 'csv'
    same_capitalization = False

    def __init__(self, tables, input_data_file_type, same_capitalization):
        self.tables = tables
        self.input_data_file_type = input_data_file_type
        self.same_capitalization = same_capitalization
        self.G = nx.Graph()
        self.fill_graph()
    
    def get_graph_from_table(self, df, table, idx_rows):

        G = nx.Graph()

        # Create Attribute Nodes (Add metadata of column_name and filename for each attribute node)
        nodes_attributes = []
        for column in df.columns:
            node_attribute = column + "_" + table
            nodes_attributes.append(node_attribute)
            G.add_node(node_attribute, type='attr', column_name=column, filename=table)
        
        # Create Nodes Cell (Convert all cell values to strings)
        if self.same_capitalization:
            cell_nodes_list = list(map(lambda y:y.upper(), list(map(lambda x: str(x), df.values.flatten()))))
        else:
            cell_nodes_list = list(map(lambda x: str(x), df.values.flatten()))
        
        G.add_nodes_from(cell_nodes_list, type='cell')

        # Add edges between values and attributes 
        for c, n_c in zip(df.columns, nodes_attributes):
            if self.same_capitalization:
                values = list(map(lambda y:y.upper(), list(map(lambda x: str(x), df[c].values))))
            else:
                values = list(map(lambda x: str(x), df[c].values))
            edges_columns = list(itertools.product(values, [n_c]))
            G.add_edges_from(edges_columns)

        return G

    def fill_graph(self):

        n_rows = 0
        print('Constructing graph from tables...')
        for i in tqdm(range(0, len(self.tables))):
            
            if self.input_data_file_type == 'csv':
                df = pd.read_csv(self.tables[i], error_bad_lines=False, warn_bad_lines=False)
            elif self.input_data_file_type == 'tsv':
                df = pd.read_csv(self.tables[i], sep='\t', error_bad_lines=False, warn_bad_lines=False)
            else:
                raise ValueError('input_data_file_type must be one of: csv or tsv')
            
            idx_rows = range(n_rows, n_rows+len(df))
            n_rows = n_rows + len(df)
            g = self.get_graph_from_table(df, os.path.basename(self.tables[i]), idx_rows)
            self.G.add_nodes_from(g.nodes(data=True))
            self.G.add_edges_from(g.edges)

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
        
        print('Finished constructing graph from tables\n')