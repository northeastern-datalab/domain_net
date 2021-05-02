'''
Used to extract the semantic domains each cell value the TUS Benchmark belongs to
'''

import networkx as nx
import pandas as pd

import pickle
import argparse

from tqdm import tqdm
from timeit import default_timer as timer

def get_attributes_of_instance(G, instance_node):
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

def get_num_meanings_of_homograph(homograph, filename_column_unionable_pairs_dict, G):
    '''
    Return the number of meanings of a given homograph
    '''

    attrs = get_attributes_of_instance(G, homograph)

    # Get filename column tuples that we test for
    filename_column_tuples = []
    for attr in attrs:
        column_name = G.nodes[attr]['column_name']
        file_name = G.nodes[attr]['filename']
        filename_column_tuples.append((file_name, column_name))
    

    # print('There are', len(attrs), 'attributes the homograph connects to')
    sets_of_unionable_vals_set = set([])
    for tup in filename_column_tuples:
        sets_of_unionable_vals_set.add(frozenset(filename_column_unionable_pairs_dict[tup]))

    return len(sets_of_unionable_vals_set)


def get_domains(filename_column_tuple_to_unionable_pairs_dict):
    '''
    Returns a dictionary of a domain ID mapping to the set of filename_col_tuples that are unionable with each other

    Notice that each domain must not be a subset of any other domain
    '''
    domains = set([])
    for filename_col in filename_column_tuple_to_unionable_pairs_dict:
        cur_domain = filename_column_tuple_to_unionable_pairs_dict[filename_col]

        # Make sure that no domain is a subset of another domain
        cur_domain_is_subset = False
        for s in domains.copy():
            if cur_domain.issubset(s):
                cur_domain_is_subset = True
                break
            elif s.issubset(cur_domain):
                # Domain s should be removed from the set of domains as it is a subset of cur_domain
                domains.remove(s)

        # Add the current set as it isn't a subset of any other domain        
        if not cur_domain_is_subset:
            domains.add(frozenset(cur_domain))

    # Give an ID to each domain
    domains_dict = {}
    id = 0
    for d in domains:
        domains_dict[id] = d
        id += 1
    
    return domains_dict

def main(args):

    start = timer()
    print('Loading input files')
    filename_column_unionable_pairs_dict = pickle.load(open(args.unionable_pairs_dict, 'rb'))
    G = pickle.load(open(args.graph, 'rb'))
    print('Finished loading input files \nElapsed time:', timer()-start, 'seconds\n')


    attr_nodes = [n for n, d in G.nodes(data=True) if d['type']=='attr']
    filename_column_tuples_in_graph = set()
    for attr_node in attr_nodes:
        file_name = G.nodes[attr_node]['filename']
        column_name = G.nodes[attr_node]['column_name']
        filename_column_tuples_in_graph.add((file_name, column_name))

    # A dictionary keyed by (filename, column) tuple to set of unionable (filename, column) tuples
    filename_column_tuple_to_unionable_pairs_dict = {}

    # Loop through all files in the groundtruth
    for filename in tqdm(filename_column_unionable_pairs_dict):
        for other_filename in filename_column_unionable_pairs_dict[filename]:
            for column_pair in filename_column_unionable_pairs_dict[filename][other_filename]:

                # Check if (filename, column_pair[0]) is in the dictionary (if not initialize it)
                if (filename, column_pair[0]) not in filename_column_tuple_to_unionable_pairs_dict:
                    # Make sure that (filename, column_pair[0]) exists in the filename_column_tuples_in_graph set
                    if (filename, column_pair[0]) in filename_column_tuples_in_graph:
                        filename_column_tuple_to_unionable_pairs_dict[((filename, column_pair[0]))] = set()
                else:
                    # Make sure that (other_filename, column_pair[1]) exists in the filename_column_tuples_in_graph
                    if (other_filename, column_pair[1]) in filename_column_tuples_in_graph:
                        filename_column_tuple_to_unionable_pairs_dict[((filename, column_pair[0]))].add((other_filename, column_pair[1]))

    # Get the domains. Each domain is specified by a set of filename_col_tuples that are unionable with each other
    domains_dict = get_domains(filename_column_tuple_to_unionable_pairs_dict)
    print("There are:", len(domains_dict), "domains")

    # TODO: Construct a mapping from each attribute node to each domainID 

    # Check for subsets included within a set
    num_domains_that_are_subdomains = 0
    for domain1 in domains_dict.values():
        for domain2 in domains_dict.values():
            if (domain1 != domain2):
                if domain1.issubset(domain2):
                    num_domains_that_are_subdomains += 1
    print("There are", num_domains_that_are_subdomains, "domains that are subsets of another domain")


if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Find the domains corresponding to each value in the TUS Benchmark')

    # Path to the input graph
    parser.add_argument('-g', '--graph', metavar='graph', required=True,
    help='Path to the input graph')

    # Path to the filename_column_unionable_pairs dictionary
    parser.add_argument('--unionable_pairs_dict', metavar='unionable_pairs_dict', required=True,
    help='Path to the filename_column_unionable_pairs dictionary')

    # # Output directory for the homograph to number of meanings dictionary
    # parser.add_argument('-od', '--output_dir', metavar='df', required=True,
    # help='Output directory for the homograph to number of meanings dictionary')

    # Parse the arguments
    args = parser.parse_args()

    print('##### ----- domains.py with the following parameters ----- #####\n')

    print('Graph path:', args.graph)
    print('Unionable pairs dictionary path:', args.unionable_pairs_dict)
    # print('Graph stats with groundtruth dataframe path:', args.dataframe)
    # print('Output directory:', args.output_dir)
    print('\n\n')

    main(args)