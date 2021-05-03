'''
Used to extract the semantic domains each cell value the TUS Benchmark belongs to
'''

import networkx as nx

import pickle
import argparse
import json
import copy

from tqdm import tqdm
from timeit import default_timer as timer

def get_filename_column_tuple_to_unionable_pairs_dict(filename_column_unionable_pairs_dict, filename_column_tuples_in_graph):
    '''   
    Arguments
    -------
    groundtruth (dict): Dictionary of (key: filename, value: dictionary_per_file)
    dictionary_per_file is a dictionary of (key: another_filename, list_of_column_pairs)
    Each column_pair in list_of_column_pairs is a pair/tuple of columns from two tables that are unionable

    filename_column_tuples_in_graph (set of tuples): A set of tuples corresponding to the
    (filename, column) tuples found in the graph

    Returns
    -------
    A dictionary keyed by (filename, column) tuple to set of unionable (filename, column) tuples that are present in the graph
    '''
    
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
                        # Every (filename, column_pair[0]) tuple is unionable with itself so add it in the set
                        filename_column_tuple_to_unionable_pairs_dict[((filename, column_pair[0]))].add((filename, column_pair[0]))
                else:
                    # Make sure that (other_filename, column_pair[1]) exists in the filename_column_tuples_in_graph
                    if (other_filename, column_pair[1]) in filename_column_tuples_in_graph:
                        filename_column_tuple_to_unionable_pairs_dict[((filename, column_pair[0]))].add((other_filename, column_pair[1]))
    
    return filename_column_tuple_to_unionable_pairs_dict


def get_domains(filename_column_tuple_to_unionable_pairs_dict):
    '''
    Returns a dictionary of a domain ID mapping to the set of filename_col_tuples that are unionable with each other

    Notice that each domain must not be a subset of any other domain
    '''
    domains = set([])
    for filename_col in filename_column_tuple_to_unionable_pairs_dict:
        cur_domain = filename_column_tuple_to_unionable_pairs_dict[filename_col]
        assert len(cur_domain) >= 1, "Each domain must be composed by at least one (filename, column) tuple!"

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

def improve_domains(domains_dict):
    '''
    Arguments
    -------
    domains_dict (dict): a dictionary that maps a domain_id to the set of filename_col_tuples that are unionable with each other

    Returns
    -------
    Return a new domains_dict that reduces the number of domains by identifying domains that are made up of the same set or subset of 
    column names and consolidates them into one domain

    Example
    -------
    If domain 1 has as column names: {A, B, C} and domain 2 also has as column names: {A, B} then join the two domains into one 
    '''
    new_domains_dict = copy.deepcopy(domains_dict)

    # Dictionary keyed by the set of col_names mapping to a list of domain_ids having that set of col_names
    col_names_sets_to_domain_ids = {}
    for domain_id in domains_dict:
        cur_col_names = frozenset(set([tup[1] for tup in domains_dict[domain_id]]))

        ### UNCOMMEND BELOW IF WE WANT TO USE SUBSET DEFINITION FOR COMBINING DOMAINS ######
        # Make sure that col_names is not a subset of another col_names set
        cur_col_names_is_subset = False
        for s in col_names_sets_to_domain_ids.copy():
            if cur_col_names.issubset(s):
                cur_col_names_is_subset = True
                col_names_sets_to_domain_ids[s].append(domain_id)
                break
            elif s.issubset(cur_col_names):
                cur_col_names_is_subset = True
                # cur_col_names set s should be removed from the set of col_names as it is a subset of cur_col_names
                domain_ids = col_names_sets_to_domain_ids[s]
                domain_ids.append(domain_id)
                col_names_sets_to_domain_ids.pop(s, None)
                col_names_sets_to_domain_ids[cur_col_names] = domain_ids
                break
        # Add the current set as it isn't a subset of any other domain        
        if not cur_col_names_is_subset:
            if cur_col_names not in col_names_sets_to_domain_ids:
                col_names_sets_to_domain_ids[cur_col_names] = [domain_id]
            else:
                col_names_sets_to_domain_ids[cur_col_names].append(domain_id)

        # if cur_col_names not in col_names_sets_to_domain_ids:
        #     col_names_sets_to_domain_ids[cur_col_names] = [domain_id]
        # else:
        #     col_names_sets_to_domain_ids[cur_col_names].append(domain_id)

    num_domains_removed = 0

    # Join all domains for which there are 2 or more domains mapping to the same set of col_names
    for col_name_set in col_names_sets_to_domain_ids:
        if len(col_names_sets_to_domain_ids[col_name_set]) >= 2:
            domain_ids = col_names_sets_to_domain_ids[col_name_set]
            # Get combined set of (filename, column) tuples for all the domains in `domain_ids` list
            combined_set = set()
            for domain_id in domain_ids:
                combined_set |= new_domains_dict[domain_id]
            
            # Use the first id in `domain_ids` to serve as the combined domain 
            new_domains_dict[domain_ids[0]] = combined_set

            # Remove the remaining domains as they have been combined
            for i in range(1, len(domain_ids)):
                num_domains_removed += 1
                new_domains_dict.pop(domain_ids[i], None)

    print(num_domains_removed, "domains were removed by removing multiple domains with the same set or subset of column names")
    
    # Rename the domain ids so they are consecutive integers starting from 0
    domain_ids_sorted = sorted(list(new_domains_dict.keys()))
    old_id_to_new_id_map = {}
    for new_id, old_id in zip(range(len(domain_ids_sorted)), domain_ids_sorted):
        old_id_to_new_id_map[old_id] = new_id
    improved_domains_dict = dict((old_id_to_new_id_map[key], value) for (key, value) in new_domains_dict.items())

    return improved_domains_dict


def main(args):

    start = timer()
    print('Loading input files')
    filename_column_unionable_pairs_dict = pickle.load(open(args.unionable_pairs_dict, 'rb'))
    G = pickle.load(open(args.graph, 'rb'))
    print('Finished loading input files \nElapsed time:', timer()-start, 'seconds\n')

    # Get list of attribute nodes and valid (filename, column) tuples for the current graph
    attr_nodes = [n for n, d in G.nodes(data=True) if d['type']=='attr']
    print("There are", len(attr_nodes), "attribute nodes in the graph\n")
    filename_column_tuples_in_graph = set()
    for attr_node in attr_nodes:
        file_name = G.nodes[attr_node]['filename']
        column_name = G.nodes[attr_node]['column_name']
        filename_column_tuples_in_graph.add((file_name, column_name))

    # A dictionary keyed by (filename, column) tuple to set of unionable (filename, column) tuples
    print("Constructing the filename_column_tuple_to_unionable_pairs_dict...")
    filename_column_tuple_to_unionable_pairs_dict = get_filename_column_tuple_to_unionable_pairs_dict(
        filename_column_unionable_pairs_dict,
        filename_column_tuples_in_graph
    )

    # Get the domains. Each domain is specified by a set of filename_col_tuples that are unionable with each other
    print("\nGrouping (filename, column) tuples into domains...")
    domains_dict = get_domains(filename_column_tuple_to_unionable_pairs_dict)
    print("There are:", len(domains_dict), "domains\n")

    # Some domains seem to share the same column names but are still considered different domains. Improve upon that
    domains_dict = improve_domains(domains_dict)
    print("There are now", len(domains_dict), "domains\n")

    # Construct a mapping from each attribute node to the set of domainIDs
    attr_to_domain_ids = {}
    num_attrs_with_more_than_one_domain = 0
    for attr in attr_nodes:
        cur_filename_col_tuple = (G.nodes[attr]['filename'], G.nodes[attr]['column_name'])
        matching_domain_ids = []
        for domain_id in domains_dict:
            if cur_filename_col_tuple in domains_dict[domain_id]:
                matching_domain_ids.append(domain_id)
        if len(matching_domain_ids) > 1:
            num_attrs_with_more_than_one_domain += 1
        attr_to_domain_ids[attr] = matching_domain_ids

    print("There are", num_attrs_with_more_than_one_domain, "attribute nodes that map to more than one domain")
    
    # Write domains info to file specified by args.output_path
    output_dict = {}
    domains_dict = {dom_id: list(domains_dict[dom_id]) for dom_id in domains_dict}
    output_dict['domains'] = domains_dict
    output_dict['attributes'] = attr_to_domain_ids

    with open(args.output_path, "w") as outfile: 
        json.dump(output_dict, outfile, indent=4)


if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Find the domains corresponding to each value in the TUS Benchmark')

    # Path to the input graph
    parser.add_argument('-g', '--graph', metavar='graph', required=True,
    help='Path to the input graph')

    # Path to the filename_column_unionable_pairs dictionary
    parser.add_argument('--unionable_pairs_dict', metavar='unionable_pairs_dict', required=True,
    help='Path to the filename_column_unionable_pairs dictionary')

    # Output path for the homograph to number of meanings json file
    parser.add_argument('-op', '--output_path', metavar='df', required=True,
    help='Output path for the homograph to number of meanings json file')

    # Parse the arguments
    args = parser.parse_args()

    print('##### ----- Running domains.py with the following parameters ----- #####\n')

    print('Graph path:', args.graph)
    print('Unionable pairs dictionary path:', args.unionable_pairs_dict)
    print('Output directory:', args.output_path)
    print('\n\n')

    main(args)