from cdlib import algorithms, viz
from sklearn.metrics import f1_score
import networkx as nx
import pickle

import argparse
import json

from timeit import default_timer as timer
from pathlib import Path

def relabel_mappings(G):
    '''
    Given a graph with non-integer labeled nodes, relabel all its nodes to be integers.

    Return 3 objects:
    
    1) The new integer labeled graph
    2) A dictionary mapping the old ids to the new integer ids
    3) A dictionary mapping the new integer ids to the old ids
    '''
    old_id_to_int_id_dict = dict((id, int_id) for (id, int_id) in zip(G.nodes(), range(G.number_of_nodes())))

    new_G = nx.relabel_nodes(G, old_id_to_int_id_dict)
    int_id_to_old_id_dict = {int_id: old_id for old_id, int_id in old_id_to_int_id_dict.items()}

    return new_G, old_id_to_int_id_dict, int_id_to_old_id_dict

def get_node_to_communities_dict(coms):
    '''
    Return a dictionary mapping each node to a list of its community ids
    '''
    node_id_to_communities_dict = {}
    for community_id in range(len(coms)):
        for node in coms[community_id]:
            if node in node_id_to_communities_dict:
                node_id_to_communities_dict[node].append(community_id)
            else:
                node_id_to_communities_dict[node] = [community_id]
    
    return node_id_to_communities_dict

def select_cell_nodes_in_multiple_communities(orig_graph, node_id_to_communities_dict, int_id_to_old_id_dict):
    '''
    Given the node_id_to_communities_dict identify all the cell nodes that belong in more than one community

    The returned list of cell nodes are labeled using the original graph
    '''

    # Find the nodes that belong in more than one community  
    nodes_in_multiple_communities = []
    for node in node_id_to_communities_dict:
        if len(node_id_to_communities_dict[node]) > 1:
            nodes_in_multiple_communities.append(node)

    # Select only cell nodes and use the original graph labels
    cell_nodes_in_multiple_communities = []
    for node in nodes_in_multiple_communities:
        if orig_graph.nodes[int_id_to_old_id_dict[node]]['type'] == 'cell':
            cell_nodes_in_multiple_communities.append(int_id_to_old_id_dict[node])

    return cell_nodes_in_multiple_communities

def get_precision_recall_f1_score(gt_homographs, predicted_homographs):
    precision = len(gt_homographs & predicted_homographs) / len(predicted_homographs)
    recall = len(gt_homographs & predicted_homographs) / len(gt_homographs)

    if (precision + recall) == 0:
        f1_score=0
    else:
        f1_score = (2* precision * recall) / (precision + recall)

    return precision, recall, f1_score

def get_communities(G, method='big_clam'):
    if method == 'big_clam':
        coms = algorithms.big_clam(G)
    elif method == 'core_expansion':
        coms = algorithms.core_expansion(G)
    elif method == 'lpanni':
        coms = algorithms.lpanni(G)
    elif method == 'danmf':
        coms = algorithms.danmf(G)
    
    return coms

def main(args):
    selected_methods=['core_expansion', 'lpanni', 'big_clam', 'danmf']

    # Load the ground truth
    gt = pickle.load(open(args.groundtruth_path, 'rb'))
    gt_homographs = set([val for val in gt if gt[val]=='homograph'])

    # Load the graph and convert to to integer only indices
    G = pickle.load(open(args.graph, 'rb'))
    G_new, old_id_to_int_id_dict, int_id_to_old_id_dict = relabel_mappings(G=G)

    # For each method compute the communities and evaluate them 
    for method in selected_methods:
        method_output_dir = args.output_dir+method+'/'
        Path(method_output_dir).mkdir(parents=True, exist_ok=True)

        output_dict = {
            "time": -1, "num_communities": -1, "num_nodes_in_multiple_communities": -1,
            "precision": 0, "recall": 0, "f1_score": 0
        }

        print("Running community detection using:", method, "...")
        start = timer()
        coms = get_communities(G=G_new, method=method)
        clustering_time = timer()-start
        output_dict['time'] = clustering_time
        print("Elapsed time using", method, 'is', clustering_time, 'seconds')

        num_communities = len(coms.communities)
        output_dict['num_communities'] = num_communities
        print("Identified", num_communities, 'communities using:', method)

        if num_communities > 1:        
            # Find cell nodes in multiple communities
            node_id_to_communities_dict = get_node_to_communities_dict(coms.communities)
            cell_nodes_in_multiple_communities = select_cell_nodes_in_multiple_communities(
                orig_graph=G,
                node_id_to_communities_dict=node_id_to_communities_dict,
                int_id_to_old_id_dict=int_id_to_old_id_dict
            )

            if len(cell_nodes_in_multiple_communities) > 0:
                precision, recall, f1_score = get_precision_recall_f1_score(gt_homographs=gt_homographs, predicted_homographs=set(cell_nodes_in_multiple_communities))
                output_dict['num_nodes_in_multiple_communities'] = len(cell_nodes_in_multiple_communities)
                output_dict['precision'] = precision
                output_dict['recall'] = recall
                output_dict['f1_score'] = f1_score
                print("Precision:", precision, "Recall:", recall, "F1-Score:", f1_score)
            else:
                print("There are no cell nodes that belong in more than one community")

        # Save the communities and the related metadata
        with open(method_output_dir + 'communities.pickle', 'wb') as fp:
            pickle.dump(coms, fp)

        with open(method_output_dir + 'cell_nodes_in_multiple_communities.pickle', 'wb') as fp:
            pickle.dump(cell_nodes_in_multiple_communities, fp)

        with open(method_output_dir + 'stats.json', 'w') as fp:
            json.dump(output_dict, fp, indent=4)

        print('\n\n')


if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Perform overlapping community detection algorithms \
    to identify homographs')

    # Output directory where output files and figures are stored
    parser.add_argument('-o', '--output_dir', required=True,
    help='Path to the output directory where output files are stored. \
    Path must terminate with backslash "\\"')

    # Input graph representation of the set of tables
    parser.add_argument('-g', '--graph', required=True,
    help='Path to the Graph representation of the set of tables')

    # Path to the Ground truth
    parser.add_argument('-gt', '--groundtruth_path',
    help='Specifies the ground truth file')


    # Parse the arguments
    args = parser.parse_args()

    print('##### ----- Running network_analysis/community_detection.py with the following parameters ----- #####\n')

    print('Output directory:', args.output_dir)
    print('Graph path:', args.graph)
    print('Ground Truth path:', args.groundtruth_path)
    print('\n\n')

    main(args)