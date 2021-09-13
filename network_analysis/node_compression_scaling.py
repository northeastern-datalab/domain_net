import networkx as nx
import pickle
import random
import math
from pathlib import Path

seed = 1
output_dir = "output/node_compression/scaling/education/"
graph_path = "../graph_construction/combined_graphs_output/D4_NYC_datasets/education/bipartite/bipartite.graph"

random.seed(seed)

# Graph sizes
min_sampling_percentage=5
max_sampling_percentage=100
percentage_step_size=5

# Load base graph
base_graph = pickle.load(open(graph_path, 'rb'))

for sampling_percentage in range(min_sampling_percentage, max_sampling_percentage+1, percentage_step_size):
    print("Sampling percentage:", sampling_percentage)
    nodes_to_sample = math.ceil((base_graph.number_of_nodes()) * (sampling_percentage / 100))

    if nodes_to_sample > base_graph.number_of_nodes():
        nodes_to_sample = base_graph.number_of_nodes()

    # Extract the random nodes
    random_nodes = random.sample(base_graph.nodes(), nodes_to_sample)

    # Construct and save the subgraph
    subgraph = base_graph.subgraph(random_nodes).copy()
    subgraph_path = output_dir+str(sampling_percentage)+'_percent_nodes/' 
    Path(subgraph_path).mkdir(parents=True, exist_ok=True)
    nx.write_gpickle(subgraph, subgraph_path+'bipartite.pickle')
    print("Subgraph has", subgraph.number_of_nodes(), 'nodes and', subgraph.number_of_edges(), 'edges\n')