# Network Analysis

This module is used to calculate the betweenness centrality (BC) scores for each node in the graph and evaluate the results (precision/recall/f1-scores) based on ground truth.

## Running
To run the homograph injection procedure run the `main.py` file by specifying the output directory, the input graph representation
as well as the ground truth of the homographs vs unambiguous values in the repository,

There are many more input line arguments such as specifying how many samples are used in the approximation of BC.

For more details on the available command line arguments you can run 
```
python main.py -h
```

### Example 1: Calculate BC scores for every node in the Synthetic Benchmark (SB)
```
python main.py \
-g ../graph_construction/combined_graphs_output/synthetic_benchmark_bipartite/bipartite/bipartite.graph \
-o output/synthetic_example_bipartite/ \
--betweenness_mode exact
```
The above command calculated the BC scores for every node in the graph representation of the synthetic benchmark.
Once the script finishes running a pandas dataframe is saved in the specified output directory with the BC score for each node.

### Example 2: Calculate Approximate BC scores for every node in the TUS benchmark by sampling 5000 nodes
```
python main.py \
-g ../graph_construction/combined_graphs_output/TUS/bipartite/bipartite.graph \
-o output/TUS/ \
--betweenness_mode approximate \
--num_samples 5000 --groundtruth_path ground_truth/groundtruth_TUS.pickle
```

### Example 3: Calculate Approximate BC scores for every node in the TUS benchmark with no homographs by sampling 5000 nodes
```
python main.py \
-g ../graph_construction/combined_graphs_output/TUS_no_homographs/bipartite/bipartite.graph \
-o output/TUS_no_homographs/ \
--betweenness_mode approximate \
--num_samples 5000 --groundtruth_path ground_truth/groundtruth_TUS.pickle
```