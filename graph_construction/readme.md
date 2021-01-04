# Graph Construction

This module is used to construct a graph representation given a set of input tables.
The graph representation can be either a bipartite graph (there are cell value nodes and attribute nodes) or a cell-values-only graph.
The produced graph is returned as a pickled networkx graph file in the `combined_graphs_output/` directory.

## Running
To construct a graph run the `main.py` file by specifying the input and output directories as well as the graph type for the constructed graph representation.
For more details on the available command line arguments you can run 
```
python main.py -h
```

### Example 1: Construct the bipartite graph representation for the Synthetic Benchmark (SB)
```
python main.py \
-id ../DATA/synthetic_benchmark/ \
-od combined_graphs_output/synthetic_benchmark_bipartite/ \
--input_data_file_type csv \
--graph_type bipartite
```
The above command line will construct the bipartite graph representation for the synthetic benchmark which can be found in the `combined_graphs_output/synthetic_example_bipartite/` directory.

### Example 2: Construct the bipartite graph representation for the Table Union Search (TUS) benchmark
```
python main.py \
-id ../DATA/table_union_search/csvfiles/ \
-od combined_graphs_output/TUS/ \
--input_data_file_type csv \
--graph_type bipartite
```
The above command line will construct the bipartite graph representation for the TUS benchmark which can be found in the `combined_graphs_output/TUS/` directory.

### Example 3: Construct the bipartite graph representation for the Table Union Search Injected (TUS-I) benchmark
```
python main.py \
-id ../DATA/table_union_search/csvfiles_no_homographs/ \
-od combined_graphs_output/TUS_no_homographs/ \
--input_data_file_type csv \
--graph_type bipartite
```
The above command line will construct the bipartite graph representation for the TUS-I benchmark which can be found in the `combined_graphs_output/TUS_no_homographs/` directory.