#!/bin/bash

echo "Constructing the bipartite graph representation for the Synthetic Benchmark..."

cd graph_construction/
python main.py \
-id ../DATA/synthetic_benchmark/ \
-od combined_graphs_output/synthetic_benchmark_bipartite/ \
--input_data_file_type csv \
--graph_type bipartite
cd ..

echo "Computing BC scores on the bipartite graph representation..."

cd network_analysis/
python main.py \
-g ../graph_construction/combined_graphs_output/synthetic_benchmark_bipartite/bipartite/bipartite.graph \
-o output/synthetic_example_bipartite_cell_nodes/ \
--betweenness_mode exact \
--betweenness_source_target_nodes cell_nodes
cd ..