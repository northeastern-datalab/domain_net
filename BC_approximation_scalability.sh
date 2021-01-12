#!/bin/bash

cd network_analysis/

echo "Running BC approximation scalability on the synthetic benchmark ..."

python betweenness_approximation.py \
-g ../graph_construction/combined_graphs_output/synthetic_benchmark_bipartite/bipartite/bipartite.graph \
-df output/synthetic_example_bipartite/graph_stats_df.pickle \
-o output/synthetic_example_bipartite/ --sample_size 1 250 1

echo "Running BC approximation scalability on the TUS benchmark ..."

python betweenness_approximation.py \
-g ../graph_construction/combined_graphs_output/TUS/bipartite/bipartite.graph \
-df output/TUS/graph_stats_with_groundtruth_df.pickle \
-o output/TUS/ --sample_size 100 5000 100