#!/bin/bash

# Synthetic Benchmark parameters
# output_dir="output/synthetic_example_bipartite/community_detection/"
# graph="../graph_construction/combined_graphs_output/synthetic_benchmark_bipartite/bipartite/bipartite.graph"
# groundtruth_path="ground_truth/synthetic_example_groundtruth_dict.pickle"

# TUS Benchmark parameters
output_dir="output/TUS/community_detection/"
graph="../graph_construction/combined_graphs_output/TUS/bipartite/bipartite.graph"
groundtruth_path="ground_truth/groundtruth_TUS_short_format.pickle"

python community_detection.py \
--output_dir $output_dir --graph $graph --groundtruth_path $groundtruth_path