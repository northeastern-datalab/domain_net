#!/bin/bash

output_dir="output/node_compression/TUS/"
graph_path="../graph_construction/combined_graphs_output/TUS/bipartite/bipartite.graph"

# Sampling percentage settings
min_sampling_percentage=5
max_sampling_percentage=100
percentage_step_size=5

for sampling_percentage in `seq $min_sampling_percentage $percentage_step_size $max_sampling_percentage`; do

    # Run BC with weighted_sampling
    output_dir_local="${output_dir}"weighted_sampling/sampling_"$sampling_percentage"_percent/
    python main.py -g $graph_path -o $output_dir_local \
    --betweenness_mode approximate \
    --node_compression --sampling_percentage $sampling_percentage --weighted_sampling \
    --seed 1 

    # Run BC with unweighted sampling
    output_dir_local="${output_dir}"unweighted_sampling/sampling_"$sampling_percentage"_percent/
    python main.py -g $graph_path -o $output_dir_local \
    --betweenness_mode approximate \
    --node_compression --sampling_percentage $sampling_percentage \
    --seed 1 
    
done
