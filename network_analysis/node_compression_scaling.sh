#!/bin/bash

output_dir="output/node_compression/scaling/education/"
graph_path="output/node_compression/scaling/education/"

# Sampling percentage settings
min_sampling_percentage=5
max_sampling_percentage=100
percentage_step_size=5

for sampling_percentage in `seq $min_sampling_percentage $percentage_step_size $max_sampling_percentage`; do

    # Run BC with weighted_sampling
    output_dir_local="${output_dir}""$sampling_percentage"_percent_nodes/
    python main.py -g "${output_dir}""$sampling_percentage"_percent_nodes/bipartite.pickle -o $output_dir_local \
    --betweenness_mode approximate \
    --node_compression --sampling_percentage $sampling_percentage --weighted_sampling \
    --seed 1
    
done
