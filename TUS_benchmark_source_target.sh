#!/bin/bash

# echo "Constructing the bipartite graph representation for the TUS Benchmark..."

echo "Computing Approximate BC scores on the bipartite graph representation using 5000 samples..."
cd network_analysis/

# Set up source/target parameters must be one of ['all', 'cell_nodes', 'attribute_nodes']
num_runs=5
source_nodes=cell_nodes
target_nodes=cell_nodes
output_dir_name=TUS_source_cell_target_cell


for cur_run in $(seq 1 $num_runs);
do
    seed=$cur_run 
    
    python main.py \
    -g ../graph_construction/combined_graphs_output/TUS/bipartite/bipartite.graph \
    -o output/$output_dir_name/seed$seed/ \
    --betweenness_mode approximate \
    --num_samples 5000 --groundtruth_path ground_truth/groundtruth_TUS.pickle \
    --betweenness_source_nodes $source_nodes \
    --betweenness_target_nodes $target_nodes \
    --seed $seed
done

echo "Create top-k evaluation figures"
for cur_run in $(seq 1 $num_runs);
do
    seed=$cur_run 

    python TUS_topk_figures.py \
    --df_path output/$output_dir_name/seed$seed/graph_stats_with_groundtruth_df.pickle \
    --save_dir figures/$output_dir_name/seed$seed/
done

cd ..