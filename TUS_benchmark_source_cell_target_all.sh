#!/bin/bash

# echo "Constructing the bipartite graph representation for the TUS Benchmark..."

echo "Computing Approximate BC scores on the bipartite graph representation using 5000 samples..."
cd network_analysis/

num_runs=5
for cur_run in $(seq 1 $num_runs);
do
    seed=$cur_run 
    
    python main.py \
    -g ../graph_construction/combined_graphs_output/TUS/bipartite/bipartite.graph \
    -o "output/TUS_source_cell_target_all/seed$seed/" \
    --betweenness_mode approximate \
    --num_samples 5000 --groundtruth_path ground_truth/groundtruth_TUS.pickle \
    --betweenness_source_nodes cell_nodes \
    --betweenness_target_nodes all \
    --seed $seed
done

echo "Create top-k evaluation figures"
for cur_run in $(seq 1 $num_runs);
do
    seed=$cur_run 

    python TUS_topk_figures.py \
    --df_path "output/TUS_source_cell_target_all/seed$seed/graph_stats_with_groundtruth_df.pickle" \
    --save_dir figures/TUS_source_cell_target_all/seed$seed/
done

cd ..