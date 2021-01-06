#!/bin/bash

echo "Constructing the bipartite graph representation for the TUS Benchmark..."

cd graph_construction/
python main.py \
-id ../DATA/table_union_search/csvfiles/ \
-od combined_graphs_output/TUS/ \
--input_data_file_type csv \
--graph_type bipartite
cd ..

echo "Computing Approximate BC scores on the bipartite graph representation using 5000 samples..."

cd network_analysis/
python main.py \
-g ../graph_construction/combined_graphs_output/TUS/bipartite/bipartite.graph \
-o output/TUS/ \
--betweenness_mode approximate \
--num_samples 5000 --groundtruth_path ground_truth/groundtruth_TUS.pickle


echo "Create top-k evaluation figures"
python TUS_topk_figures.py \
--df_path output/TUS/graph_stats_with_groundtruth_df.pickle \
--save_dir figures/TUS/

cd ..