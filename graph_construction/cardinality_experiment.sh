#!/bin/bash

# Shell script to construct multiple graph representations from a set of directories for the TUS-I cardinality experiments

input_dir_main=../DATA/table_union_search/TUS_injected_homographs/cardinality_experiments/
output_dir_main=combined_graphs_output/TUS_injected_homographs/cardinality_experiments/
input_data_file_type=csv
graph_type=bipartite

for input_dir in $input_dir_main*/;
do
    for input_dir_inner in $input_dir*/;
    do 
    output_dir="${output_dir_main}$(basename $input_dir)/$(basename $input_dir_inner)"/

    python main.py -id $input_dir_inner -od $output_dir \
    --input_data_file_type $input_data_file_type --graph_type $graph_type
    done
done