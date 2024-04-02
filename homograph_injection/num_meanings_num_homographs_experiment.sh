#!/bin/bash

# Shell script to construct the various number of homographs and number of meaning datasets used for comparison with D4

mode="random"
graph="../graph_construction/combined_graphs_output/TUS_no_homographs/bipartite/bipartite.graph"
value_stats_dict="value_stats_dict.pickle"
input_dir="../DATA/table_union_search/csvfiles_no_homographs/"

# Filtering parameters
min_str_length=4
max_str_length=50

# Cardinality parameters
min_cardinality=500

# General parameters
num_injected_homographs_list=(50 100 150 200)
num_values_replaced_per_homograph_list=(2 4 6)
seed=1

output_dir="../DATA/table_union_search/TUS_injected_homographs/num_meanings_num_homographs_experiments/"
metadata_output_dir="metadata/num_meanings_num_homographs_experiments/"


for num_values_replaced_per_homograph in ${num_values_replaced_per_homograph_list[@]}; do
    for num_injected_homographs in ${num_injected_homographs_list[@]}; do
        python homograph_injection.py -id $input_dir \
            -od "${output_dir}num_meanings_${num_values_replaced_per_homograph}/num_homographs_${num_injected_homographs}/" \
            -g $graph --value_stats_dict $value_stats_dict \
            --filter --min_str_length $min_str_length --max_str_length $max_str_length \
            --min_cardinality $min_cardinality --remove_numerical_vals \
            --injection_mode $mode -nih $num_injected_homographs -nvr $num_values_replaced_per_homograph --seed $seed \
            --metadata_output_dir "${metadata_output_dir}num_meanings_${num_values_replaced_per_homograph}/num_homographs_${num_injected_homographs}/"
    done
done