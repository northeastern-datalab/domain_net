#!/bin/bash

# Shell script to run a certain homograph injection configuration multiple times with different seeds

mode=random
graph=../graph_construction/combined_graphs_output/TUS_no_homographs/bipartite/bipartite.graph
value_stats_dict=value_stats_dict.pickle
input_dir=../DATA/table_union_search/csvfiles_no_homographs/

# Filtering parameters
min_str_length=4
max_str_length=50

# Cardinality parameters
cardinality_group_size=100
start_cardinality=500
final_cardinality=500

num_injected_homographs=50
num_values_replaced_per_homograph=2
num_runs=4

cur_cardinality=$start_cardinality
while (($cur_cardinality <= $final_cardinality));
do
    min_cardinality=$cur_cardinality
    inner_dir_name=cardinality_"$min_cardinality"_infty/
    output_dir=../DATA/table_union_search/TUS_injected_homographs/cardinality_experiments/"$inner_dir_name"
    metadata_output_dir=metadata/cardinality_experiments/"$inner_dir_name"

    for cur_run in $(seq 1 $num_runs);
    do
        seed=$cur_run 
        
        # Run the homograph_injection file
        python homograph_injection.py -id $input_dir -od "${output_dir}seed$seed/" -g $graph \
            --value_stats_dict $value_stats_dict \
            --filter --min_str_length $min_str_length --max_str_length $max_str_length \
            --min_cardinality $min_cardinality --remove_numerical_vals \
            --injection_mode $mode -nih $num_injected_homographs -nvr $num_values_replaced_per_homograph\
            --seed $seed --metadata_output_dir "${metadata_output_dir}seed$seed/"
    done

    # Update cardinality threshold for next itteration
    cur_cardinality=$((cur_cardinality+cardinality_group_size))
done