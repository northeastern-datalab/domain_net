#!/bin/bash

# Run the cardinality injection pipeline experiments

# vals_per_homograph=8
with_cleaning=false

# Cardinality parameters
cardinality_group_size=100
start_cardinality=0
final_cardinality=500

# Num samples for betweenness
num_samples=5000

# Cleaning parameters
min_str_length=4

# Calculate the BC scores for each injected dataset
cur_cardinality=$start_cardinality
while (($cur_cardinality <= $final_cardinality));
do
    min_cardinality=$cur_cardinality
    inner_dir_name=cardinality_"$min_cardinality"_infty/
    output_dir_main=output/TUS_injected_homographs/cardinality_experiments/"$inner_dir_name"
    graph_dir=../graph_construction/combined_graphs_output/TUS_injected_homographs/cardinality_experiments/"$inner_dir_name"

    for cur_graph_dir in $graph_dir*/;
    do
        graph_path="${cur_graph_dir}"bipartite/bipartite.graph
        output_dir="${output_dir_main}$(basename $cur_graph_dir)/"

        if $with_cleaning; then
            # Calculate approximate betweeness for each dataset WITH cleaning
            python main.py -o $output_dir -g $graph_path \
            -bm approximate --num_samples $num_samples \
            --perform_cleaning --remove_numerical_vals --min_str_length $min_str_length
        else
            # Calculate approximate betweeness for each dataset WITHOUT cleaning
            python main.py -o $output_dir -g $graph_path \
            -bm approximate --num_samples $num_samples
        fi
    done

    # Update cardinality threshold for next itteration
    cur_cardinality=$((cur_cardinality+cardinality_group_size))
done



# Perform evaluation for the injected homograph datasets
betweenness_scores_df=output/TUS_no_homographs/graph_stats_with_groundtruth_df.pickle
cur_cardinality=$start_cardinality

while (($cur_cardinality <= $final_cardinality));
do
    min_cardinality=$cur_cardinality
    inner_dir_name=cardinality_"$min_cardinality"_infty/
    betweenness_scores_df_injected_dir_main=output/TUS_injected_homographs/cardinality_experiments/"$inner_dir_name"
    metadata_dir_main=../homograph_injection/metadata/cardinality_experiments/"$inner_dir_name"

    for cur_dir in $betweenness_scores_df_injected_dir_main*/;
    do
        betweenness_scores_df_injected="${betweenness_scores_df_injected_dir_main}"$(basename $cur_dir)/graph_stats_df.pickle
        output_dir="${betweenness_scores_df_injected_dir_main}"$(basename $cur_dir)/
        metadata_dir="${metadata_dir_main}"$(basename $cur_dir)/

        python homograph_injection_evaluation.py --betweenness_scores_df $betweenness_scores_df \
        --betweenness_scores_df_injected $betweenness_scores_df_injected \
        --metadata_dir $metadata_dir --output_dir $output_dir
    done

    # Update cardinality threshold for next itteration
    cur_cardinality=$((cur_cardinality+cardinality_group_size))
done