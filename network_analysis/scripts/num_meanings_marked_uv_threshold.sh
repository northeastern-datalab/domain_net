# Script that runs number of meanings estimation with various thresholds for the bottom X percent of nodes that are marked as unambiguous

out_dir="../output/synthetic_example_large/threshold_experiment/complete_coverage/"
g_path="../../graph_construction/combined_graphs_output/synthetic_benchmark_large/bipartite/bipartite.graph"
df_path="../output/synthetic_example_large/graph_stats_with_groundtruth_df.pickle"
input_nodes_path="../input/synthetic_large/input.json"
seed=1

bottom_percent_low=2
bottom_percent_high=60
bottom_percent_delta=1

complete_coverage=true

for bottom_percent in `seq $bottom_percent_low $bottom_percent_delta $bottom_percent_high`;
do  
    out_loc_dir=$out_dir$bottom_percent/
    mkdir -p $out_loc_dir

    if $complete_coverage; then
        python ../semantic_type_propagation.py -o $out_loc_dir -g $g_path \
        -df $df_path --seed 1 \
        --input_nodes $input_nodes_path \
        --bottom_percent $bottom_percent --marked_unambiguous_values_complete_coverage
    else
        python ../semantic_type_propagation.py -o $out_loc_dir -g $g_path \
        -df $df_path --seed 1 \
        --input_nodes $input_nodes_path \
        --bottom_percent $bottom_percent
    fi
done