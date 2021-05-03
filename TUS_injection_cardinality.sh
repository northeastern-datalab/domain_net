#!/bin/bash

echo "Construct bipartite graph representation for the TUS with no homographs dataset"

cd graph_construction/
python main.py \
-id ../DATA/table_union_search/csvfiles_no_homographs/ \
-od combined_graphs_output/TUS_no_homographs/ \
--input_data_file_type csv \
--graph_type bipartite
cd ..

echo "Constructing the injection datasets..."

cd homograph_injection/
chmod +x cardinality_experiment.sh
./cardinality_experiment.sh
cd ..

echo "Construction graph representations for the injected datasets"

cd graph_construction/
chmod +x cardinality_experiment.sh
./cardinality_experiment.sh
cd ..

echo "Calculate BC scores for the TUS no homographs dataset"
cd network_analysis/

python main.py \
-g ../graph_construction/combined_graphs_output/TUS_no_homographs/bipartite/bipartite.graph \
-o output/TUS_no_homographs/ \
--betweenness_mode approximate \
--num_samples 5000 --groundtruth_path ground_truth/groundtruth_TUS.pickle \
--seed 1

echo "Calculate BC scores for each graph representation"
chmod +x cardinality_experiment.sh
./cardinality_experiment.sh

echo "Perform aggregate evaluation over all seeds for each cardinality size"
python homograph_injection_aggregate_evaluation.py \
-id output/TUS_injected_homographs/cardinality_experiments/ \
-od homograph_injection_evaluation/cardinality_experiment/