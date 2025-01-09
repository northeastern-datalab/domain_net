#!/bin/bash

# Shell script to generate the gpt queries for the unionability experiments

homograph_num_meanings_list=(0 2 4 6 8)
input_tables_dir=ugen_v2/datalake/
input_queries_dir=ugen_v2/datalake_with_injected_homographs/
table_pairs_path=ugen_v2/groundtruth.csv
output_dir=ugen_v2/gpt_queries/

for homograph_num_meanings in "${homograph_num_meanings_list[@]}"; do
    python unionability_gpt_query_construction.py --input_tables_dir $input_tables_dir --input_queries_dir "${input_queries_dir}num_meanings_${homograph_num_meanings}/" \
    --table_pairs_path $table_pairs_path --output_dir "${output_dir}zero_shot/num_meanings_${homograph_num_meanings}/"

    python unionability_gpt_query_construction.py --input_tables_dir $input_tables_dir --input_queries_dir "${input_queries_dir}num_meanings_${homograph_num_meanings}/" \
    --table_pairs_path $table_pairs_path --output_dir "${output_dir}few_shot/num_meanings_${homograph_num_meanings}/" --few_shot_learning 
    
done
