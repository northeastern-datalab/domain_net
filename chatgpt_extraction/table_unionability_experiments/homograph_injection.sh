#!/bin/bash

# Shell script to inject homographs in the generated table unionability tables

seed=1
num_homographs=3
homograph_num_meanings_list=(0 2 4 6 8)
input_tables_dir=ugen_v2/datalake/
input_queries_dir=ugen_v2/query/
table_pairs_path=ugen_v2/groundtruth.csv
output_dir=ugen_v2/datalake_with_injected_homographs/

for homograph_num_meanings in "${homograph_num_meanings_list[@]}"; do
    python homograph_injection.py --input_tables_dir $input_tables_dir --input_queries_dir $input_queries_dir \
    --table_pairs_path $table_pairs_path --output_dir "${output_dir}num_meanings_${homograph_num_meanings}/" --homograph_num_meanings $homograph_num_meanings --num_homographs $num_homographs --seed $seed
done
