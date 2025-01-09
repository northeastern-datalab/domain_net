#!/bin/bash

# Shell script to generate the gpt queries for the unionability experiments

# homograph_num_meanings_list=(0 2 4 6 8)
homograph_num_meanings_list=(0)
queries_dir=ugen_v2/gpt_queries/
output_dir=ugen_v2/gpt_output/

for homograph_num_meanings in "${homograph_num_meanings_list[@]}"; do
    python unionability_query_chatgpt.py --queries_dir "${queries_dir}zero_shot/num_meanings_${homograph_num_meanings}/" --output_dir "${output_dir}zero_shot/num_meanings_${homograph_num_meanings}/"
done
