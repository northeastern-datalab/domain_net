# Homograph Injection

This module is used to inject artificial homographs into a repository of tables for the purpose of having a controlled environment for the number and nature of the homographs.
This is done by replacing 2 or different strings with a single unique string in the repository (i.e. the injected homograph).
For example, consider that the repository of tables includes the strings "USA" and "Panda" we can replace all their instances by a new string say "InjectedHomograph1".
Now "InjectedHomgraph1" has become an artificial homograph that we have injected.

## Running
To run the homograph injection procedure run the `homograph_injection.py` file by specifying the input output directories
as well as a path to the graph representation of the input directory.
The values to be replaced with the injected homographs can be specified manually or randomly with a specified seed number.
There are many more input line arguments that specify the conditions for a value to be replaced with the injected homograph.
For example, we can specify not to replace values with small cardinalities as well as specify how many meanings the injected homographs will have.

For more details on the available command line arguments you can run 
```
python homograph_injection.py -h
```

### Example 1: Inject 50 homographs in the TUS-I Benchmark
First create the bipartite graph representation for the TUS-I benchmark if you haven't already.
To do so first CD to the graph_construction/ directory and run
```
python main.py \
-id ../DATA/table_union_search/csvfiles_no_homographs/ \
-od combined_graphs_output/TUS_no_homographs/ \
--input_data_file_type csv \
--graph_type bipartite
```
The above command creates the bipartite graph representation for the TUS-I benchmark.
The graph representation is needed for the next step where we injected the homographs.

Now let us CD back to the homograph_injection/ directory and run the homograph injection procedure

```
python homograph_injection.py \
--input_dir ../DATA/table_union_search/csvfiles_no_homographs/ \
--output_dir ../DATA/table_union_search/TUS-I_50_homographs/ \
-g  ../graph_construction/combined_graphs_output/TUS_no_homographs/bipartite/bipartite.graph \
--value_stats_dict value_stats_dict.pickle \
--filter --min_str_length 4 --max_str_length 50 --min_cardinality 500 --remove_numerical_vals \
--injection_mode random \
--num_injected_homographs 50 \
--num_values_replaced_per_homograph 2 \
--seed 1
```
The above command line will inject 50 homograph in the dataset each with 2 meanings.
The injected homographs are produced by replacing 2 different random strings from different columns and they must have a length of at least 4 characters and at most 50 characters.
The minimum cardinality of the strings must be 500 and will ignore numerical values.

Once the script finishes running, a new repository is created as specified by the `output_dir` argument.
The the metadata regarding what values were replaced and their statistics can be found in the `metadata/` directory.  