# Domain Net

## Repository Overview

- **DATA**: a folder that will contain all the input data (Instructions on how to download the data can be found at: https://github.com/northeastern-datalab/DomainNet-Datasets).
 Once you have downloaded the (raw tables) data, unzip it and create a `DATA/` directory and insert all the extracted data into it.
- **graph_construction**: module to construct a graph representation given a repository of tables
- **homograph_injection**: module to artificially inject homographs in a repository 
- **network_analysis**: module to run network centrality measures on our graph representation and provide a score for each value in the repository.

## Setup

1. Clone the repo
2. CD to the repo directory. Create and activate a virtual environment for this project  
  * On macOS or Linux:
   ```
   python3 -m venv env
   source env/bin/activate
   which python
   ```
3. Install necessary packages
   ```
   pip install -r requirements.txt
   ```
We recommend using python version 3.8.

If there are issues when installing the networkit package make sure you have installed g++ and python3-dev on your machine.
On Ubuntu this can be done with the following commands:
```
sudo apt-get install python3-dev 
sudo apt install g++
```


## Reproducibility

### Synthetic Benchmark (SB)
To reproduce our results and analysis on the synthetic benchmark run the `synthetic_benchmark.sh` script.
You can do that by running:
```
chmod +x synthetic_benchmark.sh && ./synthetic_benchmark.sh
```

The script will produce the bipartite graph representation for the synthetic benchmark and then calculate the BC scores for every node in that graph.
Finally open and run all cells in the [synthetic_benchmark_analysis.ipynb](network_analysis/synthetic_benchmark_analysis.ipynb) jupyter notebook file to see the analysis and produced figures.

### Table Union Search (TUS) Benchmark
To reproduce our results and analysis on the table union search (TUS) benchmark run the `TUS_benchmark.sh` script.
You can do that by running:
```
chmod +x TUS_benchmark.sh && ./TUS_benchmark.sh
```

The script will produce the bipartite graph representation for the synthetic benchmark and then calculate the approximate BC scores for every node in that graph.
5000 nodes are used for sampling.
Finally the precision/recall/f1-score curves at various top-k values will be produced and can be found in the [network_analysis/figures/TUS/](network_analysis/figures/TUS/) directory.

### Table Union Search Injected (TUS-I) Experiments
To run the cardinality experiments with injected homographs run the `TUS_injection_cardinality.sh` script.
You can do that by running:
```
chmod +x TUS_injection_cardinality.sh && ./TUS_injection_cardinality.sh
```
Please be patient when running the script as it can take a long time (2-3 hours) since generating the injected datasets is a slow process.

The script will produce a set of datasets with injected homographs that were introduced by replacing values of varying cardinalities.
For each allowed cardinality range there 4 runs with 4 different seeds.
The injected datasets can be found in the `DATA/table_union_search/TUS_injected_homographs/cardinality_experiments` directory.

Once the dataset generation is complete, a bipartite graph representation is generated for each dataset and then the approximate BC scores are computed for all nodes.
The analysis, evaluation and relevant figures of the experiment can be found by running all the cells in the [homograph_injection_analysis.ipynb](network_analysis/homograph_injection_analysis.ipynb) jupyter notebook.

### BC approximation scalability Experiments
To run the BC approximation scalability experiments run the `BC_approximation_scalability.sh` script.
You can do that by running:
```
chmod +x BC_approximation_scalability.sh && ./BC_approximation_scalability.sh
```

The script will run approximate BC with different sample sizes on the synthetic benchmark (SB) and the TUS benchmark.
Once the script is finished, you can see the results and evaluation of the approximation by running all the cells in the [betweenness_approximation.ipynb](network_analysis/betweenness_approximation.ipynb) jupyter notebook.

## How to use with your own data
Domain Net follows a two step process where (1) converts a set of tables into a graph representation and (2) using the constructed graph calculates the betweenness centrality (BC) of every data value in that graph.

* **Graph Construction** is handled by the [graph_construction/](graph_construction/) module.
Given a set of csv or tsv files in a specified directory a bipartite graph representation is constructed.
The first row in the csv or tsv files is considered as a header row and its values are not considered as data values.
If you know that your data has no header rows then inject an artificial dummy row as the first row of your data to act as a header.
To construct the bipartite graph representation follow the instructions and examples in the graph_construction/ [readme](graph_construction/readme.md)

* **BC computation** is handled by the [network_analysis/](network_analysis/) module.
Given the graph representation as input from the previous step it will calculate the BC scores for each data value node in the graph.
Nodes with higher BC scores are more likely to be homographs than data values with lower BC scores.
To calculate the BC scores and set if an exact or approximate computation needs to be run follow the instructions and examples in the network_analysis/ [readme](network_analysis/readme.md).  
