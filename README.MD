# Domain Net

## Repository Overview

- **DATA**: a folder that will contain all the input data (Instruction on how to download the data can be found at: https://github.com/northeastern-datalab/DomainNet-Datasets)
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