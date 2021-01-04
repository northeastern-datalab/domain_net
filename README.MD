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