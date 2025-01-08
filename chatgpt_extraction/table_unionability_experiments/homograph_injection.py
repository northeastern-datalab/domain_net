import pandas as pd
import os
import argparse
import random

from tqdm import tqdm
from collections import defaultdict
from pathlib import Path


def main(args):
    table_pairs_df=pd.read_csv(args.table_pairs_path)
    query_tables=table_pairs_df['query_table'].unique()
    for q_table in tqdm(query_tables):
        # Ensure both table files exist
        if not os.path.isfile(args.input_queries_dir+q_table):
            continue 
        
        # Load the tables into dataframes
        T1=pd.read_csv(args.input_queries_dir+q_table, sep=';')
        T1 = T1.drop(T1.columns[0], axis=1)

        # Select columns for the homographs
        selected_cols=[]
        for i in range(args.num_homographs):
            selected_cols.append(random.sample(T1.columns.to_list(), args.homograph_num_meanings))

        ########## Inject The Homographs ##########
        
        # Number of rows to be modified for each homograph (usually one, can be tweaked)
        num_rows_per_homograph=1
        
        # Ensure we have enough rows in the table for the injected homographs
        assert(T1.shape[0] >= num_rows_per_homograph*args.num_homographs)
        
        row_idx=0
        for i in range(args.num_homographs):
            for _ in range(num_rows_per_homograph):
                T1[selected_cols[i]] = T1[selected_cols[i]].astype(object)
                T1.loc[row_idx, selected_cols[i]]='InjectedHomograph'+str(i)
                row_idx+=1
        
        T1.to_csv(args.output_dir+q_table, index=False)  
    

if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Inject Homographs over the Generated Unionability Dataset')

    parser.add_argument('--input_tables_dir', required=True, help='Path to the input directory containing the tables to be injected with homographs.')
    
    parser.add_argument('--input_queries_dir', required=True, help='Path to the input queries directory containing the query tables to be injected with homographs.')

    parser.add_argument('--table_pairs_path', required=True,
        help='Path to the file containing the ground truth table pairs for which table unionability will be tested')

    parser.add_argument('--output_dir', required=True, help='Path to the output directory where the tables with the injected homographs are stored.')

    parser.add_argument('--homograph_num_meanings', type=int, help='Number of meanings for injected homographs. The value must be at least 2')

    parser.add_argument('--num_homographs', type=int, help='Total number of homographs injected in a table. The value must be at least 1')

    parser.add_argument('--dynamic_num_homographs', 
        help='If specified then the number of homographs injected is a fixed proportion of the number of unique values in a pair of tables')
    
    parser.add_argument('--homograph_proportion', type=int, default=3, help='Specifies the percentage of unique values that will be injected as homographs')
    
    parser.add_argument('--seed', default=0, type=int, help='The seed used by any random number generator')

    # Parse the arguments
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    
    main(args)