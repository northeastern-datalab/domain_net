import argparse
import pickle
import itertools
import os
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append('../network_analysis/')
import utils

def get_homograph_sample_from_table(file_path, col_name, homograph, sample_size=10, seed=0):
    '''
    Returns a pandas dataframe of the select sample for a given homograph from the specified file
    '''
    cur_df=pd.read_csv(file_path)
    hom_idx=cur_df[cur_df[col_name]==homograph].index.values[0]    
    sampled_indices = cur_df.sample(n=sample_size-1, random_state=seed).index.values.tolist()
    selected_indices=[hom_idx]+sampled_indices
    sample_df = cur_df.iloc[selected_indices]
    return sample_df


def main(args):
    # Get list of the GT homographs
    df=pd.read_pickle(args.df_path)
    homs=df[df['is_homograph']==True]['node'].tolist()
    
    # Construct dictionary of each homograph mapping to the list of table files and respective columns it is found 
    G = pickle.load(open(args.graph_path, "rb"))
    hom_to_info_dict={}
    for hom in homs:
        info_dicts=[]
        for attr in utils.graph_helpers.get_attribute_of_instance(G, hom):
            filename=G.nodes[attr]['filename']
            column_name=G.nodes[attr]['column_name']
            info_dicts.append({"filename": filename, "column_name": column_name})
        hom_to_info_dict[hom]=info_dicts
        
    # Construct the output files
    for hom in tqdm(hom_to_info_dict):
        # Extract a sample for each homograph from each file it is present in
        Path(args.output_dir+'per_homograph/'+hom+'/').mkdir(parents=True, exist_ok=True)
        for info_dict in hom_to_info_dict[hom]:
            sample_df = get_homograph_sample_from_table(
                file_path=args.files_dir+info_dict['filename'], col_name=info_dict['column_name'],
                homograph=hom, sample_size=args.sample_size
            )
            # Save the selected sample_df
            sample_df.to_csv(args.output_dir+'per_homograph/'+hom+'/'+info_dict['column_name']+'_'+info_dict['filename'], index=False, header=False)
        
        # Construct all pairwise queries for this homograph (i.e., all pairs filenames the homograph is present in)
        Path(args.output_dir+'all_pairs/'+hom+'/').mkdir(parents=True, exist_ok=True)
        filenames=os.listdir(args.output_dir+'per_homograph/'+hom+'/')
        filename_pairs=list(itertools.combinations(filenames, 2))
        
        for pair in filename_pairs:
            with open(args.output_dir+'per_homograph/'+hom+'/'+pair[0], 'r') as f:
                file1= f.read()
            with open(args.output_dir+'per_homograph/'+hom+'/'+pair[1], 'r') as f:
                file2= f.read()
            query_str='Table 1:\n' + file1 + '\n\nTable 2:\n' + file2
            
            out_file_name=pair[0].split('.')[0]+'___'+pair[1].split('.')[0]+'.csv'
            with open(args.output_dir+'all_pairs/'+hom+'/' + out_file_name, 'w') as f:
                f.write(query_str)


if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Query Construction to be used by LLMs')

    parser.add_argument('--df_path', required=True,
        help='Path to the graph_stats dataframe that contains ground truth information for node in the graph')

    parser.add_argument('--graph_path', required=True, help='Path to the networkx graph file.')

    parser.add_argument('--files_dir', required=True, help='Path to the directory containing the files')

    parser.add_argument('--output_dir', required=True, help='Path to the output directory.')

    parser.add_argument('--sample_size', default=10, type=int,
        help='Number of rows to sample from a table includes the row containing the homograph.')

    parser.add_argument('--seed', default=0, type=int, help='seed used for sampling')

    # Parse the arguments
    args = parser.parse_args()

    print('\nInput Dataframe path:', args.df_path)
    print('Graph path:', args.graph_path)
    print('Files directory:', args.files_dir)
    print('Output directory:', args.output_dir)
    print('Sample Size:', args.sample_size)
    print('Seed:', args.seed)
    print('\n')

    Path(args.output_dir+'per_homograph/').mkdir(parents=True, exist_ok=True)
    Path(args.output_dir+'all_pairs/').mkdir(parents=True, exist_ok=True)

    main(args)