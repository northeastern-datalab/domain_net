import argparse
import itertools
import os
import pandas as pd

from pathlib import Path
from tqdm import tqdm

# prompt_part_1='Are the following tables unionable? Answer in the following format:\nUnionable: {yes/no}\n\n'
prompt_part_1='''Answer with the following JSON: {"unionable": "True"} the tables are unionable {"unionable": "False"} if they are not.\n\n'''

few_shot_example='''
{'icl_ind_0': 'Table:\nNatural Community|Uplands|Open Uplands|Shale cliff and talus community|Shale Cliff and Talus Community|2000|not applicable\nTable:\n16-0242|04/13/2016 12:00:00 AM|DateofDeath|38.0|Male|BERLIN|HARTFORD|Hospital|Substance abuse|Parking Lot|"Complications of Acute Fentanyl and Acetyl Fentanyl Toxicity| cocaine use"\nUnionable: no',
 'icl_ind_1': 'Table:\nNatural Community|Uplands|Open Uplands|Shale cliff and talus community|Shale Cliff and Talus Community|2000|not applicable\nTable:\n16-0719|10/23/2016 12:00:00 AM|DateofDeath|30.0|Male|ANSONIA|NEW HAVEN|Residence|Substance Abuse|Church|Acute Intoxication due to the Combined Effects of Fentanyl and Alcohol\nUnionable: no',
 'icl_ind_2': 'Table:\nNatural Community|Uplands|Open Uplands|Shale cliff and talus community|Shale Cliff and Talus Community|2000|not applicable\nTable:\n16-0899|12/24/2016 12:00:00 AM|DateofDeath|STRATFORD|FAIRFIELD|Hospital|Ingestion|Residence|"Acute Intoxication Combined Effects of Clonazepam. Alprazolam| Methadone| and Amphetamine"|"Bridgeport| CT\nUnionable: no',
 'icl_ind_3': 'Table:\nNatural Community|Uplands|Open Uplands|Shale cliff and talus community|Shale Cliff and Talus Community|2000|not applicable\nTable:\n13-0315|09/21/2013 12:00:00 AM|DateofDeath|MILFORD|NEW HAVEN|Hospital|Substance Abuse|Unknown|Acute intoxication due to the combined effects of heroin and ethanol|"Milford| CT\nUnionable: no',
 'icl_ind_4': 'Table:\nRegistered Party|Liberal Party of Canada|Businesses / Commercial organizations|ST. Lawrence Cement Inc|H3R 1H1\nTable:\n16-0242|04/13/2016 12:00:00 AM|DateofDeath|38.0|Male|BERLIN|HARTFORD|Hospital|Substance abuse|Parking Lot|"Complications of Acute Fentanyl and Acetyl Fentanyl Toxicity| cocaine use"\nUnionable: no'}\n\n
'''

def main(args):
    # Get list of the GT homographs
    table_pairs_df=pd.read_csv(args.table_pairs_path)
    for _, row in tqdm(table_pairs_df.iterrows()):
        query_table_path=args.input_queries_dir+row['query_table']
        datalake_table_path=args.input_tables_dir+row['data_lake_table']
        if not os.path.isfile(query_table_path) or not os.path.isfile(datalake_table_path):
            continue


        with open(query_table_path, 'r') as f:
            q_table_str= f.read()
        
        with open(datalake_table_path, 'r') as f:
            datalake_table_str= f.read()
        
        if args.few_shot_learning:
            gpt_str=prompt_part_1+few_shot_example+'\n\nTable1:\n'+q_table_str+'\n\nTable2:\n'+datalake_table_str
        else:
            gpt_str=prompt_part_1+'\n\nTable1:\n'+q_table_str+'\n\nTable2:\n'+datalake_table_str
        
        # Save the GPT query
        q_table_name=os.path.splitext(row['query_table'])[0]
        d_table_name=os.path.splitext(row['data_lake_table'])[0]
        with open(args.output_dir+q_table_name+'___'+d_table_name+'.txt', 'w') as f:
                f.write(gpt_str)
    


if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Query Construction to be used by LLMs')

    parser.add_argument('--table_pairs_path', required=True,
        help='Path to the file containing the ground truth table pairs for which table unionability will be tested')

    parser.add_argument('--input_queries_dir', required=True, help='Path to the input queries directory containing the query tables with injected homographs.')

    parser.add_argument('--input_tables_dir', required=True, help='Path to the input directory containing the other tables not containing homographs')

    parser.add_argument('--output_dir', required=True, help='Path to the output directory.')

    parser.add_argument('--few_shot_learning', action='store_true', help='If specified, then it adds the few shot examples into the prompt')

    # Parse the arguments
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)