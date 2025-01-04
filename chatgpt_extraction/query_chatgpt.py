import openai
import os
import ast
import json
import argparse

from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv


prompt_part_1='Given the following 2 tables in CSV format can you determine if the value '
prompt_part_2=''' 
is a homograph or not? Only return a yes or no answer, no other accompaniying text in your response.
Return the JSON: {"is_homograph": "True"} if it is a homograph or return {"is_homograph": "False"} if its not a homograph.\n 
'''

in_context_learning_prompt_part_1='''Given 2 tables in CSV format and a target value your task is to determine if the target value is a homograph in the context of the two tables.
A value is a homograph if it has more than 1 meaning in the two tables, and is not a homograph if it only has one meaning in the two tables.
Only return a yes or no answer, no other accompanying text in your response.
Return the JSON: {"is_homograph": "True"} if it is a homograph or return {"is_homograph": "False"} if it's not a homograph.

To aid with this task consider the following 4 few-shot examples (each with 2 tables) where the target value "Jaguar" can appear either as a homograph or not.

Example 1:

Table 1:
Google,Panda,1M
Volkswagen,Puma,2M
BMW,Jaguar,0.9M
Amazon,Pelican,1.5M

Table 2:
Panda,Memphis,2
Panda,Atlanta,2
Lemur,National,20
Jaguar,San Diego,8
Pelican,Boston,0

Output for example 1 is {"is_homograph": "False"} since "Jaguar" has only 1 meaning in these two tables, it only appears as an animal.

Example 2:

Table 1:
XE,Jaguar,UK
Prius,Toyota,Japan
500,Fiat,Italy
BMW,X4,Germany

Table 2:
Jaguar,25.80,43224
Puma,4.64,13000
Apple,456,370B
Toyota,123,138B

Output for example 2 is {"is_homograph": "False"} since "Jaguar" has only 1 meaning in these two tables, it only appears as a car brand.

Example 3:

Table 1:
Google,Panda,1M
Volkswagen,Puma,2M
BMW,Jaguar,0.9M
Amazon,Pelican,1.5M

Table 2:
Jaguar,25.80,43224
Puma,4.64,13000
Apple,456,370B
Toyota,123,138B

Output for example 3 is {"is_homograph": "True"} since "Jaguar" has 2 meanings in these two tables. In the first table it refers to the animal but in the second table it refers to the car brand.

Example 4:

Table 1:
Panda,Memphis,2
Panda,Atlanta,2
Lemur,National,20
Jaguar,San Diego,8
Pelican,Boston,0

Table 2:
XE,Jaguar,UK
Prius,Toyota,Japan
500,Fiat,Italy
BMW,X4,Germany

Output for example 4 is {"is_homograph": "True"} since "Jaguar" has 2 meanings in these two tables. In the first table it refers to the animal but in the second table it refers to the car brand.

Query: In the following 2 tables is the value '''
in_context_learning_prompt_part_2= " a homograph?\n\n"

def get_reply(content, model='gpt-3.5-turbo-0125'):
    '''
    Sends a query to ChatGPT with the specified `content`

    Returns the response 
    '''
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": content}
        ],
    )
    reply=response.choices[0].message.content
    return json.loads(reply)


def main(args):
    client = openai.OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    if args.queried_homographs:
        per_homograph_pairs_queries=args.queried_homographs
    else:
        per_homograph_pairs_queries=list(os.listdir(args.per_homograph_pairs_queries_dir))

    # Populate the `per_homograph_pairs_queries_dir` by combining the prompt with the constructed query
    print("Populating the per_homograph_pairs_queries_dir...")
    for hom in tqdm(per_homograph_pairs_queries):
        Path(args.constructed_queries_dir+hom).mkdir(parents=True, exist_ok=True)
        for pair in os.listdir(args.per_homograph_pairs_queries_dir+hom):
            with open(args.per_homograph_pairs_queries_dir+hom+'/'+pair, 'r') as f:
                contents= f.read()
            if args.prompt_mode=='zero_shot':
                gpt_str=prompt_part_1+'"'+hom+'"'+prompt_part_2+contents
            elif args.prompt_mode=='few_shot':
                gpt_str=in_context_learning_prompt_part_1+'"'+hom+'"'+in_context_learning_prompt_part_2+contents
            with open(args.constructed_queries_dir+hom+'/' + pair, 'w') as f:
                f.write(gpt_str)

    # Loop over all constructed queries, and query ChatGPT
    print("\nQuerying chatGPT...")
    for hom in tqdm(os.listdir(args.constructed_queries_dir)):
        Path(args.output_dir+hom).mkdir(parents=True, exist_ok=True)
        for pair in os.listdir(args.constructed_queries_dir+hom):
            with open(args.constructed_queries_dir+hom+'/'+pair, 'r') as f:
                query= f.read()
            
            # Query ChatGPT
            try:
                response=get_reply(content=query)
                # Save response to `output_dir`
                with open(args.output_dir+hom+'/'+pair.split('.')[0]+'.json', "w") as outfile: 
                    json.dump(response, outfile, indent=4)
            except Exception as er:
                print(er)
            
if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Query ChatGPT to idenfity if a valuye')

    parser.add_argument('--per_homograph_pairs_queries_dir', required=True, 
        help='Path to where the query files for each homograph table pairs are stored')

    parser.add_argument('--constructed_queries_dir', required=True,
        help='Path to the directory that contains the queries to sent to chatGPT verbatum')

    parser.add_argument('--output_dir', required=True,
        help='Path to the directory that the chatgpt responses will be stored.')
    
    parser.add_argument('--prompt_mode', choices=['zero_shot', 'few_shot'], default='zero_shot',
        help='Prompting mode for chatGPT. If few-shot is used then the prompts include a set of examples demonstrating instances where a value is a homograph or not')

    parser.add_argument('--queried_homographs', nargs='+', help='If specified queries are sent only for the specified homographs.')

    # Parse the arguments
    args = parser.parse_args()

    print('\nPer Homograph Pairs Queries Directory:', args.per_homograph_pairs_queries_dir)
    print('Constructed Queries Directory:', args.constructed_queries_dir)
    print('Output Directory', args.output_dir)
    print('Prompt Mode:', args.prompt_mode)
    if args.queried_homographs:
        print("Querying GPT for homographs:", args.queried_homographs)
    print('\n')

    Path(args.constructed_queries_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    load_dotenv()
    main(args)