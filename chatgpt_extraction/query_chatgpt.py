import openai
import os
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


    # Populate the `per_homograph_pairs_queries_dir` by combining the prompt with the constructed query
    print("Populating the per_homograph_pairs_queries_dir...")
    for hom in tqdm(os.listdir(args.per_homograph_pairs_queries_dir)):
        Path(args.constructed_queries_dir+hom).mkdir(parents=True, exist_ok=True)
        for pair in os.listdir(args.per_homograph_pairs_queries_dir+hom):
            with open(args.per_homograph_pairs_queries_dir+hom+'/'+pair, 'r') as f:
                contents= f.read()
            gpt_str=prompt_part_1+'"'+hom+'"'+prompt_part_2+contents
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

            print(response)
            exit()


if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Query ChatGPT to idenfity if a valuye')

    parser.add_argument('--per_homograph_pairs_queries_dir', required=True, 
        help='Path to where the query files for each homograph table pairs are stored')

    parser.add_argument('--constructed_queries_dir', required=True,
        help='Path to the directory that contains the queries to sent to chatGPT verbatum')

    parser.add_argument('--output_dir', required=True,
        help='Path to the directory that the chatgpt responses will be stored.')

    # Parse the arguments
    args = parser.parse_args()

    print('\nPer Homograph Pairs Queries Directory:', args.per_homograph_pairs_queries_dir)
    print('Constructed Queries Directory:', args.constructed_queries_dir)
    print('Output Directory', args.output_dir)
    print('\n')

    Path(args.constructed_queries_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    load_dotenv()
    main(args)