import openai
import os
import re
import json
import argparse
import tiktoken

from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

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

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")

    # Loop over all constructed queries, and query ChatGPT
    print("\nQuerying chatGPT...")
    for q_file in tqdm(os.listdir(args.queries_dir)):
        with open(args.queries_dir+q_file, 'r') as f:
            query= f.read()
        
        # Query ChatGPT
        q_filename=os.path.splitext(q_file)[0]
        try:
            response=get_reply(content=query)
            # Save response to `output_dir`
            with open(args.output_dir+q_filename+'.json', "w") as outfile: 
                json.dump(response, outfile, indent=4)
        except Exception as er:
            print(er)
            
if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Query ChatGPT to idenfity if a valuye')

    parser.add_argument('--queries_dir', required=True, 
        help='Path to where the query prompts are stored')

    parser.add_argument('--output_dir', required=True,
        help='Path to the directory that the chatgpt responses will be stored.')
    
    # Parse the arguments
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    load_dotenv()
    main(args)