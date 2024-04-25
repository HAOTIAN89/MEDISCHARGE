
import json
import os
import argparse
from tqdm import tqdm
from openai import AsyncOpenAI
import time
import tiktoken
import asyncio
import nest_asyncio
import pandas as pd
import sys

root_path = os.path.abspath(os.path.join(os.curdir, os.pardir, os.pardir))
os.chdir(root_path)
sys.path.append(root_path)

from src.utils.loading_saving import load_file, save_file
tqdm.pandas()

#from utils.prompts import advice_prompt, student_prompt, rec_prompt, rec_prompt_fast

KEY_PATH = 'src/inference/keys.json'
# MODEL = 'gpt-4-0125-preview'
MODEL = 'gpt-3.5-turbo-0125'

class GPTWrapper(): 
    '''
    A GPT wrapper serving as an interface to the OpenAI API.
    '''
    def __init__(self, model=MODEL, key_path=KEY_PATH):

        self.model = model
        self.max_tokens_per_batch = 5000000
        self.prompt = None
        self.max_tokens = 2000
        self.temperature = 0.7
        self.top_p = 0.95
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.stop = None
        self.key_path = key_path
        self.client = self.login_openai()
        self.cost_per_input_token = 10 / 1e6
        self.cost_per_output_token = 30 / 1e6
    
    def format_response(self, response):
        ''' Format response from OpenAI API. '''
        '''Default is identity function. Shouild be overrided when applies.'''
        return response

    def login_openai(self):
        ''' Login to OpenAI API. '''
        if not os.path.exists(self.key_path):
            raise ValueError(f'No OpenAI API key found at {self.key_path}. Please provide a valid path.') 
        with open(self.key_path) as f:
            api_key = json.load(f)['api_key']
        client = AsyncOpenAI(api_key=api_key)
        return client

    def load(self, path): 
        '''
        Load records from a json/jsonl file.
        '''
        print(f'Loading records from {path}.')
        if not os.path.exists(path):
            print(f'Could not load records from {path}.')
            return []
        elif '.jsonl' in path: 
            items = []
            with open(path, 'r') as f:
                for line in f:
                    items.append(json.loads(line))
            return items
        elif '.json' in path: 
            with open(path, "r") as f:
                items = json.load(f)
        else:
            raise ValueError('File must be json or jsonl.')
        return items

    def save(self, save_path, records):
        '''
        Save a record to a .jsonl file.
        '''
        if '.jsonl' in save_path:
            with open(save_path, 'a') as f:
                for record in records:
                    f.write(json.dumps(record) + '\n')
        elif '.json' in save_path:
            existing = self.load(save_path)
            existing.append(records)
            with open(save_path, 'w') as f:
                json.dump(existing, f, indent=4)
        else:
            raise ValueError('File must be json or jsonl.')
    
    def count_text_tokens(self, text):
        ''' Count number of tokens in text. '''
        try: 
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    
    def count_message_tokens(self, messages):
        ''' Count number of tokens in text. '''
        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += self.count_text_tokens(value)
                if key == "name":
                    num_tokens += tokens_per_name
            num_tokens += 3
        return num_tokens

    def estimate_input_cost(self, nb_tokens):
        ''' Estimate cost of text generation. 
        '''
        return (nb_tokens * self.cost_per_input_token)
    
    def estimate_output_cost(self, nb_tokens):

        ''' Estimate cost of text generation. 
        '''
        return (nb_tokens * self.cost_per_output_token)
    
    def new_batch(self, idx ,msg, msg_len):
        ''' Create a new batch. '''
        return {"batch": pd.DataFrame({'idx': [idx], 'messages': [msg]}), "total_nb_token": msg_len}

    def partition(self, all_messages):
        ''' Build most optimal partitions given max number of tokens, model and messages.
        arguments:
            - all_messages: pandas_df containing all messages to partition (idx, messages)
        returns:
            - list of dictionary ("batch": batch of messages as a pandas df (idx, message), 
                                "total_nb_token": total number of tokens in the batch)
        '''

        batches = []
        for _,idx,message in tqdm(all_messages.itertuples(), total=len(all_messages), desc="Partitioning messages"):
            nb_batches = len(batches)
            msg_len = self.count_message_tokens(message)

            if nb_batches == 0:
                batches.append(self.new_batch(idx, message, msg_len)) 
            else:
                current_partition = batches[-1]
                if (current_partition["total_nb_token"] + msg_len) <= self.max_tokens_per_batch:
                    current_partition["batch"] = pd.concat([current_partition["batch"], pd.DataFrame({'idx': [idx], 'messages': [message]})])
                    current_partition["total_nb_token"] += msg_len
                else:

                    batches.append(self.new_batch(idx, message, msg_len))
        
        return batches

    async def dispatch_openai_requests(
      self,      
      batch_of_messages):
        ''' 
        Multiple calls to chat. 
        Arguments: 
            batch_of_messages: batch of messages (in the form of list of dictionaries) to send to the chat
        
        Returns:
            corresponding batch of responses given by the async_ask function
        
        '''
        nb_done = 0 #for monitoring and verbose use only, may use it for safety savings
        async def one_call(messages: str):
            ''' One async call to ask_chat. '''
            nonlocal nb_done
            response = await self.async_ask(
                messages= messages)
            nb_done +=1
            print('.', end = "")
            if nb_done % 20 == 0: #informative prints
                print(f"{nb_done} calls done.")
            return response 
        
        responses = [one_call(x) for x in batch_of_messages] #multiple calls but yet uncalled
        responses_called = await asyncio.gather(*responses)
        return responses_called
  
    def build_messages(self, user_prompt, sys_prompt=None):
        ''' Build messages in the right format given system prompt and user prompt.
        Arguments:
            - user_prompt: user prompt (str)
            - sys_prompt: system prompt (str)'''
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages
 
    async def async_ask(self, messages):
        '''
        Chat with the OpenAI API. Only one call.
        Input: 
            - messages: a dictionary of messages {"role": "user", "content": "message"}. This is only one call!
        Output:
            - the response from the API, formatted if applies
        '''
        try: 
            response = await self.client.chat.completions.create(
                messages=messages,
                model=self.model
                #max_tokens=self.max_tokens,
                #temperature=self.temperature,
                #top_p=self.top_p,
                #frequency_penalty=self.frequency_penalty,
                #presence_penalty=self.presence_penalty,
                #stop=self.stop
            )
        except Exception as e:
            print(f'Error during generation: {e}')
            print(f'Messages: {messages}')
            return ''
        
        response = response.choices[0].message.content
        
        try: 
            response = self.format_response(response)
        except Exception as e:
            print(f'Error formatting response: {e}')
            print(f'Messages: {messages}')
            print(f'Response: {response}')
            return response
        return response

    def generate_answers(self, user_prompts, save_path, sys_prompt=None):
        ''' Generate answers given a user prompt. 
        Arguments:
            - user_prompts: pd dataframe of user prompts idx,prompt (int,str), will be splitted in batches
            - sys_prompt: system prompt (str)
            - save_path: path to store the answers (str)
        Returns:
            - answers_df : dataframe of answers idx,prompt,answer (int,str,str)
        '''

        #Load what was already done
        if os.path.exists(save_path):
            answers = load_file(save_path)
            print(f"Loaded {answers.shape[0]} answers from {save_path}. Resuming generation.")
            if answers.shape[0] == user_prompts.shape[0]:
                print("All prompts have already been answered.")
                return answers
        else:
            answers = pd.DataFrame(columns=["idx", "prompt", "answer", "gold"])
        
        #remove whats already done
        done_idx = answers["idx"].tolist()
        user_prompts = user_prompts[~user_prompts["idx"].isin(done_idx)].reset_index(drop=True)
        
        user_prompts['messages'] = user_prompts['prompt'].apply(lambda x: self.build_messages(x, sys_prompt))
        batches = self.partition(user_prompts[['idx', 'messages']])

        for i, batch_ in enumerate(batches):
            batch = batch_["batch"]['messages']
            idxs = batch_["batch"]['idx']
            nb_tokens = batch_["total_nb_token"]
            print(f"Batch {i+1}/{len(batches)}: {batch.shape[0]} calls, {nb_tokens} total tokens, estimated input cost: {self.estimate_input_cost(nb_tokens)}$")
            loop = asyncio.get_event_loop()
            nest_asyncio.apply()
            start_time = time.time()
            current_answers = loop.run_until_complete(self.dispatch_openai_requests(batch))

            current_answers = pd.DataFrame({'idx': idxs,'answer': current_answers})
            current_answers = current_answers.merge(user_prompts[['idx', 'prompt', 'gold']], on='idx', how='left')
            answers = pd.concat([answers, current_answers], axis=0)
            save_file(answers, save_path)

            if i < len(batches) - 1:
                time_to_wait = max(5, 60 - (time.time() - start_time))
                print(f"\nWaiting {time_to_wait} seconds before next batch.")
                time.sleep(time_to_wait)
        
        return answers
            

    def estimate_all_cost(self, user_prompts, sys_prompt=None):
        ''' Estimate cost of generating answers for all user prompts. '''
        all_messages = user_prompts['prompt'].progress_apply(lambda x: self.build_messages(x, sys_prompt))
        input_tokens_counts = all_messages.progress_apply(self.count_message_tokens)
        input_tokens_count = input_tokens_counts.sum()

        output_tokens_counts = user_prompts['gold'].progress_apply(self.count_text_tokens)
        output_tokens_count = output_tokens_counts.sum()

        return {'input_token_count': input_tokens_count, 'input_token_cost': self.estimate_input_cost(input_tokens_count),
                'output_token_count': output_tokens_count, 'output_token_cost': self.estimate_output_cost(output_tokens_count)}
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path', type=str, required=True, help='Path to the input data file.')
    parser.add_argument(
        '--output_path', type=str, required=True, help='Path to the output data file.')
    parser.add_argument(
        '--sys_prompt_path', type=str, required=False, default=None, help='Path to the system prompt file.')


    args = parser.parse_args()

    #check paths
    if not os.path.exists(args.input_path):
        raise ValueError(f'Input file not found at {args.input_path}.')
    if args.sys_prompt_path is None or not os.path.exists(args.sys_prompt_path):
        sys_prompt = None
    else:
        with open(args.sys_prompt_path, 'r') as f:
            sys_prompt = f.read()
    
    #load data
    input_data_df = load_file(args.input_path)

    #initialize GPTWrapper
    gpt = GPTWrapper()

    #generate answers
    answers_df = gpt.generate_answers(input_data_df, args.output_path, sys_prompt=sys_prompt)