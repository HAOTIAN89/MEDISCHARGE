
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

#from utils.prompts import advice_prompt, student_prompt, rec_prompt, rec_prompt_fast

KEY_PATH = 'utils/keys.json'
# MODEL = 'gpt-4-0125-preview'
MODEL = 'gpt-3.5-turbo-0125'

class GPTWrapper(): 
    '''
    A GPT wrapper serving as an interface to the OpenAI API.
    '''
    def __init__(self, model=MODEL, key_path=KEY_PATH):

        self.model = model
        self.max_tokens_per_batch = 2000
        self.prompt = None
        self.max_tokens = 2000
        self.temperature = 0.7
        self.top_p = 0.95
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.stop = None
        self.key_path = key_path
        self.client = self.login_openai()
        self.cost_per_k_tokens = 1
    
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

    def count_tokens(self, messages):
        ''' Count number of tokens in text. '''
        tokens_per_message = 3
        tokens_per_name = 1
        try: 
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens
    
    def estimate_cost(self, nb_tokens):
        ''' Estimate cost of text generation. 
        '''
        return (nb_tokens * self.cost_per_k_tokens) / 1000
    
    def new_batch(self, msg, msg_len):
        ''' Create a new batch. '''
        return {"batch": pd.Series([msg]), "total_nb_token": msg_len}

    def partition(self, all_messages):
        ''' Build most optimal partitions given max number of tokens, model and messages.
        arguments:
            - all_messages: pandas_series containing all messages to partition
        returns:
            - list of dictionary ("batch": batch of messages as a pandas serie, 
                                "total_nb_token": total number of tokens in the batch)
        '''

        batches = []
        for _,message in tqdm(all_messages.items(), total=len(all_messages), desc="Partitioning messages"):
            nb_batches = len(batches)
            msg_len = self.count_tokens(message)

            if nb_batches == 0:
                batches.append(self.new_batch(message, msg_len)) 
            else:
                current_partition = batches[-1]
                if current_partition["total_nb_token"] + msg_len <= self.max_tokens_per_batch:
                    current_partition["batch"] = pd.concat([current_partition["batch"], pd.Series([message])])
                    current_partition["total_nb_token"] += msg_len
                else:
                    batches.append(self.new_batch(message, msg_len))
        
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
        ''' Build messages in the right format given system prompt and user prompt. '''
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
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stop=self.stop
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

    def generate_answers(self, user_prompts, sys_prompt=None):
        ''' Generate answers given a user prompt. 
        Arguments:
            - user_prompts: list of user prompts (list(str)), will be splitted in batches
            - sys_prompt: system prompt (str)
        Returns:
            - list of answers (list(str))
        '''

        #
        '''We may wanna implement restart here, in case of failure of the API.'''
        #

        answers = []
        

        all_messages = pd.Series(user_prompts).apply(lambda x: self.build_messages(x, sys_prompt))
        batches = self.partition(all_messages)

        for i, batch_ in enumerate(batches):
            batch = batch_["batch"]
            nb_tokens = batch_["total_nb_token"]
            print(f"Batch {i+1}/{len(batches)}: {batch.shape[0]} calls, {nb_tokens} total tokens, estimated cost: {self.estimate_cost(nb_tokens)}$")
            loop = asyncio.get_event_loop()
            nest_asyncio.apply()
            start_time = time.time()
            current_answers = loop.run_until_complete(self.dispatch_openai_requests(batch))
            answers.extend(current_answers)
            
            #some saving stuf here for the restart

            time_to_wait = max(5, 60 - (time.time() - start_time))
            print(f"\nWaiting {time_to_wait} seconds before next batch.")
            if i < len(batches) - 1:
                time.sleep(time_to_wait)
        
        return answers
            

class RecExtractor(GPTWrapper):
    def __init__(self, model=MODEL, key_path=KEY_PATH):
        super().__init__(model, key_path)
        self.prompt = rec_prompt_fast
        self.mode = 'rec_extraction_fast'

    def filter_guidelines(self, source_path, save_path):
        ''' Filter guidelines to remove already extracted guidelines. '''
        guidelines = self.load(source_path)
        print(f'Loaded {len(guidelines)} guidelines.')
        generated = self.load(save_path)
        generated_ids = set([g['id'] + '-' + 'q_type' for g in generated])
        if len(generated) > 0:
            print(f'Already extracted {len(generated_ids)} guidelines.')
        guidelines = [g for g in guidelines if g['id'] + '-' + self.mode not in generated_ids]
        return guidelines

    def generate(self, source_path, save_path, verbose=False):
        '''
        Extract a list of clinical recommendations
        from clinical practice guidelines.
        '''
        guidelines = self.filter_guidelines(source_path, save_path)

        for guid_idx, guideline in tqdm(enumerate(guidelines), desc=f'Current cost: {self.cost}'):
            user_prompt = self.prompt['user_prompt'].render(text=guideline['text'])
            if verbose: print(user_prompt)
            response = self.ask(user_prompt, sys_prompt=self.prompt['sys_prompt'], history=None)
            if verbose: print(response)
            recs = self.extract_recs(response)
            print(f'Extracted {len(recs)} recommendations.')
            for rec in recs:
                record = {
                    'id': guideline['id'],
                    'source': guideline['source'],
                    'q_type': self.mode,
                    'text': rec
                }
                self.save(save_path, record)

            if guid_idx > 5: 
                break
    
    async def async_generate(self, source_path, save_path, verbose=False):
        '''
        Extract a list of clinical recommendations
        from clinical practice guidelines.
        '''
        guidelines = self.filter_guidelines(source_path, save_path)

        # partition guidelines by batches
        partitions = self.partition(pd.DataFrame(guidelines), self.max_tokens)
        print(f'Partitioned guidelines into {len(partitions)} batches.')
        # [ [batch1], ..., [batch n] ]

        async def one_call(guideline): # 1 guideline --> 1 call
            ''' One async call to ask_chat. '''
            user_prompt = self.prompt['user_prompt'].render(text=guideline['text'])
            response = await self.async_ask(user_prompt, sys_prompt=self.prompt['sys_prompt'], history=None)
            recs = self.extract_recs(response)
            print(f'Extracted {len(recs)} recommendations.')
            for rec in recs:
                record = {
                    'id': guideline['id'],
                    'source': guideline['source'],
                    'q_type': self.mode,
                    'text': rec
                }
                self.save(save_path, record)
            return recs
        

        async_responses = [one_call(guideline) for guideline in guidelines] 
        new_responses = await asyncio.gather(*async_responses)
        return new_responses
    

    

    def extract_recs(self, response): 
        ''' 
        Extract recommendations from a response.
        '''
        recs = []
        try: 
            recs = json.loads(response)
        except Exception as e:
            print(f'Error extracting recommendations: {e}')
        return recs

class PatientExtractor(GPTWrapper):
    def __init__(self, model=MODEL, key_path=KEY_PATH):
        super().__init__(model, key_path)
        raise NotImplementedError('PatientExtractor not yet implemented.')
    
class StudentExtractor(GPTWrapper):
    def __init__(self, model=MODEL, key_path=KEY_PATH):
        super().__init__(model, key_path)
        raise NotImplementedError('StudentExtractor not yet implemented.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default='recommendation',
        help="Mode for extraction. Options: 'recommendation', 'patient', 'student'. Defaults to 'recommendation'."
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default='guidelines.jsonl',
        help="Data for scraped guidelines. Defaults to 'guidelines.jsonl'.")
    parser.add_argument(
        "--save_path",
        type=str,
        default='guidelines_QA.jsonl',
        help="Path to store extracted QA samples. Defaults to 'guidelines_QA.jsonl'.")
    parser.add_argument(
        "--keys_path",
        type=str,
        default='keys.json',
        help="Path to OpenAI API keys. Defaults to 'keys.json'.")
    args = parser.parse_args()
    
    extractors = {
        'recommendation': RecExtractor,
        'patient': PatientExtractor,
        'student': StudentExtractor
    }

    extractor = extractors[args.mode](args.keys_path)
    extractor.generate(args.source_path, args.save_path)
