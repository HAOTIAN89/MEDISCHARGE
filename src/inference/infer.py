'''
Inference pipeline.

'''

import torch
import os
import re
import argparse
import sys
import os
import time
import numpy as np
import pandas as pd
try:
    import vllm
except ImportError:
    print("")
import json as json
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from utils.data import *
from utils.chat import *

# ----------------------- Constants ----------------------- #

BOS_TOKEN, EOS_TOKEN = '<|im_start|>', '<|im_end|>'
TODO_VAL = -1
BATCH_SIZE = 4

# ----------------------- Inference parameters ----------------------- #

GREEDY_PARAMETERS = {
    'best_of': 1,
    'presence_penalty': 0.0,
    'frequency_penalty': 1.0,
    'top_k': -1,
    'top_p': 1.0,
    'temperature': 0.0,
    'stop': EOS_TOKEN,
    'use_beam_search': False,
    'max_tokens': 2048
}

PARAMETERS = {
    'meditron-7b': GREEDY_PARAMETERS
}

# ----------------------- Inference utilities ----------------------- #

def todo_list(data_df, gen_df, input_key, output_key, num_samples=None):
    '''
    Returns the list of samples to generate.

    :param data_df: pd.DataFrame, the input data
    :param gen_df: pd.DataFrame, the generated data
    :param input_key: str, remove samples for which the input key is None
    :param output_key: str, remove samples for which the output key has already been generated in gen_df
    :param num_samples: int, keep only the first num_samples samples (default: None --> all)
    :return: list, the list of indices to generate
    '''
    if input_key not in data_df.columns:
        raise ValueError(f'Input key {input_key} not found in input file.')
    valid_data = data_df[data_df[input_key].notnull()]
    idx_todo = valid_data['idx'].tolist()
    if num_samples and len(idx_todo) > num_samples:
        idx_todo = idx_todo[:num_samples]
    idx_done = gen_df[gen_df[output_key].notnull()]['idx'].tolist()
    idx_todo = [i for i in idx_todo if i not in idx_done]
    if len(idx_todo) == 0:
        raise ValueError(f'All samples already generated.')
    return idx_todo
    
def load_data(input_path, output_path, mode, num_samples=None):
    '''
    Loads the input data file and initializes the output data file. 
    '''
    data_df = load_file(input_path)
    print(f"\nLoaded data file...\n\tSamples: {data_df.shape[0]}\n\tColumns: {list(data_df.columns)}")
    if 'idx' not in data_df.columns:
        data_df['idx'] = data_df.index
    data_df = data_df.reset_index(drop=True)
    input_key, output_key = KEYS[mode]['input'], KEYS[mode]['output']
    if input_key not in data_df.columns:
        raise ValueError(f'Input key {input_key} not found in output file.')
    
    if os.path.exists(output_path):
        gen_df = load_file(output_path)
        print(f"Loading output file...\n\tSamples already generated: {gen_df.shape[0]}")
    else:
        print(f"Creating output file...\n\tPath: {output_path}\n\tColumns: {list(data_df.columns)}")
        gen_df = pd.DataFrame(columns = data_df.columns)
        gen_df[output_key] = TODO_VAL

    idx_todo = todo_list(data_df, gen_df, input_key, output_key, num_samples)
    print(f"\tSample left to generate: {len(idx_todo)}")
    data_df = data_df[data_df['idx'].isin(idx_todo)]
    return data_df, gen_df

def format_prompt(model_name, input, instructions):
    """
    Format prompt for inference with model-specific formatting.
    Models supported: meditron, llama, mistral. 
    """     
    if 'mistral' in model_name.lower():
        prompt = f"[INST]\n{input}[/INST]\n"
    elif 'llama' in model_name.lower():
        prompt = f"<s>[INST] <<SYS>>\n{instructions[0]}\n<</SYS>>\n\n{input}\n\n{instructions[1]} [/INST]"
    else: 
        prompt = f"{BOS_TOKEN}question\n{input}{EOS_TOKEN}\n{BOS_TOKEN}answer\n"

    return prompt

def infer_vllm(client, mode, prompt):
    """
    Inference using the VLLM backend (offline mode). 
    Returns the output text.

    Reference: https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py

    :param client: vllm.LLM, the LLM offline generation engine to use for querying the VLLM backend
    :param mode: str, the mode to use for inference
    :param prompt: str, the prompt to generate from
    """
    sampling_params = vllm.SamplingParams(**PARAMETERS[mode])
    response = client.generate(prompt, sampling_params=sampling_params)
    if len(response) > 0:
        return [r.outputs[0].text for r in response]
    else:
        return response[0].outputs[0].text

def load_few_shot(train_path, shots=1):
    '''
    Load a few-shot example from the training data for direct-gpt inference.
    
    :param train_path: str, path to the training data file. If None --> no few-shot example.
    :param shots: int, number of few-shot examples to load
    '''
    if train_path is not None and shots > 0:
        print(f'Loading {shots}-shot exemplar from {train_path}...')
        train_df = load_file(train_path)
        sample = train_df.sample(shots)
        few_shot_prompt = f"Here are {shots} example(s) of patient-doctor conversations and their corresponding clinical notes.\n\n"
        for i in range(shots):
            dialogue = sample.iloc[i]['conversation']
            note = sample.iloc[i]['data']
            few_shot_prompt += f'Example {i+1}:\n\nConversation:\n\n{dialogue}\n\nClinical note:\n\n{note}\n\n'
    else: 
        few_shot_prompt = 'Your answer should consist in one or a few paragrpahs of text, not overstructured.'
    return few_shot_prompt + '\n\n'

class Timer(): 
    def __init__(self): 
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        lapse = self.end_time - self.start_time
        breaktime = max(int(60 - lapse) + 2, 5)
        print(f"Break for {breaktime} seconds.")
        time.sleep(breaktime)


# ----------------------- Summary inference ----------------------- #

def complete_json(text): 
    ''' 
    Format a (potentially partial) JSON string. 
    Removes the last character until the string is valid.
    '''
    json_string = text.replace('\n', '')
    while True:
        if not json_string:
            return None
        try:
            data = json.loads(json_string + '}')
        except json.decoder.JSONDecodeError:
            json_string = json_string[:-1]
            continue
        break
    return data

# ----------------------- Inference ----------------------- #

    
def infer(model_name,
          model_path, 
          input_path=None,
          output_path=None, 
          num_samples=None,
          verbose=False):
    '''
    Loads a model and generates clinical notes. 
    Can be used for either 
    - Generator: Generate clinical notes from patient summaries
    - Direct: Generate clinical notes from patient-doctor conversations

    Arguments: 
        - model_name: Name of the model to be loaded.
        - model_path: Path to the model and tokenizer.
        - input_path: Path to the data file with columns.
        - output_path: Path to the output file with generated notes.
        - num_samples: Number of samples to generate (default: None --> all)
    '''

    print(f"\n\n# ----- INFERENCE: mode = {mode}, model = {model_name} ----- #\n\n")
    instructions = INSTRUCTIONS[mode.replace('-gpt', '').replace('-gold', '')]
    input_key, output_key = KEYS[mode]['input'], KEYS[mode]['output']
    data_df, gen_df = load_data(input_path, output_path, mode, num_samples=num_samples)
    template = None if mode != 'summarizer' else load_template(template_path)
    batch_size = BATCH_SIZE if mode != 'summarizer' else 1
    inference_data = json.loads(data_df.to_json(orient='records'))
    data_loader = DataLoader(inference_data, batch_size=batch_size, shuffle=False)
    print(f"Created data loader\n\tBatches to generate: {len(data_loader)}\n\tBatch size: {batch_size}")

    print(f"Initializing vLLM client...")
    kwargs = {
        "model": model_path,
        "tokenizer": model_path,
        "trust_remote_code": True,
        "max_num_seqs": 2048,
        "tensor_parallel_size": torch.cuda.device_count(),
    }
    client = vllm.LLM(**kwargs)

    for batch in tqdm(data_loader, total=len(data_loader), position=0, leave=True):
        prompts = [format_prompt(model_name, input, mode, instructions) for input in batch[input_key]]
        answers = infer_vllm(client, mode, prompts)

        if verbose:
            for prompt, answer in zip(prompts, answers):
                print(f'\n\n### PROMPT:\n\n{prompt}')
                print(f'\n\n### ANSWER:\n\n{answer}')

        new_batch = pd.DataFrame(batch)
        new_batch[output_key] = answers
        gen_df = pd.concat([gen_df, new_batch], ignore_index = True)
        save_file(gen_df, output_path, mode='w')
            

# ----------------------- Main ----------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', 
                        type=str, 
                        required=True,
                        help='Model name to be loaded.')
    parser.add_argument('--model_path', 
                        type=str, 
                        default=None,
                        help='Path to the model.')
    parser.add_argument('--input_path', 
                        type=str, 
                        required=True,
                        help='Path to the data file.')
    parser.add_argument('--output_path', 
                        type=str,
                        required=True,
                        help='Path to the output file with generated notes. ')
    parser.add_argument('--num_samples',
                        type=int,
                        default=None,
                        help='Number of samples to generate')
    parser.add_argument('--template_path',
                        type=str, 
                        default='data/template.json',
                        help='For summarizer mode only: path to the patient summary template.')
    parser.add_argument('--train_path',
                        type=str,
                        default=None,
                        help='Path to the training data file. \
                            Used to sample few-shot examples for direct-gpt inference.')
    parser.add_argument('--shots',
                        type=int,
                        default=0,
                        help='Number of few-shot examples for GPT inference. \
                            Must be provided with a train_path to sample exemplars.')
    parser.add_argument('--verbose',
                        type=int,
                        default=1,
                        help='Whether to print prompts and answers.')
    
    args = parser.parse_args()
        
    if args.dialogue: 
        print(f"Initializing vLLM client...")
        kwargs = {
            "model": args.model_path,
            "tokenizer": args.model_path,
            "trust_remote_code": True,
            "max_num_seqs": 2048,
            "tensor_parallel_size": torch.cuda.device_count(),
        }
        client = vllm.LLM(**kwargs)
        prompt = format_prompt('medinote', args.dialogue, 'direct', INSTRUCTIONS['direct'])
        answer = infer_vllm(client, 'direct', prompt)
        print(f'\n\n{answer}')

    
    else:
        infer(
            model_name=args.model_name,
            model_path=args.model_path,
            mode=args.mode,
            input_path=args.input_path,
            output_path=args.output_path,
            num_samples=args.num_samples,
            template_path=args.template_path,
            verbose=args.verbose
        )
