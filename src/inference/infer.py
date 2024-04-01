'''
Inference pipeline.

'''

import torch
import os
import re
import argparse
import sys
import os
import numpy as np
import pandas as pd
import vllm
import json as json
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)
from utils.loading_saving import load_file, save_file

# ----------------------- Constants ----------------------- #

BOS_TOKEN, EOS_TOKEN = '<|im_start|>', '<|im_end|>'
TODO_VAL = -1
BATCH_SIZE = 128

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
    'meditron-7b': GREEDY_PARAMETERS,
    'medischarge-7b-BHC': GREEDY_PARAMETERS,
    'medischarge-7b-DI': GREEDY_PARAMETERS
}

# ----------------------- Inference utilities ----------------------- #

def todo_list(data_df, gen_df, INPUT_KEY, GEN_OUTPUT_KEY, IDX_COL, num_samples=None):
    '''
    Returns the list of samples to generate.

    :param data_df: pd.DataFrame, the input data
    :param gen_df: pd.DataFrame, the generated data
    :param input_key: str, remove samples for which the input key is None
    :param num_samples: int, keep only the first num_samples samples (default: None --> all)
    :return: list, the list of indices to generate
    '''
    if INPUT_KEY not in data_df.columns:
        raise ValueError(f'Input key {INPUT_KEY} not found in input file.')
    valid_data = data_df[data_df[INPUT_KEY].notnull()]
    idx_todo = valid_data[IDX_COL].tolist()
    if num_samples and len(idx_todo) > num_samples:
        idx_todo = idx_todo[:num_samples]
    idx_done = gen_df[gen_df[GEN_OUTPUT_KEY].notnull()][IDX_COL].tolist()
    idx_todo = [i for i in idx_todo if i not in idx_done]
    if len(idx_todo) == 0:
        raise ValueError(f'All samples already generated.')
    return idx_todo
    
def load_data(input_path, INPUT_KEY, GEN_OUTPUT_KEY, IDX_COL, output_path, num_samples=None):
    '''
    Loads the input data file and initializes the output data file.
    Arguments:
        - input_path: str, path to the input data file
        - input_key: str, column name for the input data
        - output_key: str, column name for the output data
        - idx_col: str, column name for the index
        - output_path: str, path to the output data file
        - num_samples: int, number of samples to generate
    Returns:
        - data_df: pd.DataFrame, the input data
        - gen_df: pd.DataFrame, already generated samples (may be empty)
    '''
    data_df = load_file(input_path)
    print(f"\nLoaded data file...\n\tSamples: {data_df.shape[0]}\n\tColumns: {list(data_df.columns)}")
    if IDX_COL not in data_df.columns:
        data_df[IDX_COL] = data_df.index
    data_df = data_df.reset_index(drop=True)

    if INPUT_KEY not in data_df.columns:
        raise ValueError(f'Input key {INPUT_KEY} not found in output file.')
    
    if os.path.exists(output_path):
        gen_df = load_file(output_path)
        print(f"Loading output file...\n\tSamples already generated: {gen_df.shape[0]}")
    else:
        print(f"Creating output file...\n\tPath: {output_path}\n\tColumns: {list(data_df.columns)}")
        gen_df = pd.DataFrame(columns = data_df.columns)
        gen_df[GEN_OUTPUT_KEY] = TODO_VAL

    idx_todo = todo_list(data_df, gen_df, INPUT_KEY, GEN_OUTPUT_KEY, IDX_COL, num_samples)
    print(f"\tSample left to generate: {len(idx_todo)}")
    data_df = data_df[data_df[IDX_COL].isin(idx_todo)]
    return data_df, gen_df

def format_prompt(model_name, input):
    """
    Format prompt for inference with model-specific formatting.
    Models supported: meditron, llama, mistral. 
    """   
    inner_prompt = input  
    if 'mistral' in model_name.lower():
        prompt = f"[INST]\n{inner_prompt}[/INST]\n"
    elif 'llama' in model_name.lower():
        prompt = f"<s>[INST] <<SYS>>\n{inner_prompt} [/INST]"
    elif 'meditron' in model_name.lower():
        prompt = {inner_prompt}
    elif 'medischarge' in model_name.lower() :
        prompt = f"{BOS_TOKEN}question\n{inner_prompt}{EOS_TOKEN}\n{BOS_TOKEN}answer\n"
    else:
        raise ValueError(f'{model_name} is not a supported model name')

    return prompt

def infer_vllm(client, model_name, prompt):
    """
    Inference using the VLLM backend (offline mode). 
    Returns the output text.

    Reference: https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py

    :param client: vllm.LLM, the LLM offline generation engine to use for querying the VLLM backend
    :param prompt: str, the prompt to generate from
    """
    sampling_params = vllm.SamplingParams(**PARAMETERS[model_name])
    response = client.generate(prompt, sampling_params=sampling_params)
    if len(response) > 0:
        return [r.outputs[0].text for r in response]
    else:
        return response[0].outputs[0].text

# ----------------------- Inference ----------------------- #

def infer(model_name,
          model_path, 
          INPUT_KEY,
          GEN_OUTPUT_KEY,
          IDX_COL,
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

    print(f"\n\n# ----- INFERENCE: model = {model_name}, parameters = {PARAMETERS[model_name]} ----- #\n\n")
    data_df, gen_df = load_data(input_path, INPUT_KEY, GEN_OUTPUT_KEY, IDX_COL, output_path, num_samples=num_samples)
    batch_size = BATCH_SIZE
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
    print(f"vLLM client initialized")
    for batch in tqdm(data_loader, total=len(data_loader), position=0, leave=True):
        prompts = [format_prompt(model_name, input) for input in batch[INPUT_KEY]]
        answers = infer_vllm(client, model_name, prompts)

        if verbose:
            for prompt, answer in zip(prompts, answers):
                print(f'\n\n### PROMPT:\n\n{prompt}')
                print(f'\n\n### ANSWER:\n\n{answer}')

        new_batch = pd.DataFrame(batch)
        new_batch[GEN_OUTPUT_KEY] = answers
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
    parser.add_argument('--input_key',
                        type=str,
                        default='prompt',
                        help='Column name for the input data.')
    parser.add_argument('--gen_output_key',
                        type=str,
                        default='generated',
                        help='Column name for the output data.')
    parser.add_argument('--idx_col',
                        type=str,
                        default='idx',
                        help='Column name for the index.')
    parser.add_argument('--output_path', 
                        type=str,
                        required=True,
                        help='Path to the output file with generated notes. ')
    parser.add_argument('--num_samples',
                        type=int,
                        required=False,
                        default = None,
                        help='Number of samples to generate')
    parser.add_argument('--verbose',
                        type=int,
                        default=1,
                        help='Whether to print prompts and answers.')
    
    args = parser.parse_args()

    infer(
            model_name=args.model_name,
            model_path=args.model_path,
            INPUT_KEY = args.input_key,
            GEN_OUTPUT_KEY = args.gen_output_key,
            IDX_COL = args.idx_col,
            input_path=args.input_path,
            output_path=args.output_path,
            num_samples=args.num_samples,
            verbose=args.verbose
        )
