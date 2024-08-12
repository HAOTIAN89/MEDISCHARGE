## MEDISCHARGE: An LLM System for Automatically Generating Discharge Summaries of Clinical Electronic Health Record
This is the official code repository for the ACL BioNLP 2024 workshop paper: MEDISCHARGE: An LLM System for Automatically Generating Discharge Summaries of Clinical Electronic Health Record (the link will be released soon)

## About MEDISCHARGE
MEDISCHARGE is an LLM-based system to generate Brief Hospital Course and Discharge Instruction summaries based on a patient’s Electronic Health Record. 

Our system is build on a Meditron-7B with context window extension, ensuring the system can handle cases of variable lengths with high quality. When the length of the input exceeds the system input limitation, we use a dynamic information selection framework to automatically extract important sections from the full discharge text. Then, extracted sections are removed in increasing order of importance until the input length requirement is met. We demonstrate our approach outperforms tripling the size of the context window of the model. 

Our system obtains a 0.289 overall score in the leaderboard, an improvement of 183% compared to the baseline, and a ROUGE-1 score of 0.444, achieving a second place performance in the shared task.

## Requirements
We used [Megatron-LLM codebase](https://github.com/epfLLM/Megatron-LLM) by EPFL LLM Team to distributely train our model so the requirements are the same as it.

```
transformers >= 4.31.0
torch >= 2.0.0
flash-attn >= 2.3.3
datasets >= 2.14.0
nltk >= 3.8.0
sentencepiece >= 0.1.0
git+https://github.com/facebookresearch/llama.git
```

For any other packages that may be required, please see the error messages and install accordingly.

## Dataset
Our dataset is 

## Run MEDISCHARGE System
In order to run our system to generate the Brief Hospital Course and Discharge Instruction summaries based on a patient’s Electronic Health Record, you should follow these instructions.

### Inference

### Evaluation

## Other Files
- `webapp/`: a simplified frondend to check the patient’s Electronic Health Records.
- `context-extension/`: ROPE experiments determine optimal context window expansion.
- `notebooks/`: dataset cleaning and preprocessing.

