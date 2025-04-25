## MEDISCHARGE: An LLM System for Automatically Generating Discharge Summaries of Clinical Electronic Health Record
This is the official code repository for the ACL BioNLP 2024 workshop paper: MEDISCHARGE: An LLM System for Automatically Generating Discharge Summaries of Clinical Electronic Health Record [[Paper]](https://aclanthology.org/2024.bionlp-1.61/)

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
Our dataset is from [MIMIC-IV](https://physionet.org/content/mimiciv/3.0/), covering 109,168 Emergency Department (ED) visits. Each patient visit record encompasses several key components: the chief complaints logged by the ED, diagnosis codes (either ICD-9 or ICD-10), at least one radiology report, and a comprehensive discharge summary. 

## Run MEDISCHARGE System
In order to run our system to generate the Brief Hospital Course and Discharge Instruction summaries based on a patient’s Electronic Health Record, you should follow these instructions.

### Inference
In the `src/inference` folder:
- `BHC_construct.sh`: construct the patient’s Electronic Health Records into prompt for generating Brief Hospital Course.
- `DI_construct.sh`: construct the patient’s Electronic Health Records into prompt for generating Discharge Instruction.
- `submit.sh`: run the BHC or DI inference based on the constructed prompts.
- `submit_all.sh`: combine the generated BHC and DI as one csv file and reorder the index to satisfy the submission requirement(optional)
  
### Evaluation
In the `src/inference` folder:
- `run_experiment.sh`: evaluate the genetation in one time based on six metrics(BLEU, ROUGE, BERTScore, METEOR, AlignScore&MEDCON)
  
## Other Files
- `webapp/`: a simplified frondend to check the patient’s Electronic Health Records.
- `context-extension/`: ROPE experiments determine optimal context window expansion.
- `notebooks/`: dataset cleaning and preprocessing.

