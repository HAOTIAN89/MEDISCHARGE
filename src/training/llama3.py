from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, Trainer, BitsAndBytesConfig
import torch
from datasets import load_dataset, Dataset
import pandas as pd
import argparse
import os
from pathlib import Path

# ====== PARAMETERS ====== #

MAX_SEQ_LENGTH = 8192
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ======================== #

# prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# ### Instruction:
# {}

# ### Input:
# {}

# ### Response:
# {}"""

# EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

def preprocess_function(data):
    instructions = data['prompt']
    responses = data['reference']
    return tokenizer(
        instructions,
        text_target=responses,
        return_tensors='pt',
        max_length=MAX_SEQ_LENGTH,
        padding=True,
        return_attention_mask=True,
        truncation=True,  # Enable truncation
    )


if __name__ == "__main__":
    
    
    args = argparse.ArgumentParser()
    
    args.add_argument("--model_path", type=str, default="/pure-mlo-scratch/make_project/spring2024/trial-runs/llama3-8b")
    args.add_argument("--max_seq_length", type=int, default=8192)
    args.add_argument('--output_dir', type=str, default='/pure-mlo-scratch/make_project/spring2024/trial-runs/llama3-8b')
    args.add_argument('--train_path', type=str, default='/pure-mlo-scratch/make_project/spring2024/llama3.jsonl')
    args.add_argument('--eval_path', type=str, default='/pure-mlo-scratch/make_project/spring2024/llama3.jsonl')
    args.add_argument('--epochs', type=int, default=3)
    args.add_argument('--batch_size', type=int, default=1)
    args.add_argument('--save_steps', type=int, default=10000)
    args.add_argument('--eval_steps', type=int, default=5000)
    
    args = args.parse_args()
    
    #model_path = str(Path(args.model_path).relative_to(Path('.')))
    model_path = "./../../../../pure-mlo-scratch/make_project/spring2024/trial-runs/llama3-8b"
    
    # Model    
    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16
    
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        # args.model_path,
        #os.path.join("./../../../..", args.model_path),
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
        local_files_only=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
     
    # Load dataset
    # train_data_df = pd.read_json(args.train_path, orient='records', lines=True)[:1000]
    # train_dataset = Dataset.from_pandas(train_data_df)
    
    # eval_data_df = pd.read_json(args.eval_path, orient='records', lines=True)[:1000]
    # eval_dataset = Dataset.from_pandas(eval_data_df)

    # #dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
    # train_dataset_tokenized = train_dataset.map(preprocess_function, batched=True)
    # eval_dataset_tokenized = eval_dataset.map(preprocess_function, batched=True)
    
    train_dataset = load_dataset(
        'jsonl',
        data_files=args.train_path,
        split='train'
    )
    
    eval_dataset = load_dataset(
        'jsonl',
        data_files=args.eval_path,
        split='train'
    )
    
    def template_dataset(examples):
        messages = [{
            'role': 'user',
            'content': examples['prompt']
        }]
        return{"text":  tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
    
    train_dataset_tokenized = train_dataset.map(template_dataset, remove_columns=["messages"])
    eval_dataset_tokenized = eval_dataset.map(template_dataset, remove_columns=["messages"])
    
    print(train_dataset_tokenized[0])# template dataset
    
    
    train_dataset_tokenized = train_dataset_tokenized.select_columns(['input_ids', 'attention_mask', 'labels'])
    eval_dataset_tokenized = eval_dataset_tokenized.select_columns(['input_ids', 'attention_mask', 'labels'])
    
    print(train_dataset_tokenized[0])
    
     ################
    # PEFT
    ################

    # # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    # peft_config = LoraConfig(
    #     lora_alpha=8,
    #     lora_dropout=0.05,
    #     r=16,
    #     bias="none",
    #     target_modules="all-linear",
    #     task_type="CAUSAL_LM",
    #     # modules_to_save = ["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
    # )

    ################
    # Training
    ################
    # trainer = SFTTrainer(
    #     model=model,
    #     args=TrainingArguments(
    #         output_dir=args.output_dir,
    #         per_device_train_batch_size=1,  # Batch size per device
    #         dataloader_num_workers=1,  # Number of subprocesses for data loading
    #         num_train_epochs=1,  # Number of training epochs
    #         gradient_accumulation_steps=2,  # Number of updates steps to accumulate before performing a backward/update pass
    #         save_total_limit=1,  # Maximum number of checkpoints to keep
    #         learning_rate=2e-4,  # Learning rate
    #         weight_decay=0.,  # Weight decay
    #         warmup_ratio=0.03,  # Proportion of training to perform linear learning rate warmup
    #         lr_scheduler_type="cosine",  # Learning rate scheduler type
    #         save_strategy="steps",  # Save strategy to use
    #         save_steps=500,  # Save checkpoint every 500 steps
    #         logging_steps=1,  # Log training information every step
    #         evaluation_strategy="no",  # Do not perform evaluation
    #         do_eval=False,  # Disable evaluation as we won't use an evaluation set
    #         bf16=True,  # Use bf16 precision
    #         tf32=True,  # Use tf32 precision
    #         gradient_checkpointing=True,  # Enable gradient checkpointing
    #         gradient_checkpointing_kwargs={"use_reentrant": False},  # Gradient checkpointing settings
    #         report_to="wandb"
    #     ),
    #     train_dataset=train_dataset_tokenized,
    #     dataset_text_field="text",
    #     eval_dataset=eval_dataset_tokenized,
    #     peft_config=peft_config,
    #     max_seq_length=args.max_seq_length,
    #     tokenizer=tokenizer,
    #     packing=True,
    #     dataset_kwargs={
    #         "add_special_tokens": False,  # We template with special tokens
    #         "append_concat_token": False,  # No need to add additional separator token
    #     },
    # )
    # if trainer.accelerator.is_main_process:
    #     trainer.model.print_trainable_parameters()

    # ##########################
    # # Train model
    # ##########################
    # trainer.train()

    # ##########################
    # # SAVE MODEL FOR SAGEMAKER
    # ##########################
    # if trainer.is_fsdp_enabled:
    #     trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # trainer.save_model()
    
    

    # # Define training arguments
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir, exist_ok=True)
    # training_args = TrainingArguments(
    #     output_dir=args.output_dir,
    #     #num_train_epochs=3,           # Adjust based on your needs
    #     #save_steps=10000,             # Adjust based on your preferences
    #     #eval_steps=5000,              # Adjust based on your preferences
    #     # Add additional arguments like learning rate, weight decay, etc.
    #     per_device_train_batch_size=32,  # Batch size per device
    #     dataloader_num_workers=1,  # Number of subprocesses for data loading
    #     num_train_epochs=1,  # Number of training epochs
    #     gradient_accumulation_steps=1,  # Number of updates steps to accumulate before performing a backward/update pass
    #     save_total_limit=1,  # Maximum number of checkpoints to keep
    #     learning_rate=2e-4,  # Learning rate
    #     weight_decay=0.,  # Weight decay
    #     warmup_ratio=0.03,  # Proportion of training to perform linear learning rate warmup
    #     lr_scheduler_type="cosine",  # Learning rate scheduler type
    #     save_strategy="steps",  # Save strategy to use
    #     save_steps=500,  # Save checkpoint every 500 steps
    #     logging_steps=1,  # Log training information every step
    #     evaluation_strategy="no",  # Do not perform evaluation
    #     do_eval=False,  # Disable evaluation as we won't use an evaluation set
    #     bf16=True,  # Use bf16 precision
    #     tf32=True,  # Use tf32 precision
    #     gradient_checkpointing=True,  # Enable gradient checkpointing
    #     gradient_checkpointing_kwargs={"use_reentrant": False},  # Gradient checkpointing settings
    #     report_to="wandb"  # Reporting settings
    # )
    

    # # Define trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset_tokenized,
    #     #eval_dataset=eval_dataset_tokenized,
    #     # Add data collator if needed
    # )

    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = tokenized_dataset,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )
    
    print(model(**train_dataset_tokenized[0]))

    # Start training
    #trainer.train()

    # # Save the model
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
