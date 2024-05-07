from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse

# Save the model and tokenizer

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    args.add_argument('--output_dir', type=str, default='/pure-mlo-scratch/make_project/spring2024/trial-runs/llama3-8b')
    args.add_argument('--hf_token', type=str, default=None)
    args = args.parse_args()
    
    # Login to the Hugging Face model hub
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.hf_token)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=args.hf_token
    )
    
    # Save the model and tokenizer
    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)
