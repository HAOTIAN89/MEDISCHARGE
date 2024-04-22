from transformers import AutoModelForCausalLM
import torch

# Example usage
model = AutoModelForCausalLM.from_pretrained("/pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI-v4/hf_checkpoint_2000", local_files_only=True)

print(model.config)

model.config.rope_scaling = {
    "type": "dynamic",
    "factor": 3.0,
}

print(model.config)

# save model 
model.save_pretrained("/pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI-v4-extended", torch_dtype=torch.bfloat16)

print("Model saved")