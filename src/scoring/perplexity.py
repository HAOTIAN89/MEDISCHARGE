import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.nn.utils.rnn import pad_sequence

class Perplexity:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval() 
        if torch.cuda.is_available():
            self.model.to('cuda') 

    def __call__(self, text):
        perplexities = []
        for t in text:
            inputs = self.tokenizer(t, return_tensors="pt", truncation=True, max_length=1024)
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')

            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                log_likelihood = outputs[0]

            # Calculate perplexity
            perplexity = torch.exp(log_likelihood / inputs["input_ids"].shape[1]).item()
            perplexities.append(perplexity)

        return perplexities
