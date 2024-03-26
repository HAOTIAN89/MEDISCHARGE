"""Utility to count the number of tokens in the datasets."""

from transformers import AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt

model = "meta-llama/Llama-2-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(model, token = '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')


def get_token_count(text, tokenizer = tokenizer):
    output = tokenizer(text, return_length = True)
    return output.length[0]


def plot_token_count(counts_series, title: str):
    plt.hist(counts_series, bins = 100)
    plt.xlabel('values')
    plt.ylabel('counts')
    plt.title(title)
    plt.show()

