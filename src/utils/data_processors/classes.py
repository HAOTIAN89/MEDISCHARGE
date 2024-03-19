# This file contains all the data processors used to preprocess the data.
# Authors: Farouk Boukil

from typing import List, Dict, Any, Optional, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-dark")
from transformers import AutoTokenizer

from .base import BaseDataProcessor

class ExplorationDataProcessor(BaseDataProcessor):
    """This class is used to process the data for exploration."""
    
    DEFAULT_TOKENIZER = "meta-llama/Llama-2-7b"
     
    def __init__(self,
                 append_transforms: Optional[List[Callable]] = None):
        """
        Args:
            append_transforms (Optional[List[Callable]], optional): Callables to be called
            sequentially in-order on the output of this data processor. Defaults to None.
        """
        super(ExplorationDataProcessor, self).__init__()
        self.append_transforms = append_transforms
        
    def __call__(self, inputs: Dict[str, Dict[str, pd.DataFrame]], **kwargs)\
        -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
        """Processes the inputs and returns an exploration report.
        
        Args:
            inputs: The inputs to process.
            kwargs: Additional arguments.
            
        Returns:
            Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]: The exploration report.
        """
        return self._process_inputs(inputs, **kwargs)
   
    def _process_inputs(self, inputs, **kwargs):
        """Processes the inputs and returns an exploration report.
        
        Args:
            inputs: The inputs to process.
            kwargs: Additional arguments.
            
        Returns:
            Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]: The exploration report.
        """
        # Parse kwargs.
        tokenizers = kwargs.get("tokenizers", None)
        text_columns = kwargs["text_columns"]
        device_map = kwargs.get("device_map", "cpu")
        
        # Fetch the tokenizers.
        if not tokenizers:
            tokenizers = (ExplorationDataProcessor.DEFAULT_TOKENIZER,)
        tokenizers = {
            tokenizer: AutoTokenizer.from_pretrained(tokenizer, device_map=device_map)
            for tokenizer in tokenizers
            }
        
        # Produce a report for each tokenizer.
        # The final report is the combination of all the tokenizer reports,
        # each enriched with the summary per tokenizer.
        report = {}
        for tokenizer_name, tokenizer in tokenizers.items():
            report[tokenizer_name] = {}
            tokenizer_report = report[tokenizer_name]
            
            for key, dict_ in inputs.items():
                tokenizer_report[key] = {}
                
                for entry, filepath in dict_.items():
                    dataframe = pd.read_csv(filepath, compression="gzip", sep=",", encoding="utf-8")
                    tokenizer_report[key][entry] = {}
                    
                    # Count the samples
                    tokenizer_report[key][entry]["samples"] = {}
                    tokenizer_report_key_entry_samples = tokenizer_report[key][entry]["samples"]
                    tokenizer_report_key_entry_samples["counts"] = len(dataframe)
                    
                    # Tokenize the text, count the tokens, compute statistics, and build plots.
                    tokenizer_report[key][entry]["tokens"] = {}
                    tokenizer_report_key_entry_tokens = tokenizer_report[key][entry]["tokens"]
                    token_counts = self._token_counts(dataframe, tokenizer=tokenizer, text_columns=text_columns)
                    tokenizer_report_key_entry_tokens["counts"] = token_counts
                    tokenizer_report_key_entry_tokens["stats"] = self._token_stats(token_counts)
                    tokenizer_report_key_entry_tokens["plots"] = self._token_plots(token_counts, figname=f"{key}_{entry}")
                    dataframe = None
                    
                # Build summary for tokenizer report per key.
                tokenizer_report[key][":summary:"] = self._per_key_summary(tokenizer_report[key], key=key)
                
        return report
                
    def _token_counts(self, dataframe: pd.DataFrame, tokenizer: AutoTokenizer, text_columns: List[str])\
        -> List[int]:
        """Returns the number of tokens in each entry of the dataframe.
        
        Args:
            dataframe: The dataframe containing the text to tokenize.
            tokenizer: The tokenizer to use.
            text_columns: The columns containing the text to tokenize.
            
        Returns:
            List[int]: The number of tokens in each text entry of the dataframe.
        """
        text_columns = list(set(text_columns).intersection(set(dataframe.columns)))
        if len(text_columns) == 0:
            return []
        
        token_counts = []
        for _, row in dataframe[text_columns].iterrows():
            for text in row.values:
                batch_enc = tokenizer(
                    text,
                    padding=False,
                    truncation=False,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_special_tokens_mask=False,
                    return_token_type_ids=False)
                n_tokens = len(batch_enc["input_ids"])
                token_counts.append(n_tokens)
        return token_counts
    
    def _token_stats(self, token_counts: List[int]) -> Dict[str, float | int]:
        """Returns the statistics of the token counts.
        
        Args:
            token_counts: The token counts.
            
        Returns:
            Dict[str, float | int]: The statistics of the token counts.
        """
        if len(token_counts) == 0:
            return {
                "total": 0,
                "max": 0,
                "min": 0,
                "mean": 0,
                "std": 0,
                "skew": 0,
                "kurt": 0,
                "pct_25": 0,
                "median": 0,
                "pct_75": 0,
                "pct_95": 0,
                "pct_99": 0
            }
        return {
            "total": len(token_counts),
            "max": np.max(token_counts),
            "min": np.min(token_counts),
            "mean": np.mean(token_counts),
            "skew": pd.Series(token_counts).skew().item(),
            "kurt": pd.Series(token_counts).kurt().item(),
            "std": np.std(token_counts),
            "pct_25": np.percentile(token_counts, 25),
            "median": np.median(token_counts),
            "pct_75": np.percentile(token_counts, 75),
            "pct_95": np.percentile(token_counts, 95),
            "pct_99": np.percentile(token_counts, 99)
        }
    
    def _token_plots(self, token_counts: List[int], figname: str) -> Dict[str, Any]:
        """Returns the plots for the token counts.
        
        Args:
            token_counts: The token counts.
            figname: The name of the figure.
        
        Returns:
            Dict[str, Any]: The plots for the token counts."""
        if len(token_counts) == 0:
            return {
                "hist": None,
                "box": None
            }
            
        # Create a histogram
        hist_fig, hist_ax = plt.subplots(1, 1, figsize=(6, 5))
        hist_fig.suptitle(figname)
        hist_ax.hist(token_counts, bins=30, color="blue", edgecolor="black")
        hist_ax.set_title("Token Counts Histogram")
        hist_ax.set_xlabel("Token Count")
        hist_ax.set_ylabel("Frequency")
        
        # Create a boxplot
        box_fig, box_ax = plt.subplots(1, 1, figsize=(6, 5))
        box_fig.suptitle(figname)
        box_ax.boxplot(token_counts, vert=False, patch_artist=True, boxprops=dict(facecolor="blue"))
        box_ax.set_title("Token Counts Boxplot")
        box_ax.set_xlabel("Token Count")
        
        return {
            "hist": hist_fig,
            "box": box_fig
        }
        
    def _per_key_summary(self, tokenizer_report_per_key: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]], key: str)\
        -> Dict[str, Any]:
        """Returns the per-key summary of the tokenizer report.
        
        Args:
            tokenizer_report_per_key: The tokenizer report per key.
            key: The key.
        
        Returns:
            Dict[str, Any]: The summary.
        """
        per_key_summary = {
            "samples": {},
            "tokens": {}
        }
        total_samples = 0
        token_counts = []
        for _, reported in tokenizer_report_per_key.items():
            token_counts.extend(reported["tokens"]["counts"])
            total_samples += reported["samples"]["counts"]
        per_key_summary["samples"]["total"] = total_samples
        per_key_summary["tokens"]["stats"] = self._token_stats(token_counts)
        per_key_summary["tokens"]["plots"] = self._token_plots(token_counts, figname=f"{key}_summary")
        return per_key_summary