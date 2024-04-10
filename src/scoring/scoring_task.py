import os
import json
import numpy as np
import pandas as pd

import evaluate

from bleu import Bleu
from rouge import Rouge
from bertscore import BertScore  #
from perplexity import Perplexity  #

import argparse # added
from tqdm import tqdm # added


POSSIBLE_METRICS = set({"bleu", "rouge", "bertscore", "meteor", "perplexity"}) # added


def calculate_scores(df, metrics, batch_size=128):
    if not metrics:
        raise ValueError("No metrics specified for scoring.")
    print("Beginning scoring...")

    scores = {}
    for metric in metrics:
        scores[metric] = []

    # initialize scorers
    if "bleu" in metrics:
        bleuScorer = Bleu()
        print("bleuScorer initialized")
    if "rouge" in metrics:
        scores["rouge"] = [[], [], []]
        rougeScorer = Rouge(["rouge1", "rouge2", "rougeL"])
        print("rougeScorer initialized")
    if "bertscore" in metrics:
        bertScorer = BertScore()
        print("bertScorer initialized")
    if "meteor" in metrics:
        meteorScorer = evaluate.load("meteor")
        print("meteorScorer initialized")
        
    pbar = tqdm(total=df.shape[0], desc="Processing samples")

    def calculate_scores(df):
        if "bleu" in metrics:
            temp = bleuScorer(
                refs=df["gold"].tolist(),
                hyps=df["generated"].tolist(),
            )
            scores["bleu"].extend(temp)
        if "rouge" in metrics:
            temp = rougeScorer(
                refs=df["gold"].tolist(),
                hyps=df["generated"].tolist(),
            )
            scores["rouge"][0].extend(
                    temp['rouge1'],
            )
            scores["rouge"][1].extend(
                    temp['rouge2'],
            )
            scores["rouge"][2].extend(
                    temp['rougeL'],
            )
        if "bertscore" in metrics:
            temp = bertScorer(
                refs=df["gold"].tolist(),
                hyps=df["generated"].tolist(),
            )
            scores["bertscore"].extend(temp)
        if "meteor" in metrics:
            temp = meteorScorer.compute(
                references=df["gold"].tolist(),
                predictions=df["generated"].tolist(),
            )
            scores["meteor"].append(temp["meteor"])
    
        # print progress
        current_row = i + batch_size
        if current_row % batch_size == 0:
            pbar.update(batch_size)

    df.set_index("idx", drop=False, inplace=True)

    for i in range(0, df.shape[0], batch_size):
        df_batch = df[i : i + batch_size]
        calculate_scores(df_batch)

    print(f"Processed {df.shape[0]}/{df.shape[0]} samples.", flush=True)
    print("Done.")
    return scores


def compute_overall_score(scores):
    print("Computing overall score...")
    leaderboard = {}

    metrics = list(scores.keys())

    if "bleu" in metrics:
        bleu_score = np.mean(scores["bleu"])
        
        leaderboard["bleu"] = bleu_score
        
    if "rouge" in metrics:
        rouge_1_score = np.mean(
            scores["rouge"][0]
        )
        rouge_2_score = np.mean(
            scores["rouge"][1]
        )
        rouge_l_score = np.mean(
            scores["rouge"][2]
        )
       
        leaderboard["rouge1"] = rouge_1_score
        leaderboard["rouge2"] = rouge_2_score
        leaderboard["rougeL"] = rouge_l_score
        
    if "bertscore" in metrics:
        bertscore_score = np.mean(
            scores["bertscore"]
        )
        
        leaderboard["bertscore"] = bertscore_score
        
    if "meteor" in metrics:
        meteor_score = np.mean(
            scores["meteor"]
        )
        
        leaderboard["meteor"] = meteor_score

    # normalize sacrebleu to be between 0 and 1
    for key in leaderboard.keys():
        if key == "sacrebleu":
            leaderboard[key] = leaderboard[key] / 100

    overall_score = np.mean(list(leaderboard.values()))
    leaderboard["overall"] = overall_score

    print("Done.")
    return leaderboard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scoring script for the MIMIC-III discharge summary generation task.")
    parser.add_argument("--input_dir", type=str, help="Path to the input directory")
    parser.add_argument("--score_dir", type=str, help="Path to the directory to save the scores")
    parser.add_argument("--metrics", nargs="+", help="Metrics to calculate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for scoring")
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    score_dir = args.score_dir
    
    input_metrics = args.metrics
    print(input_metrics)
    for metric in input_metrics:
        if metric not in POSSIBLE_METRICS:
            raise ValueError(f"Invalid metric: {metric}. Please choose from {POSSIBLE_METRICS}")
    
        
    #reference_dir = os.path.join("/app/input/", "ref")
    #generated_dir = os.path.join("/app/input/", "res")
    #score_dir = "/app/output/"

    print("Reading generated texts...")
    df = pd.read_csv(
        os.path.join(input_dir), keep_default_na=False
    )

    # covert all elements to string
    df["gold"] = df["gold"].astype(str)
    df["generated"] = df["generated"].astype(str)

    # convert to single-line strings by removing newline characters
    df["gold"] = df["gold"].str.replace(
        "\n", " "
    )
    df["generated"] = df["generated"].str.replace(
        "\n", " "
    )

    # convert all idx to int
    df["idx"] = df["idx"].astype(int)

    df = df.sort_values(by="idx")
    print("Done.")

    scores = calculate_scores(
        df,
        metrics=input_metrics,
        batch_size=args.batch_size,
    )

    leaderboard = compute_overall_score(scores)
    
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)

    with open(os.path.join(score_dir, "scores.json"), "w") as score_file:
        score_file.write(json.dumps(leaderboard))
