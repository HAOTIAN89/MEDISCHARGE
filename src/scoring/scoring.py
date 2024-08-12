import os
import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
# from pathlib import Path

import evaluate

from .bleu import Bleu
from .rouge import Rouge
from .bertscore import BertScore
from .align import AlignScorer
# from .UMLSScorer import UMLSScorer

SUPPORTED_METRICS = {"bleu", "rouge", "bertscore", "meteor", "align"}#, "medcon"}
def calculate_scores(generated, reference, metrics, batch_size=8):
    if not metrics:
        raise ValueError("No metrics specified for scoring.")
    print("Beginning scoring...")

    scores = {}
    metrics = [metric.lower() for metric in metrics] # lowercase all metrics
    for metric in metrics:
        scores[metric] = {"discharge_instructions": [], "brief_hospital_course": []}
        if metric == "rouge":
            scores[metric]["discharge_instructions"] = [[] for _ in range(3)]
            scores[metric]["brief_hospital_course"] = [[] for _ in range(3)]

    # initialize scorers
    if "bleu" in metrics:
        bleuScorer = Bleu()
        print("bleuScorer initialized")
    if "rouge" in metrics:
        rougeScorer = Rouge(["rouge1", "rouge2", "rougeL"])
        print("rougeScorer initialized")
    if "bertscore" in metrics:
        bertScorer = BertScore()
        print("bertScorer initialized")
    if "meteor" in metrics:
        meteorScorer = evaluate.load("meteor")
        print("meteorScorer initialized")
    if "align" in metrics:
        alignScorer = AlignScorer()
        print("alignScorer initialized")
    # if "medcon" in metrics:
    #     quickumls_fp = (Path(__file__).parent / "quickumls/").as_posix()
    #     medconScorer = UMLSScorer(quickumls_fp=quickumls_fp)
    #     print("medconScorer initialized")
        
    pbar = tqdm(total=len(generated), desc="Processing samples")
        
    def calculate_scores(rows_ref, rows_gen):
        if "bleu" in metrics:
            # Discharge Instructions
            temp = bleuScorer(
                refs=rows_ref["discharge_instructions"].tolist(),
                hyps=rows_gen["discharge_instructions"].tolist(),
            )
            scores["bleu"]["discharge_instructions"].extend(temp)
            # Brief Hospital Course
            temp = bleuScorer(
                refs=rows_ref["brief_hospital_course"].tolist(),
                hyps=rows_gen["brief_hospital_course"].tolist(),
            )
            scores["bleu"]["brief_hospital_course"].extend(temp)
        if "rouge" in metrics:
            # Discharge Instructions
            scores["rouge"]["discharge_instructions"] = [[] for _ in range(3)]
            temp = rougeScorer(
                refs=rows_ref["discharge_instructions"].tolist(),
                hyps=rows_gen["discharge_instructions"].tolist(),
            )
            for i, rouge_sc in enumerate(["rouge1", "rouge2", "rougeL"]):
                scores["rouge"]["discharge_instructions"][i].extend(temp[rouge_sc])
            # Brief Hospital Course
            scores["rouge"]["brief_hospital_course"] = [[] for _ in range(3)]
            temp = rougeScorer(
                refs=rows_ref["brief_hospital_course"].tolist(),
                hyps=rows_gen["brief_hospital_course"].tolist(),
            )
            for i, rouge_sc in enumerate(["rouge1", "rouge2", "rougeL"]):
                scores["rouge"]["brief_hospital_course"][i].extend(temp[rouge_sc])
        if "bertscore" in metrics:
            # Discharge Instructions
            temp = bertScorer(
                refs=rows_ref["discharge_instructions"].tolist(),
                hyps=rows_gen["discharge_instructions"].tolist(),
            )
            scores["bertscore"]["discharge_instructions"].extend(temp)
            # Brief Hospital Course
            temp = bertScorer(
                refs=rows_ref["brief_hospital_course"].tolist(),
                hyps=rows_gen["brief_hospital_course"].tolist(),
            )
            scores["bertscore"]["brief_hospital_course"].extend(temp)
        if "meteor" in metrics:
            # Discharge Instructions
            temp = meteorScorer.compute(
                references=rows_ref["discharge_instructions"].tolist(),
                predictions=rows_gen["discharge_instructions"].tolist(),
            )
            scores["meteor"]["discharge_instructions"].append(temp["meteor"])
            # Brief Hospital Course
            temp = meteorScorer.compute(
                references=rows_ref["brief_hospital_course"].tolist(),
                predictions=rows_gen["brief_hospital_course"].tolist(),
            )
            scores["meteor"]["brief_hospital_course"].append(temp["meteor"])
        if "align" in metrics:
            # Discharge Instructions
            temp = alignScorer(
                refs=rows_ref["discharge_instructions"].tolist(),
                hyps=rows_gen["discharge_instructions"].tolist(),
            )
            scores["align"]["discharge_instructions"].extend(temp)
            # Brief Hospital Course
            temp = alignScorer(
                refs=rows_ref["brief_hospital_course"].tolist(),
                hyps=rows_gen["brief_hospital_course"].tolist(),
            )
            scores["align"]["brief_hospital_course"].extend(temp)
        # if "medcon" in metrics:
        #     # Discharge Instructions
        #     temp = medconScorer(
        #         rows_ref["discharge_instructions"].tolist(),
        #         rows_gen["discharge_instructions"].tolist(),
        #     )
        #     scores["medcon"]["discharge_instructions"].extend(temp)
        #     # Brief Hospital Course
        #     temp = medconScorer(
        #         rows_ref["brief_hospital_course"].tolist(),
        #         rows_gen["brief_hospital_course"].tolist(),
        #     )
        #     scores["medcon"]["brief_hospital_course"].extend(temp)

        # print progress
        current_row = i + batch_size
        if current_row % batch_size == 0:
            pbar.update(batch_size)

    reference.set_index("hadm_id", drop=False, inplace=True)
    generated.set_index("hadm_id", drop=False, inplace=True)

    for i in range(0, len(generated), batch_size):
        rows_ref = reference[i : i + batch_size]
        rows_gen = generated[i : i + batch_size]
        calculate_scores(rows_ref=rows_ref, rows_gen=rows_gen)

    print(f"Processed {len(generated)} samples.", flush=True)
    print("Done.")
    return scores


def compute_overall_score(scores):
    print("Computing overall score...")
    leaderboard = {}

    metrics = list(scores.keys())

    if "bleu" in metrics:
        bleu_discharge_instructions = np.mean(scores["bleu"]["discharge_instructions"])
        bleu_brief_hospital_course = np.mean(scores["bleu"]["brief_hospital_course"])
        leaderboard["bleu"] = np.mean(
            [bleu_discharge_instructions, bleu_brief_hospital_course]
        )
    if "rouge" in metrics:
        rouge_1_discharge_instructions = np.mean(
            scores["rouge"]["discharge_instructions"][0]
        )
        rouge_2_discharge_instructions = np.mean(
            scores["rouge"]["discharge_instructions"][1]
        )
        rouge_l_discharge_instructions = np.mean(
            scores["rouge"]["discharge_instructions"][2]
        )
        rouge_1_brief_hospital_course = np.mean(
            scores["rouge"]["brief_hospital_course"][0]
        )
        rouge_2_brief_hospital_course = np.mean(
            scores["rouge"]["brief_hospital_course"][1]
        )
        rouge_l_brief_hospital_course = np.mean(
            scores["rouge"]["brief_hospital_course"][2]
        )

        leaderboard["rouge1"] = np.mean(
            [rouge_1_discharge_instructions, rouge_1_brief_hospital_course]
        )
        leaderboard["rouge2"] = np.mean(
            [rouge_2_discharge_instructions, rouge_2_brief_hospital_course]
        )
        leaderboard["rougel"] = np.mean(
            [rouge_l_discharge_instructions, rouge_l_brief_hospital_course]
        )
    if "bertscore" in metrics:
        bertscore_discharge_instructions = np.mean(
            scores["bertscore"]["discharge_instructions"]
        )
        bertscore_brief_hospital_course = np.mean(
            scores["bertscore"]["brief_hospital_course"]
        )
        leaderboard["bertscore"] = np.mean(
            [bertscore_discharge_instructions, bertscore_brief_hospital_course]
        )
    if "meteor" in metrics:
        meteor_discharge_instructions = np.mean(
            scores["meteor"]["discharge_instructions"]
        )
        meteor_brief_hospital_course = np.mean(
            scores["meteor"]["brief_hospital_course"]
        )
        leaderboard["meteor"] = np.mean(
            [meteor_discharge_instructions, meteor_brief_hospital_course]
        )
    if "align" in metrics:
        align_discharge_instructions = np.mean(
            scores["align"]["discharge_instructions"]
        )
        align_brief_hospital_course = np.mean(scores["align"]["brief_hospital_course"])
        leaderboard["align"] = np.mean(
            [align_discharge_instructions, align_brief_hospital_course]
        )
    # if "medcon" in metrics:
    #     medcon_discharge_instructions = np.mean(
    #         scores["medcon"]["discharge_instructions"]
    #     )
    #     medcon_brief_hospital_course = np.mean(
    #         scores["medcon"]["brief_hospital_course"]
    #     )
    #     leaderboard["medcon"] = np.mean(
    #         [medcon_discharge_instructions, medcon_brief_hospital_course]
    #     )
        
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
        if metric not in SUPPORTED_METRICS:
            raise ValueError(f"Invalid metric: {metric}. Please choose from {SUPPORTED_METRICS}")
    
    print("Reading generated texts...")
    generated = pd.read_csv(
        os.path.join(input_dir, "submission.csv"), keep_default_na=False
    )
    reference = pd.read_csv(
        os.path.join(input_dir, "discharge_target.csv"), keep_default_na=False
    )

    # covert all elements to string
    generated["discharge_instructions"] = generated["discharge_instructions"].astype(str)
    reference["discharge_instructions"] = reference["discharge_instructions"].astype(str)

    generated["brief_hospital_course"] = generated["brief_hospital_course"].astype(str)
    reference["brief_hospital_course"] = reference["brief_hospital_course"].astype(str)

    # convert to single-line strings by removing newline characters
    generated["discharge_instructions"] = generated["discharge_instructions"].str.replace(
        "\n", " "
    )
    reference["discharge_instructions"] = reference["discharge_instructions"].str.replace(
        "\n", " "
    )

    generated["brief_hospital_course"] = generated["brief_hospital_course"].str.replace(
        "\n", " "
    )
    reference["brief_hospital_course"] = reference["brief_hospital_course"].str.replace(
        "\n", " "
    )

    # convert all hadm_id to int
    generated["hadm_id"] = generated["hadm_id"].astype(int)
    reference["hadm_id"] = reference["hadm_id"].astype(int)

    # get the list of hadm_ids from the reference
    ref_hadm_ids = list(reference["hadm_id"].unique())
    # filter the generated texts to only include hadm_ids from the reference
    generated = generated[generated["hadm_id"].isin(ref_hadm_ids)]
    
    # check for invalid submissions
    if not generated.shape[0] == reference.shape[0]:
        raise ValueError(
            "Submission does not contain the correct number of rows. Please check your submission file."
        )

    if set(generated["hadm_id"].unique()) != set(reference["hadm_id"].unique()):
        missing_hadm_ids = set(reference["hadm_id"].unique()) - set(
            generated["hadm_id"].unique()
        )
        extra_hadm_ids = set(generated["hadm_id"].unique()) - set(
            reference["hadm_id"].unique()
        )
        print(f"Missing hadm_ids: {missing_hadm_ids}")
        print(f"Extra hadm_ids: {extra_hadm_ids}")
        raise ValueError(
            "Submission does not contain all hadm_ids from the test set. Please check your submission file to ensure that all samples are present."
        )

    if not generated["hadm_id"].nunique() == len(generated):
        raise ValueError(
            "Submission contains duplicate hadm_ids. Please check your submission file."
        )

    generated = generated.sort_values(by="hadm_id")
    reference = reference.sort_values(by="hadm_id")
    print("Done.")

    scores = calculate_scores(
        generated,
        reference,
        metrics=input_metrics,
        batch_size=args.batch_size,
    )

    leaderboard = compute_overall_score(scores)
    
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)
    
    with open(os.path.join(score_dir, "scores.json"), "w") as score_file:
        score_file.write(json.dumps(leaderboard))
