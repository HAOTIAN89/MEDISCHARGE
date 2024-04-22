import torch
import torch.nn as nn
from bert_score import BERTScorer
from transformers import DistilBertTokenizer

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def truncate_sequences(refs, hyps, max_length=512):
    truncated_refs = []
    truncated_hyps = []
    for ref, hyp in zip(refs, hyps):
        # Tokenize and truncate references and hypotheses
        tokenized_ref = tokenizer(ref, max_length=max_length, truncation=True, return_tensors='pt')
        tokenized_hyp = tokenizer(hyp, max_length=max_length, truncation=True, return_tensors='pt')
        # Convert tokenized texts back to strings (only for demonstration; actual use might keep them tokenized)
        truncated_refs.append(tokenizer.decode(tokenized_ref.input_ids.squeeze(), skip_special_tokens=True))
        truncated_hyps.append(tokenizer.decode(tokenized_hyp.input_ids.squeeze(), skip_special_tokens=True))
    return truncated_refs, truncated_hyps

class BertScore(nn.Module):
    def __init__(self):
        super(BertScore, self).__init__()
        with torch.no_grad():
            self.bert_scorer = BERTScorer(
                model_type="distilbert-base-uncased",
                num_layers=4,
                batch_size=16,
                nthreads=16,
                all_layers=False,
                idf=False,
                device="cuda" if torch.cuda.is_available() else "cpu",
                lang="en",
                rescale_with_baseline=True,
                baseline_path=None,
            )

    def forward(self, refs, hyps):
        truncate_refs, truncate_hyps = truncate_sequences(refs, hyps)
        p, r, f = self.bert_scorer.score(
            cands=truncate_hyps,
            refs=truncate_refs,
            verbose=False,
            batch_size=16,
        )
        return f.tolist()


if __name__ == "__main__":
    x, y = BertScore()(
        hyps=[            
            "there are moderate bilateral pleural effusions with overlying atelectasis,  underlying consolidation not excluded. mild prominence of the interstitial  markings suggests mild pulmonary edema. the cardiac silhouette is mildly  enlarged. the mediastinal contours are unremarkable. there is no evidence of  pneumothorax.",
            "there are moderate bilateral pleural effusions with overlying atelectasis,  underlying consolidation not excluded. mild prominence of the interstitial  markings suggests mild pulmonary edema. the cardiac silhouette is mildly  enlarged. the mediastinal contours are unremarkable. there is no evidence of  pneumothorax.",
            "there are moderate bilateral pleural effusions with overlying atelectasis,  underlying consolidation not excluded. mild prominence of the interstitial  markings suggests mild pulmonary edema. the cardiac silhouette is mildly  enlarged. the mediastinal contours are unremarkable. there is no evidence of  pneumothorax.",
        ],
        refs=[
            "heart size is moderately enlarged. the mediastinal and hilar contours are unchanged. there is no pulmonary edema. small left pleural effusion is present. patchy opacities in the lung bases likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.",
            "heart size is mildly enlarged. the mediastinal and hilar contours are normal. there is mild pulmonary edema. moderate bilateral pleural effusions are present, left greater than right. bibasilar airspace opacities likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.",
            "heart size is mildly enlarged. the mediastinal and hilar contours are normal. there is mild pulmonary edema. moderate bilateral pleural effusions are present, left greater than right. bibasilar airspace opacities likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.",
        ],
    )
    print(x)
    print(y)
