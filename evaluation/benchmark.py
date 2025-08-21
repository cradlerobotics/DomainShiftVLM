import os
import torch
from tqdm import tqdm

# For NLP metrics
from pycocoevalcap.cider.cider import Cider
from bert_score import score as bert_score
import evaluate # For ROUGE score
from evaluation.gptscore import GPTScorer # For GPTScore
from evaluation.clipscore import CLIPScorer


def calculate_nlp_metrics(ground_truth, predictions):
    """Calculate selected NLP metrics and return individual scores."""
    matching_ids = set(ground_truth.keys()) & set(predictions.keys())
    if not matching_ids:
        print("Warning: No matching IDs between ground truth and predictions")
        empty_list = []
        return {
            'cider': empty_list,
            'bert': empty_list,
            'rouge_l': empty_list,
            'gpt': empty_list
        }
    results = {
        'cider': calculate_cider(ground_truth, predictions),
        'bert': calculate_bertscore(ground_truth, predictions),
        'rouge_l': calculate_rouge(ground_truth, predictions),
        'gpt': calculate_gptscore(ground_truth, predictions)
    }
    return results

def calculate_cider(references, candidates):
    refs = {k: [v] for k, v in references.items()}
    cands = {k: [v] for k, v in candidates.items()}
    cider_scorer = Cider()
    _, c_scores = cider_scorer.compute_score(gts=refs, res=cands)
    return c_scores.tolist() if hasattr(c_scores, 'tolist') else c_scores

def calculate_bertscore(references, candidates):
    refs = [v for v in references.values()]
    hyps = [v for v in candidates.values()]
    if refs and hyps:
        _, _, F1 = bert_score(
            hyps, refs, 
            model_type="microsoft/deberta-xlarge-mnli", 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            rescale_with_baseline=True, lang="en"
        )
        return F1.tolist()
    else:
        return []

def calculate_rouge(references, candidates):
    rouge = evaluate.load("rouge")
    rouge_scores = []
    for id in references:
        gt_caption = references[id]
        pred_caption = candidates[id]
        rouge_score = rouge.compute(predictions=[pred_caption], references=[gt_caption])
        rouge_scores.append(float(rouge_score['rougeL']))
    return rouge_scores

def calculate_gptscore(references, candidates):
    gpt_scorer = GPTScorer(model_name="gpt2-xl")
    gpt_scores = []
    for id in references:
        gt_caption = references[id]
        pred_caption = candidates[id]
        score = gpt_scorer.calculate_gptscore(gt_caption, pred_caption)
        gpt_scores.append(score)
    return gpt_scores

def calculate_clip_scores(segmentation_dir, captions, masked):
    scorer = CLIPScorer(device='cuda' if torch.cuda.is_available() else 'cpu')
    prefix = "mask" if masked else "cropped"
    clip_scores = {}
    for img_name, caption in tqdm(captions.items()):
        img_path = os.path.join(segmentation_dir, f"{prefix}_{img_name}.png")
        clip_scores[img_name] = scorer.score(img_path, caption)
    return clip_scores

if __name__ == "__main__":
    # Test methods one by one with an example
    ground_truth = {
        1: "A cat sitting on a mat.",
        2: "A dog playing with a ball.",
        3: "A bird perched on a tree branch."
    }
    predictions = {
        1: "A cat is on the mat.",
        2: "A dog is playing with a ball.",
        3: "A bird perched on a tree branch."
    }

    print("BERTScore:", calculate_bertscore(ground_truth, predictions))
