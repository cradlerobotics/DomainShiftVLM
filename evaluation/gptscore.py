import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn

class GPTScorer:
    def __init__(self, model_name="gpt2-xl", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.loss_fct = nn.CrossEntropyLoss(reduction="none")

    def calculate_gptscore(self, reference, candidate):
        """
        GPTScore = average log-likelihood per token of candidate given reference.
        Returns None if candidate too short.
        """
        # Tokenize
        ref_ids = self.tokenizer(reference, return_tensors="pt").input_ids.to(self.device)
        cand_ids = self.tokenizer(candidate, return_tensors="pt").input_ids.to(self.device)

        # Concatenate ref + cand
        text = reference + " " + candidate
        tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
        ref_len = ref_ids.shape[1]

        with torch.no_grad():
            outputs = self.model(**tokens, labels=tokens["input_ids"])
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = tokens["input_ids"][:, 1:].contiguous()

            # Per-token loss
            losses = self.loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            ).view(tokens["input_ids"].size(0), -1)

        # Only evaluate candidate tokens (skip reference part)
        cand_losses = losses[:, ref_len-1:]   # shift by -1 since labels are next-token

        cand_loss = cand_losses.mean().item()

        # Negative loss = log-prob (closer to 0 is better)
        return -cand_loss


def main():
    evaluator = GPTScorer(model_name="gpt2-xl")
    reference = "Max is the father of John."
    candidate = "John is the child of Max."
    gptscore_loss = evaluator.calculate_gptscore(reference, candidate)
    print(f"GPTScore: {gptscore_loss:.4f}")

if __name__ == "__main__":
    main()
