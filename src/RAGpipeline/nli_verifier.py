import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class NLIVerifier:
    def __init__(self, model_name="cross-encoder/nli-distilroberta-base"):
        print(f"--- Loading NLI Model: {model_name} ---")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Correct label mapping for HF NLI models
        self.id2label = {
            0: "REFUTES",   # contradiction
            1: "NEI",       # neutral
            2: "SUPPORTS"   # entailment
        }

    def verify(self, claim_triple, evidence_text):
        """
        Determines if the evidence supports or refutes the triple.
        """
        # premise = evidence, hypothesis = claim
        inputs = self.tokenizer(
            evidence_text,
            claim_triple,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            label_idx = torch.argmax(probs).item()

        confidence = probs[0][label_idx].item()
        verdict = self.id2label[label_idx]

        return {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "claim": claim_triple,
            "evidence": evidence_text[:200]
        }


if __name__ == "__main__":
    verifier = NLIVerifier()

    # CONTRADICTION
    result = verifier.verify(
        claim_triple="Drinking bleach prevents COVID",
        evidence_text="Health authorities warn that drinking bleach is dangerous and does not prevent COVID-19."
    )
    print("Test 1:", result)

    # SUPPORT
    result = verifier.verify(
        claim_triple="Vitamin C supports immune system",
        evidence_text="Vitamin C contributes to immune defense by supporting cellular immune functions."
    )
    print("Test 2:", result)
