import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import NLI_MODEL

class NLIVerifier:
    def __init__(self, model_name=NLI_MODEL):
        print(f"--- Loading NLI Model: {model_name} ---")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # NLI label ids differ between model checkpoints.  Use the checkpoint's
        # metadata instead of assuming the old DistilRoBERTa ordering.
        aliases = {
            "contradiction": "REFUTES",
            "contradict": "REFUTES",
            "neutral": "NEI",
            "entailment": "SUPPORTS",
            "entails": "SUPPORTS",
        }
        self.id2label = {
            int(label_id): aliases.get(str(label).lower(), "NEI")
            for label_id, label in self.model.config.id2label.items()
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

        label_probabilities = {
            self.id2label[int(i)]: round(probs[0][int(i)].item(), 4)
            for i in range(probs.shape[1])
        }
        supports_probability = label_probabilities.get("SUPPORTS", 0.0)
        refutes_probability = label_probabilities.get("REFUTES", 0.0)

        return {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "label_probabilities": label_probabilities,
            "supports_probability": supports_probability,
            "refutes_probability": refutes_probability,
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
