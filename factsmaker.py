import json
import requests
from tqdm import tqdm

# -------------------------------
# PATHS
# -------------------------------
INPUT_PATH = r"C:\Users\Shivam Kumar\frenemy\TruthCheck\data\verified_factsv2.json"
OUTPUT_PATH = r"C:\Users\Shivam Kumar\frenemy\TruthCheck\data\verified_facts_labelcorrect.json"

# -------------------------------
# OLLAMA CONFIG (same as yours)
# -------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"


# -------------------------------
# OLLAMA CALL FUNCTION
# -------------------------------
def get_label(claim, explanation):
    prompt = (
        "You are a strict medical fact-checking system.\n\n"
        f"Claim: {claim}\n"
        f"Scientific Explanation: {explanation}\n\n"
        "Task:\n"
        "- Determine if claim is TRUE, FALSE, or MIXED\n"
        "- Assign risk level: Low, Medium, High\n\n"
        "Rules:\n"
        "- Output EXACTLY 2 lines\n"
        "- Line 1: Verdict: TRUE or FALSE or MIXED\n"
        "- Line 2: Risk: Low or Medium or High\n"
        "- No extra text\n\n"
        "Answer:\n"
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 50,
                },
            },
            timeout=60,
        )

        response.raise_for_status()
        text = response.json().get("response", "").strip()

        # -------------------------------
        # PARSE OUTPUT
        # -------------------------------
        verdict = "UNVERIFIED"
        risk = "Medium"

        lines = text.split("\n")

        for line in lines:
            line_lower = line.lower()

            if "verdict" in line_lower:
                if "true" in line_lower:
                    verdict = "TRUE"
                elif "false" in line_lower:
                    verdict = "FALSE"
                elif "mixed" in line_lower:
                    verdict = "MIXED"

            if "risk" in line_lower:
                if "low" in line_lower:
                    risk = "Low"
                elif "medium" in line_lower:
                    risk = "Medium"
                elif "high" in line_lower:
                    risk = "High"

        return verdict, risk

    except Exception as e:
        print(f"Error: {e}")
        return "UNVERIFIED", "Medium"


# -------------------------------
# LOAD DATA
# -------------------------------
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

output = []

# -------------------------------
# PROCESS
# -------------------------------
for item in tqdm(data):
    claim = item.get("claim", "")
    explanation = item.get("explanation", "")

    verdict, risk = get_label(claim, explanation)

    new_entry = {
        "claim_subject": claim,
        "verdict": verdict,
        "scientific_truth": explanation,
        "risk_level": risk,
        "source": "multiple"
    }
    print(new_entry)
    output.append(new_entry)

# -------------------------------
# SAVE
# -------------------------------
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ Done! Saved to: {OUTPUT_PATH}")