import spacy
from spacy.pipeline import EntityRuler
import os

try:
    from config import PATTERNS_FILE
except ImportError:
    PATTERNS_FILE = "data/lifestyle_patterns.jsonl"

def build_lifestyle_ner(nlp):
    """
    Adds an EntityRuler to the spaCy pipeline and loads patterns 
    from an external JSONL file.
    """
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        ruler = nlp.get_pipe("entity_ruler")

    # 2. Check if the patterns file exists
    if os.path.exists(PATTERNS_FILE):
        try:
            # Clear existing patterns to avoid duplicates on reload
            ruler.clear() 
            # Load patterns from disk (JSONL format)
            ruler.from_disk(PATTERNS_FILE)
            print(f"--- [Success] Lifestyle NER: Loaded patterns from {PATTERNS_FILE} ---")
        except Exception as e:
            print(f"--- [Error] Failed to load patterns: {e} ---")
    else:
        print(f"--- [Warning] {PATTERNS_FILE} not found. Custom NER will be empty. ---")

    return nlp

if __name__ == "__main__":
    # Local Testing
    nlp = spacy.load("en_core_web_sm")
    nlp = build_lifestyle_ner(nlp)
    
    test_text = "Drinking warm water on an empty stomach improves digestion."
    doc = nlp(test_text)
    
    print("\nTesting NER Output:")
    for ent in doc.ents:
        print(f"Entity: {ent.text} | Label: {ent.label_}")
