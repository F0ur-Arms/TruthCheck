import spacy
from src.preprocessor import HinglishMapper
from src.lifestyle_ner import build_lifestyle_ner
from src.refine_extractor import extract_triples

def run_full_test():
    # 1. Initialize Components
    print("Initializing Pipeline...")
    mapper = HinglishMapper()
    nlp = spacy.load("en_core_sci_sm")
    nlp = build_lifestyle_ner(nlp) # Your Objective 1
    
    # 2. Test Cases (Hinglish + Medical + Habits)
    test_queries = [
        "Subah khali pet warm water reduces fat",
        "Going to gym is bad for heart",
        "Papaya seeds cause abortion",
        "Haldi milk improves immunity",
        "Sleeping at noon causes weight gain"
    ]

    print("\n" + "="*50)
    print("TRUTHCHECK EXTRACTION TEST")
    print("="*50)

    for query in test_queries:
        # Step A: Preprocess (Hinglish -> English)
        cleaned_text = mapper.clean_text(query)
        
        # Step B: Extract Triples (Objective 2)
        triples = extract_triples(cleaned_text, nlp)
        
        print(f"\nOriginal: {query}")
        print(f"Mapped:   {cleaned_text}")
        
        if not triples:
            print("Result:   No Triple Found")
        else:
            for t in triples:
                print(f"Result:   Subject({t['subject']}) -> Relation({t['relation']}) -> Object({t['object']})")
    print("="*50)

if __name__ == "__main__":
    run_full_test()