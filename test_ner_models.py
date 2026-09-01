"""Test different NER models to find one that works."""

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

models_to_test = [
    "Davlan/bert-base-multilingual-cased-ner",
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
]

test_text = "Metformin 500mg should not be stopped."

for model_name in models_to_test:
    print(f"\nTesting: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        nlp = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        result = nlp(test_text)
        print(f"✓ SUCCESS: Model has valid TokenClassification head")
        print(f"  Config: {model.config.architectures}")
        print(f"  Tokens found: {len(result)}")
        for item in result[:3]:
            print(f"    - {item['word']}: {item['entity']}")
        break
    except Exception as e:
        print(f"✗ Failed: {type(e).__name__}: {str(e)[:100]}")
