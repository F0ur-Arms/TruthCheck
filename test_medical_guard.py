"""Test the fixed medical entity guard."""

from multilingual.medical_entity_guard import MedicalEntityGuard
from multilingual.token_lang_detector import TokenLanguageDetector, HinglishLIDClassifier
from multilingual.types import TokenTag

guard = MedicalEntityGuard()

# Test cases
test_cases = [
    ("Metformin 500mg should not be stopped without medical advice.", 
     ["Metformin", "500mg"]),
    
    ("protein kidney ke liye kharab hai",
     ["protein"]),
    
    ("haldi cancer ko 100% cure karti hai",
     ["100%"]),
     
    ("doctor se bina pooche metformin 500mg band mat karo",
     ["doctor", "metformin", "500mg"]),
]

print("=" * 70)
print("MEDICAL ENTITY GUARD TEST")
print("=" * 70)

for text, expected in test_cases:
    # Create token tags (simulate MuRIL output)
    detector = TokenLanguageDetector(HinglishLIDClassifier())
    tags = detector.tag(text)
    
    # Extract entities
    entities = guard.protected_entities(text, tags)
    
    print(f"\nInput: {text}")
    print(f"Expected to find: {expected}")
    print(f"Found: {sorted(entities)}")
    
    # Check if key entities are present
    found_all = all(any(exp.lower() in ent.lower() for ent in entities) for exp in expected)
    status = "✓ PASS" if found_all else "⚠ PARTIAL"
    print(f"Status: {status}")

print("\n" + "=" * 70)
