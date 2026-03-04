import spacy
#no data used
def extract_health_triples(text, nlp):
    doc = nlp(text)
    triples = []

    for token in doc:
        # 1. We look for a Subject (nsubj)
        if token.dep_ == "nsubj":
            subject = token.text
            # The 'head' of the subject is usually the Verb/Relation
            relation = token.head.text
            
            # 2. We look for the Object (dobj, pobj, or attribute)
            # We search among the children of the verb/head
            obj = None
            for child in token.head.children:
                if child.dep_ in ("dobj", "pobj", "attr", "acomp"):
                    # For complex phrases (like "bad for health"), 
                    # we can take the whole subtree of the object
                    obj = "".join([w.text_with_ws for w in child.subtree]).strip()
            
            if subject and relation and obj:
                triples.append({
                    "subject": subject,
                    "relation": relation,
                    "object": obj
                })
    
    return triples

# Example Usage:
# nlp = spacy.load("en_core_sci_sm")
# print(extract_health_triples("Gym is bad for heart", nlp))
# Output: [{'subject': 'Gym', 'relation': 'is', 'object': 'bad for heart'}]