import spacy

def get_full_phrase(token):
    # Grab token + modifiers (e.g., "drinking warm water" instead of just "water")
    return "".join(
        [t.text_with_ws for t in token.subtree if not t.is_punct]
    ).strip()

def debug_parse(doc):
    print("\nToken           | Dep        | Head         | Pos")
    print("----------------------------------------------------")
    for token in doc:
        print(f"{token.text:<15} | {token.dep_:<10} | {token.head.text:<12} | {token.pos_}")

def extract_triples(text, nlp):
    doc = nlp(text)
    # debug_parse(doc) # Uncomment if you need to see the tags again

    triples = []
    
    # ACTION KEYWORDS: We use these to verify the ROOT is a meaningful health relation
    action_keywords = [
        "improves", "causes", "reduces", "cures", 
        "prevents", "is", "bad", "good", "harms", "helps"
    ]

    for token in doc:
        # STRATEGY: Only anchor on the ROOT verb of the sentence.
        # This prevents "khali" or other mistagged words from creating ghost triples.
        if token.dep_ == "ROOT" and (token.pos_ == "VERB" or token.text.lower() in action_keywords):
            relation = token.text

            # 1. Find Subject (nsubj)
            subject = None
            for left_token in token.lefts:
                if left_token.dep_ in ("nsubj", "nsubjpass", "csubj"):
                    subject = get_full_phrase(left_token)
                    break

            # 2. Find Object (dobj, pobj, or attr)
            objects = []
            for right_token in token.rights:
                if right_token.dep_ in ("dobj", "pobj", "attr"):
                    objects.append(get_full_phrase(right_token))

            # FALLBACK: If standard extraction fails, check noun chunks
            if not subject:
                # Look for chunks before the verb
                for chunk in doc.noun_chunks:
                    if chunk.end <= token.i:
                        subject = chunk.text
            
            if not objects:
                # Look for chunks after the verb
                for chunk in doc.noun_chunks:
                    if chunk.start > token.i:
                        objects.append(chunk.text)

            # Deduplicate and format results
            for obj in list(set(objects)):
                if subject and obj:
                    triples.append({
                        "subject": subject,
                        "relation": relation,
                        "object": obj
                    })

    # If we still have multiple identical triples, deduplicate
    unique_triples = [dict(t) for t in {tuple(d.items()) for d in triples}]
    return unique_triples

if __name__ == "__main__":
    # For standalone testing
    from lifestyle_ner import build_lifestyle_ner
    nlp = spacy.load("en_core_web_sm")
    nlp = build_lifestyle_ner(nlp)

    sample = "Drinking warm water on an empty stomach improves digestion."
    results = extract_triples(sample, nlp)

    print("\n--- Final Results ---")
    for r in results:
        print(f"Triple Found: ({r['subject']}) -> [{r['relation']}] -> ({r['object']})")