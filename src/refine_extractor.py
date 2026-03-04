# import spacy

# def get_full_phrase(token):
#     # Grab token + modifiers (e.g., "drinking warm water" instead of just "water")
#     return "".join(
#         [t.text_with_ws for t in token.subtree if not t.is_punct]
#     ).strip()

# def debug_parse(doc):
#     print("\nToken           | Dep        | Head         | Pos")
#     print("----------------------------------------------------")
#     for token in doc:
#         print(f"{token.text:<15} | {token.dep_:<10} | {token.head.text:<12} | {token.pos_}")

# def extract_triples(text, nlp):
#     doc = nlp(text)
#     # debug_parse(doc) # Uncomment if you need to see the tags again

#     triples = []
    
#     # ACTION KEYWORDS: We use these to verify the ROOT is a meaningful health relation
#     action_keywords = [
#         "improves", "causes", "reduces", "cures", 
#         "prevents", "is", "bad", "good", "harms", "helps"
#     ]

#     for token in doc:
#         # STRATEGY: Only anchor on the ROOT verb of the sentence.
#         # This prevents "khali" or other mistagged words from creating ghost triples.
#         if token.dep_ == "ROOT" and (token.pos_ == "VERB" or token.text.lower() in action_keywords):
#             relation = token.text

#             # 1. Find Subject (nsubj)
#             subject = None
#             for left_token in token.lefts:
#                 if left_token.dep_ in ("nsubj", "nsubjpass", "csubj"):
#                     subject = get_full_phrase(left_token)
#                     break

#             # 2. Find Object (dobj, pobj, or attr)
#             objects = []
#             for right_token in token.rights:
#                 if right_token.dep_ in ("dobj", "pobj", "attr"):
#                     objects.append(get_full_phrase(right_token))

#             # FALLBACK: If standard extraction fails, check noun chunks
#             if not subject:
#                 # Look for chunks before the verb
#                 for chunk in doc.noun_chunks:
#                     if chunk.end <= token.i:
#                         subject = chunk.text
            
#             if not objects:
#                 # Look for chunks after the verb
#                 for chunk in doc.noun_chunks:
#                     if chunk.start > token.i:
#                         objects.append(chunk.text)

#             # Deduplicate and format results
#             for obj in list(set(objects)):
#                 if subject and obj:
#                     triples.append({
#                         "subject": subject,
#                         "relation": relation,
#                         "object": obj
#                     })

#     # If we still have multiple identical triples, deduplicate
#     unique_triples = [dict(t) for t in {tuple(d.items()) for d in triples}]
#     return unique_triples

# if __name__ == "__main__":
#     # For standalone testing
#     from lifestyle_ner import build_lifestyle_ner
#     nlp = spacy.load("en_core_web_sm")
#     nlp = build_lifestyle_ner(nlp)

#     sample = "Drinking warm water on an empty stomach improves digestion."
#     results = extract_triples(sample, nlp)

#     print("\n--- Final Results ---")
#     for r in results:
#         print(f"Triple Found: ({r['subject']}) -> [{r['relation']}] -> ({r['object']})")

import spacy

HEALTH_KEYWORDS = {
    "water", "digestion", "metabolism", "health", "body", "skin", "diet",
    "stomach", "heart", "blood", "liver", "kidney", "brain", "lung",
    "immune", "bone", "muscle", "weight", "fat", "sugar", "pressure",
    "cancer", "diabetes", "infection", "vitamin", "mineral", "protein",
    "fiber", "sleep", "stress", "exercise", "detox", "toxin", "hormone",
    "cholesterol", "inflammation", "antioxidant", "energy", "pain", "fever"
}

ACTION_KEYWORDS = {
    # Original
    "improves", "causes", "reduces", "cures", "prevents", "is",
    "bad", "good", "harms", "helps","boosts", "increases", "decreases", "aids", "supports", "damages",
    "weakens", "strengthens", "triggers", "treats", "heals", "fights",
    "promotes", "inhibits", "stimulates", "regulates", "affects",
    "lowers", "raises", "blocks", "enhances", "speeds", "slows",
    "protects", "risks", "worsens", "relieves", "detoxifies", "cleanses"
}


def get_full_phrase(token):
    """
    Returns the full phrase for a token including its modifiers.
    e.g. for 'water' in 'warm water' → returns 'warm water'
    Punctuation is excluded to keep phrases clean.
    """
    return "".join(
        [t.text_with_ws for t in token.subtree if not t.is_punct]
    ).strip()


def is_health_sentence(text):
    """
    Quick keyword check to decide if a sentence is worth extracting from.
    Prevents non-health sentences like 'It is a common routine in households'
    from generating fake triples via the fallback noun chunk path.
    """
    text_lower = text.lower()
    return any(kw in text_lower for kw in HEALTH_KEYWORDS)


def get_relation_phrase(token):
    """
    Builds a richer relation string instead of just the bare root verb.
    Captures negations and particles so we don't lose meaning.

    Examples:
        'does not improve'  →  'does not improve'  (negation captured)
        'helps boost'       →  'helps boost'        (xcomp captured)
        'improves'          →  'improves'            (simple case)
    """
    parts = []

    # Capture negation (e.g. "does not")
    for child in token.children:
        if child.dep_ == "neg":
            parts.append(child.text)

    parts.append(token.text)

    # Capture verb particle / xcomp (e.g. "helps boost" → include "boost")
    for child in token.children:
        if child.dep_ in ("prt", "xcomp") and child.pos_ == "VERB":
            parts.append(child.text)

    return " ".join(parts)


def debug_parse(doc):
    """Prints a dependency parse table — uncomment the call in extract_triples to use."""
    print("\nToken           | Dep        | Head         | Pos")
    print("----------------------------------------------------")
    for token in doc:
        print(f"{token.text:<15} | {token.dep_:<10} | {token.head.text:<12} | {token.pos_}")


def extract_triples(text, nlp):
    """
    Extracts Subject → Relation → Object health claim triples from a sentence.

    Strategy:
    1. Health keyword guard — skip non-health sentences immediately.
    2. Anchor only on ROOT verb (same as before, prevents ghost triples).
    3. Expanded action keyword list — catches more health verbs.
    4. Richer relation phrase — captures negations and particles.
    5. Prepositional object support (prep → pobj) — was missing before.
    6. Smarter fallback — noun chunks only used when standard parse fails,
       and only on sentences that passed the health guard.
    7. Fixed deduplication — old set(tuple()) was order-sensitive and buggy.
    """
    # GUARD: Skip sentences with no health content at all
    if not is_health_sentence(text):
        return []

    doc = nlp(text)
    # debug_parse(doc)  # Uncomment to inspect dependency tags

    triples = []

    for token in doc:
        # Anchor on ROOT verb only — same logic as before, still correct
        if token.dep_ == "ROOT" and (
            token.pos_ == "VERB" or token.text.lower() in ACTION_KEYWORDS
        ):
            # Build richer relation (captures negation + particles)
            relation = get_relation_phrase(token)

            # --- 1. Find Subject ---
            subject = None
            for left_token in token.lefts:
                if left_token.dep_ in ("nsubj", "nsubjpass", "csubj"):
                    subject = get_full_phrase(left_token)
                    break

            # --- 2. Find Object ---
            # Now also handles: prep → pobj (e.g. "helps WITH digestion")
            objects = []
            for right_token in token.rights:
                if right_token.dep_ in ("dobj", "attr", "acomp"):
                    objects.append(get_full_phrase(right_token))
                elif right_token.dep_ == "prep":
                    # Dig one level deeper for prepositional objects
                    for pobj in right_token.children:
                        if pobj.dep_ == "pobj":
                            objects.append(get_full_phrase(pobj))

            # --- FALLBACK: Noun chunks (only if standard parse failed) ---
            # Only runs if we're already inside a health sentence (guard above)
            if not subject:
                for chunk in doc.noun_chunks:
                    if chunk.end <= token.i:
                        subject = chunk.text  # take the last chunk before verb

            if not objects:
                for chunk in doc.noun_chunks:
                    if chunk.start > token.i:
                        objects.append(chunk.text)

            #triples
            for obj in objects:
                if subject and obj and subject.lower() != obj.lower():
                    triples.append({
                        "subject": subject.strip(),
                        "relation": relation.strip(),
                        "object": obj.strip()
                    })
    seen = set()
    unique_triples = []
    for t in triples:
        key = (t["subject"].lower(), t["relation"].lower(), t["object"].lower())
        if key not in seen:
            seen.add(key)
            unique_triples.append(t)

    return unique_triples


#testc
if __name__ == "__main__":
    from lifestyle_ner import build_lifestyle_ner

    nlp = spacy.load("en_core_web_sm")
    nlp = build_lifestyle_ner(nlp)

    test_cases = [
        "Drinking warm water on an empty stomach improves digestion.",         # standard
        "Warm water does not cure diabetes.",                                   # negation
        "It is a common routine in many households.",                           # non-health → should return []
        "Exercise helps boost metabolism and reduces blood pressure.",          # multi-object
        "Sugar increases cholesterol levels in the body.",                      # prep object
    ]

    for sample in test_cases:
        print(f"\nInput: {sample}")
        results = extract_triples(sample, nlp)
        if not results:
            print("  → No health triples extracted (non-health sentence or parse failed)")
        for r in results:
            print(f"  Triple: ({r['subject']}) -> [{r['relation']}] -> ({r['object']})")