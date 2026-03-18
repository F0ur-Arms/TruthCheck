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
    "bad", "good", "harms", "helps", "boosts", "increases", "decreases", "aids", "supports", "damages",
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


# ---------------------------------------------------------------------------
# FIX 1: CONJUNCTION EXPANSION
# Recursively collects all conjuncts of a token.
# e.g. "boosts metabolism and energy" → [metabolism, energy]
# Without this, "energy" is missed because it's a conj child of "metabolism"
# not a direct child of the verb.
# ---------------------------------------------------------------------------
def expand_conjuncts(token):
    """
    Returns a list of the token plus all its conjunct children recursively.
    Handles: "metabolism and energy", "liver, kidney and heart" etc.
    """
    results = [token]
    for child in token.children:
        if child.dep_ == "conj":
            results.extend(expand_conjuncts(child))
    return results


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
    5. Prepositional object support (prep → pobj).
    6. Smarter fallback — noun chunks only used when standard parse fails.
    7. Fixed deduplication.

    NEW ADDITIONS:
    8. Conjunction expansion — catches all objects in "boosts X and Y and Z"
    9. Passive voice handling — flips subject/object for "X is caused by Y"
   10. Lemma stored on triple — enables robust matching against KB
    """
    # GUARD: Skip sentences with no health content at all
    if not is_health_sentence(text):
        return []

    doc = nlp(text)
    # debug_parse(doc)  # Uncomment to inspect dependency tags

    triples = []

    for token in doc:
        # Anchor on ROOT verb only
        if token.dep_ == "ROOT" and (
            token.pos_ == "VERB" or token.text.lower() in ACTION_KEYWORDS
        ):
            relation = get_relation_phrase(token)

            # ---------------------------------------------------------------------------
            # FIX 3: LEMMA STORAGE
            # Store the lemma of the root verb so downstream matching can normalize
            # "improving" → "improve", "caused" → "cause" etc.
            # This allows KB entries written with base forms to match inflected verbs.
            # ---------------------------------------------------------------------------
            relation_lemma = token.lemma_.lower()

            # ---------------------------------------------------------------------------
            # FIX 2: PASSIVE VOICE DETECTION
            # Detect passive construction by checking for nsubjpass dependency.
            # In "Diabetes is caused by sugar":
            #   - nsubjpass = "Diabetes" (grammatical subject but semantic object)
            #   - pobj of "by" = "sugar" (grammatical object but semantic subject)
            # We detect this and SWAP them so the triple reads:
            #   (sugar) -> [causes] -> (diabetes)  ← semantically correct
            # ---------------------------------------------------------------------------
            is_passive = any(child.dep_ == "nsubjpass" for child in token.children)

            # --- 1. Find Subject ---
            subject = None
            passive_grammatical_subject = None

            for left_token in token.lefts:
                if left_token.dep_ in ("nsubj", "nsubjpass", "csubj"):
                    if is_passive and left_token.dep_ == "nsubjpass":
                        # In passive voice this is actually the semantic OBJECT
                        passive_grammatical_subject = get_full_phrase(left_token)
                    else:
                        subject = get_full_phrase(left_token)
                    break

            # --- 2. Find Object ---
            objects = []
            passive_agent = None

            for right_token in token.rights:
                if right_token.dep_ in ("dobj", "attr", "acomp"):
                    # FIX 1: Expand conjuncts for each direct object
                    for conjunct in expand_conjuncts(right_token):
                        objects.append(get_full_phrase(conjunct))

                elif right_token.dep_ == "prep":
                    prep_text = right_token.text.lower()
                    for pobj in right_token.children:
                        if pobj.dep_ == "pobj":
                            if is_passive and prep_text == "by":
                                # "by sugar" in passive → semantic SUBJECT
                                passive_agent = get_full_phrase(pobj)
                            else:
                                # FIX 1: Expand conjuncts for prepositional objects too
                                for conjunct in expand_conjuncts(pobj):
                                    objects.append(get_full_phrase(conjunct))

            # --- PASSIVE SWAP ---
            if is_passive:
                if passive_agent and passive_grammatical_subject:
                    subject = passive_agent
                    objects = [passive_grammatical_subject] + objects
                elif passive_grammatical_subject and not subject:
                    subject = passive_grammatical_subject

            # --- FALLBACK: Noun chunks (only if standard parse failed) ---
            if not subject:
                for chunk in doc.noun_chunks:
                    if chunk.end <= token.i:
                        subject = chunk.text

            if not objects:
                for chunk in doc.noun_chunks:
                    if chunk.start > token.i:
                        objects.append(chunk.text)

            # Build triples — now includes relation_lemma field
            for obj in objects:
                if subject and obj and subject.lower() != obj.lower():
                    triples.append({
                        "subject":        subject.strip(),
                        "relation":       relation.strip(),
                        "relation_lemma": relation_lemma,
                        "object":         obj.strip()
                    })

    # Deduplication — uses lemma for key to deduplicate across inflections
    seen = set()
    unique_triples = []
    for t in triples:
        key = (t["subject"].lower(), t["relation_lemma"], t["object"].lower())
        if key not in seen:
            seen.add(key)
            unique_triples.append(t)

    return unique_triples


# ---------------------------------------------------------------------------
# Standalone testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    test_cases = [
        # FIX 1: Conjunction — should get metabolism AND energy
        "Exercise boosts metabolism and energy.",
        # FIX 1: Multiple conjuncts
        "Turmeric improves digestion, immunity and skin health.",
        # FIX 2: Passive voice — sugar should be subject, diabetes object
        "Diabetes is caused by sugar.",
        # FIX 2: Passive voice
        "Cancer is triggered by smoking and stress.",
        # FIX 3: Lemma — "improving" should lemmatize to "improve"
        "Warm water is improving digestion.",
        # Original cases still work
        "Drinking warm water on an empty stomach improves digestion.",
        "Warm water does not cure diabetes.",
        # Non-health → should return []
        "It is a common routine in many households.",
    ]

    for sample in test_cases:
        print(f"\nInput: {sample}")
        results = extract_triples(sample, nlp)
        if not results:
            print("  → No health triples extracted")
        for r in results:
            print(f"  Triple : ({r['subject']}) -> [{r['relation']}] -> ({r['object']})")
            print(f"  Lemma  : {r['relation_lemma']}")
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