import spacy

# ─────────────────────────────────────────────────────────────────────────────
# ACTION_KEYWORDS
# Used for compound-chain detection AND verb-anchor matching.
# ─────────────────────────────────────────────────────────────────────────────
ACTION_KEYWORDS = {
    "improves", "causes", "reduces", "cures", "prevents",
    "harms", "helps", "boosts", "increases", "decreases",
    "aids", "supports", "damages", "weakens", "strengthens", "triggers",
    "treats", "heals", "fights", "promotes", "inhibits", "stimulates",
    "regulates", "affects", "lowers", "raises", "blocks", "enhances",
    "speeds", "slows", "protects", "risks", "worsens", "relieves",
    "detoxifies", "cleanses", "damage", "cause", "cure", "prevent",
    "improve", "boost", "increase", "decrease", "harm", "help", "reduce",
    "treat", "heal", "fight", "promote", "inhibit", "stimulate", "regulate",
    "affect", "lower", "raise", "block", "enhance", "worsen", "relieve",
    "provide", "contains", "contain", "lead", "leads", "kill", "kills",
    "spread", "spreads", "sell", "sold", "made", "make", "destroy",
    "destroys", "melts", "melt", "whitens", "whiten", "burns", "burn",
    "cleans", "clean", "flushes", "flush", "purifies", "purify",
    "removes", "remove", "repairs", "repair", "restores", "restore",
    "infects", "linked", "associated", "connected",
    "are", "was", "were", "be", "been", "being",
}

COPULA_LEMMAS = {"be"}

TRIVIAL = {
    "it", "this", "that", "they", "he", "she", "we", "i", "them",
    "which", "who", "what", "there", "these", "those",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_full_phrase(token):
    return "".join(
        t.text_with_ws for t in token.subtree if not t.is_punct
    ).strip()


def get_relation_phrase(token):
    parts = []
    for child in token.children:
        if child.dep_ == "neg":
            parts.append(child.text)
    parts.append(token.text)
    for child in token.children:
        if child.dep_ in ("prt", "xcomp") and child.pos_ == "VERB":
            parts.append(child.text)
    return " ".join(parts)


def expand_conjuncts(token):
    results = [token]
    for child in token.children:
        if child.dep_ == "conj":
            results.extend(expand_conjuncts(child))
    return results


def make_triple(subj, rel, rel_lemma, obj):
    """Validates and returns a triple dict, or None if invalid."""
    if not subj or not obj:
        return None
    subj = subj.strip()
    obj  = obj.strip()
    if not subj or not obj:
        return None
    if subj.lower() == obj.lower():
        return None
    if obj.lower() in TRIVIAL:
        return None
    return {
        "subject":        subj,
        "relation":       rel.strip(),
        "relation_lemma": rel_lemma.lower(),
        "object":         obj,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1 — COMPOUND CHAIN RESOLVER
#
# THE PRIMARY FIX. Handles ~80% of failures.
#
# spaCy consistently misparsed short health claim sentences as compound noun
# chains. The diagnostic showed the dominant pattern:
#
#   "beer melts kidney stones"
#     beer/NOUN/compound → melts/NOUN/compound → stones/NOUN/ROOT
#
#   "paper cups block arteries"
#     paper/NOUN/compound → cups/NOUN/compound → block/NOUN/compound → arteries/NOUN/ROOT
#
# The action verb is tagged NOUN with dep=compound. We walk the compound
# chain of the ROOT, find the action keyword token, and split there:
#   tokens before action_token → subject
#   action_token               → relation
#   tokens after + ROOT        → object
# ─────────────────────────────────────────────────────────────────────────────

def collect_compounds(token):
    """Recursively collect compound chain ending at token."""
    chain = []
    for child in token.children:
        if child.dep_ == "compound":
            chain.extend(collect_compounds(child))
    chain.append(token)
    return chain


def resolve_compound_chain(doc):
    triples = []

    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return triples

    chain = collect_compounds(root)

    # Find leftmost action keyword in chain
    action_idx = None
    for i, token in enumerate(chain):
        if token.text.lower() in ACTION_KEYWORDS or token.lemma_.lower() in ACTION_KEYWORDS:
            action_idx = i
            break

    # Need at least one token before action (subject) and action not at end
    if action_idx is None or action_idx == 0:
        return triples

    action_token = chain[action_idx]
    subj_chain   = chain[:action_idx]
    obj_chain    = chain[action_idx + 1:]

    # Build subject from subtrees of subject-side tokens
    subj_ids  = set()
    subj_text = []
    for t in subj_chain:
        for st in t.subtree:
            if st.i not in subj_ids and not st.is_punct:
                subj_text.append(st.text_with_ws)
                subj_ids.add(st.i)
    subject = " ".join("".join(subj_text).split())

    # Relation: include negation if any
    neg      = next((c.text for c in action_token.children if c.dep_ == "neg"), None)
    relation = f"{neg} {action_token.text}" if neg else action_token.text
    rel_lem  = action_token.lemma_.lower()

    # Build object from obj-side chain tokens + ROOT's prep/npadvmod children
    obj_ids  = set()
    obj_text = []
    for t in obj_chain:
        for st in t.subtree:
            if st.i not in obj_ids and not st.is_punct:
                obj_text.append(st.text_with_ws)
                obj_ids.add(st.i)

    # Also sweep ROOT's children for prep/npadvmod objects to the right of action
    for child in root.children:
        if child.dep_ == "prep" and child.i > action_token.i:
            for pobj in child.children:
                if pobj.dep_ == "pobj":
                    for st in pobj.subtree:
                        if st.i not in obj_ids and not st.is_punct:
                            obj_text.append(st.text_with_ws)
                            obj_ids.add(st.i)
        if child.dep_ in ("npadvmod", "dobj") and child.i > action_token.i:
            if child.i not in obj_ids and not child.is_punct:
                for st in child.subtree:
                    if st.i not in obj_ids and not st.is_punct:
                        obj_text.append(st.text_with_ws)
                        obj_ids.add(st.i)

    obj_str = " ".join("".join(obj_text).split())

    # Fallback: if obj still empty, use ROOT itself
    if not obj_str and root.i not in subj_ids:
        obj_str = root.text

    t = make_triple(subject, relation, rel_lem, obj_str)
    if t:
        triples.append(t)

    # Handle conjunctions on ROOT (e.g. "kidneys and liver")
    for conj in root.children:
        if conj.dep_ == "conj" and conj.i not in subj_ids:
            t2 = make_triple(subject, relation, rel_lem, get_full_phrase(conj))
            if t2:
                triples.append(t2)

    return triples


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2 — VERB ANCHOR (standard parse)
#
# Handles sentences spaCy parsed correctly with a VERB ROOT:
#   "mustard oil on feet helps eyesight"   ROOT=helps/VERB
#   "baby food contains sawdust"           ROOT=contains/VERB
#   "eat a healthy diet and avoid sugar"   ROOT=eat/VERB
#   "implementing the international code..." ROOT=implementing/VERB
# ─────────────────────────────────────────────────────────────────────────────

def extract_verb_anchor_triples(doc):
    triples = []
    seen    = set()

    for token in doc:
        is_root_verb   = (token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"))
        is_action_verb = (token.pos_ == "VERB" and
                          token.text.lower() in ACTION_KEYWORDS and
                          token.dep_ not in ("aux", "auxpass", "punct"))
        is_advcl       = (token.dep_ in ("advcl", "relcl", "ccomp") and
                          token.text.lower() in ACTION_KEYWORDS)

        if not (is_root_verb or is_action_verb or is_advcl):
            continue
        if token.i in seen:
            continue
        seen.add(token.i)

        relation  = get_relation_phrase(token)
        rel_lem   = token.lemma_.lower()
        is_passive = any(c.dep_ == "nsubjpass" for c in token.children)

        # Subject
        subject   = None
        pass_subj = None
        for lt in token.lefts:
            if lt.dep_ in ("nsubj", "csubj") and not lt.text.lower() in TRIVIAL:
                subject = get_full_phrase(lt)
                break
            if lt.dep_ == "nsubjpass":
                pass_subj = get_full_phrase(lt)
                break

        # Compound subject fallback
        if not subject and not is_passive:
            parts = []
            for lt in token.lefts:
                if lt.dep_ in ("compound", "amod", "nmod"):
                    parts.append(get_full_phrase(lt))
            if parts:
                subject = " ".join(parts)

        if not subject and not is_passive:
            for chunk in doc.noun_chunks:
                if chunk.end <= token.i:
                    subject = chunk.text

        # Objects
        objects    = []
        pass_agent = None

        for rt in token.rights:
            if rt.dep_ in ("dobj", "attr", "acomp"):
                for conj in expand_conjuncts(rt):
                    objects.append(get_full_phrase(conj))
            elif rt.dep_ == "prep":
                for pobj in rt.children:
                    if pobj.dep_ == "pobj":
                        if is_passive and rt.text.lower() == "by":
                            pass_agent = get_full_phrase(pobj)
                        else:
                            for conj in expand_conjuncts(pobj):
                                objects.append(get_full_phrase(conj))
            elif rt.dep_ == "xcomp":
                for xc in rt.children:
                    if xc.dep_ in ("dobj", "attr", "acomp"):
                        objects.append(get_full_phrase(xc))
                if not any(c.dep_ in ("dobj", "attr", "acomp") for c in rt.children):
                    objects.append(get_full_phrase(rt))
            elif rt.dep_ == "ccomp":
                objects.append(get_full_phrase(rt))
            elif rt.dep_ == "advmod" and rt.pos_ not in ("ADV", "PUNCT"):
                objects.append(rt.text)

        # Passive swap
        if is_passive:
            if pass_agent and pass_subj:
                subject = pass_agent
                objects = [pass_subj] + objects
            elif pass_subj:
                subject = pass_subj
                for chunk in doc.noun_chunks:
                    if chunk.start > token.i:
                        objects.append(chunk.text)
                        break

        if not objects:
            for chunk in doc.noun_chunks:
                if chunk.start > token.i:
                    objects.append(chunk.text)

        for obj in objects:
            t = make_triple(subject, relation, rel_lem, obj)
            if t:
                triples.append(t)

    return triples


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 3 — COPULA + ADJ/NOUN PREDICATE
#
# Handles: "apples at night are toxic", "artificial sweeteners are safe"
# spaCy tags predicate adj/noun with dep=acomp or attr.
# ─────────────────────────────────────────────────────────────────────────────

def extract_copula_triples(doc):
    triples = []

    for token in doc:
        if token.dep_ not in ("acomp", "attr"):
            continue
        head = token.head
        if head.lemma_ not in COPULA_LEMMAS:
            continue

        relation = head.text
        rel_lem  = head.lemma_.lower()

        subject = None
        for child in head.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subject = get_full_phrase(child)
                break
        if not subject and head.dep_ == "aux":
            for child in head.head.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject = get_full_phrase(child)
                    break
        if not subject:
            for chunk in doc.noun_chunks:
                if chunk.end <= head.i:
                    subject = chunk.text

        t = make_triple(subject, relation, rel_lem, get_full_phrase(token))
        if t:
            triples.append(t)

    return triples


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 4 — INVERTED PARSE
#
# Handles the pattern where spaCy makes the health outcome the ROOT VERB
# and the action keyword ends up as nsubj:
#
#   "ladyfinger water cures diabetes"
#     cures/NOUN/nsubj → diabetes/VERB/ROOT
#
#   "hot water cures acne"
#     cures/NOUN/nsubj → acne/VERB/ROOT
#
#   "aloe vera cures aids hiv"
#     cures/NOUN/nsubj → aids/PROPN/ROOT
# ─────────────────────────────────────────────────────────────────────────────

def extract_inverted_parse_triples(doc):
    triples = []

    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return triples

    action_nsubj = next(
        (c for c in root.children
         if c.dep_ == "nsubj" and c.text.lower() in ACTION_KEYWORDS),
        None
    )
    if action_nsubj is None:
        return triples

    real_verb = action_nsubj
    relation  = real_verb.text
    rel_lem   = real_verb.lemma_.lower()

    # Subject: compound children of the action nsubj
    subj_parts = []
    for child in real_verb.children:
        if child.dep_ in ("compound", "amod", "nmod"):
            for st in child.subtree:
                if not st.is_punct:
                    subj_parts.append(st.text_with_ws)
    subject = "".join(subj_parts).strip() if subj_parts else None

    # Fallback subject: noun chunks before root
    if not subject:
        for chunk in doc.noun_chunks:
            if chunk.end < root.i and real_verb.i not in range(chunk.start, chunk.end):
                subject = chunk.text
                break

    # Object: ROOT text + its right dependents
    obj_parts = [root.text]
    for child in root.children:
        if child.dep_ in ("npadvmod", "advmod", "dobj") and child.i != real_verb.i:
            obj_parts.append(child.text)
    obj_str = " ".join(obj_parts)

    t = make_triple(subject, relation, rel_lem, obj_str)
    if t:
        triples.append(t)

    return triples


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 5 — NOUN-CHUNK PAIR FALLBACK
#
# Last resort for pure fragments: "dark circles liver failure"
# Only fires if all other strategies returned nothing.
# ─────────────────────────────────────────────────────────────────────────────

def extract_noun_chunk_pairs(doc):
    chunks = [
        c for c in doc.noun_chunks
        if c.text.lower().strip() not in TRIVIAL
        and not all(t.is_stop for t in c)
    ]
    if len(chunks) < 2:
        return []

    subject = chunks[0].text.strip()
    obj     = chunks[-1].text.strip()

    if subject.lower() == obj.lower() and len(chunks) >= 3:
        obj = chunks[1].text.strip()
    elif subject.lower() == obj.lower():
        return []

    t = make_triple(subject, "associated with", "associate", obj)
    return [t] if t else []


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def extract_triples(text, nlp):
    """
    Extracts Subject → Relation → Object triples from health claim text.

    5-strategy pipeline in priority order:
      1. Compound chain resolver  — spaCy NOUN/compound misparse (primary fix)
      2. Verb anchor              — correctly parsed VERB ROOT sentences
      3. Copula + predicate       — "X is/are toxic/safe"
      4. Inverted parse           — action verb mislabelled as nsubj of ROOT
      5. Noun-chunk pair fallback — pure fragments, fires only if 1-4 fail
    """
    doc = nlp(text)

    triples = []
    triples.extend(resolve_compound_chain(doc))
    triples.extend(extract_verb_anchor_triples(doc))
    triples.extend(extract_copula_triples(doc))
    triples.extend(extract_inverted_parse_triples(doc))

    if not triples:
        triples.extend(extract_noun_chunk_pairs(doc))

    # Deduplicate
    seen   = set()
    unique = []
    for t in triples:
        key = (t["subject"].lower(), t["relation_lemma"], t["object"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(t)

    return unique


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    test_cases = [
        
        "implementing the international code of marketing of breast milk substitutes and subsequent relevant world health assembly resolutions .",
        "free from microbial and chemical contaminants.",
        "developing school policies and programmes that encourage and enable children to adopt and maintain a healthy diet .",
        "providing nutrition and dietary counselling at primary health care facilities .",
        "educating children, adolescents and adults about nutrition and healthy dietary practices .",
        "eat a healthy diet and avoid sugar and saturated fat.",
        "reach and keep a health body weight.",
        "needing to urinate more often than usual.",
    ]

    success  = 0
    fail     = 0
    failures = []

    print(f"\nTesting {len(test_cases)} previously failing sentences\n")
    print(f"{'#':<4} {'Sentence':<52} Result")
    print("─" * 115)

    for i, sent in enumerate(test_cases, 1):
        results = extract_triples(sent, nlp)
        display = (sent[:49] + "...") if len(sent) > 52 else sent
        if results:
            success += 1
            for r in results:
                print(f"{i:<4} {display:<52} ({r['subject']}) → [{r['relation']}] → ({r['object']})")
        else:
            fail += 1
            failures.append(sent)
            print(f"{i:<4} {display:<52} ⚠  NO TRIPLE")

    print("─" * 115)
    print(f"\nResult: {success}/{len(test_cases)} extracted  |  {fail} still failing")
    if failures:
        print("\nStill failing:")
        for f in failures:
            print(f"  • {f}")