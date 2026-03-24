"""
debug_parse.py
Logs spaCy parsing output for failing sentences.
"""

import spacy

LOG_PATH = r"C:\Users\Shivam Kumar\frenemy\TruthCheck\src\loginput.txt"

def log(msg=""):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Initialize log file
with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("DEBUG PARSE RUN\n")
    f.write("=" * 90 + "\n\n")

nlp = spacy.load("en_core_web_sm")

FAILING = [
    "green stool illness cold.",
    "yellow food worsens jaundice.",
    "sniffing chilli cures migraines.",
    "curd in cold causes lung failure.",
    "warm water burns calories.",
    "implementing the international code of marketing of breast milk substitutes and subsequent relevant world health assembly resolutions .",
    "free from microbial and chemical contaminants.",
    "lemon juice in eyes cures sight.",
    "low battery high radiation.",
    "grapes medicine fatal.",
    "curd sugar boosts iq.",
    "ginger in ear cures infections.",
    "salt melts bones into water.",
    "rose water cures cataracts.",
    "lemon in nose cures jaundice.",
    "bitter gourd juice melts stones.",
    "lime powder cures stomach ache.",
    "eating with hands aids digestion.",
    "developing school policies and programmes that encourage and enable children to adopt and maintain a healthy diet .",
    "mustard oil on feet helps eyesight.",
    "providing nutrition and dietary counselling at primary health care facilities .",
    "beer melts kidney stones.",
    "hot water cures acne.",
    "eat a healthy diet and avoid sugar and saturated fat.",
    "protein supplements damage kidneys and liver.",
    "ladyfinger water cures diabetes.",
    "dirty dishes at night cause disease.",
    "educating children, adolescents and adults about nutrition and healthy dietary practices .",
    "laptops on lap cause infertility.",
    "onions clean viruses from air.",
    "aloe vera cures aids hiv.",
    "milk and salt cause leprosy.",
    "alkaline body via lemon cures cancer.",
    "baby food contains sawdust.",
    "reach and keep a health body weight.",
    "cold milk cures acidity.",
    "navel oiling cures dry lips.",
    "rock salt cures bp iodized salt causes it.",
    "heating honey makes it toxic.",
    "papaya seeds cure worms.",
    "bitter gourd juice melts stones.",
    "walking on grass cures eyesight.",
    "vicks on feet cures cough.",
    "raw milk whitens skin.",
    "copper water cures ulcers.",
    "lime powder cures stomach ache.",
    "standing and drinking water damages knees.",
    "needing to urinate more often than usual.",
    "curd in cold causes lung failure.",
    "baby food contains sawdust.",
    "cloves cure cavities permanently.",
    "bananas at night cause asthma death.",
    "potato whitens skin.",
    "papaya seeds cure worms.",
    "aloe vera cures aids hiv.",
    "onions clean viruses from air.",
    "dark circles liver failure.",
    "pink salt cures high bp.",
    "yellow food worsens jaundice.",
    "protein supplements damage kidneys and liver.",
    "paper cups block arteries.",
]

# Remove duplicates while preserving order
seen = set()
unique = []
for s in FAILING:
    if s not in seen:
        seen.add(s)
        unique.append(s)

log(f"Analyzing {len(unique)} unique failing sentences\n")
log("=" * 90)

BUCKET_KEYWORDS = {
    "no_verb":      [],
    "has_verb":     [],
    "gerund_subj":  [],
    "participial":  [],
    "compound_obj": [],
}

for sent_text in unique:
    doc = nlp(sent_text)

    root = [t for t in doc if t.dep_ == "ROOT"]
    root_token = root[0] if root else None
    root_info  = f"{root_token.text}/{root_token.pos_}/{root_token.dep_}" if root_token else "NONE"

    nsubj = [t for t in doc if t.dep_ in ("nsubj", "nsubjpass", "csubj")]
    nsubj_info = [(t.text, t.pos_, t.dep_) for t in nsubj]

    obj_deps = [t for t in doc if t.dep_ in ("dobj", "attr", "acomp", "pobj")]
    obj_info  = [(t.text, t.pos_, t.dep_) for t in obj_deps]

    action_verbs = [t for t in doc if t.pos_ == "VERB" or t.text.lower() in {
        "cures","causes","melts","boosts","helps","worsens","damages","cleans",
        "whitens","burns","contains","block","cause","cure"
    }]
    action_info = [(t.text, t.pos_, t.dep_) for t in action_verbs]

    log(f"\nSENT : {sent_text}")
    log(f"ROOT : {root_info}")
    log(f"NSUBJ: {nsubj_info}")
    log(f"OBJ  : {obj_info}")
    log(f"VERBS: {action_info}")
    log(f"FULL PARSE:")
    log(f"  {'Token':<20} {'POS':<8} {'DEP':<12} {'Head':<20}")
    log(f"  {'-'*62}")

    for t in doc:
        log(f"  {t.text:<20} {t.pos_:<8} {t.dep_:<12} {t.head.text:<20}")

    log("-" * 90)

log("""
HOW TO READ THIS:
- If ROOT is NOUN/ADJ → spaCy misparse, verb-anchor never fires (Bucket 5)
- If no NSUBJ → subject extraction fails even if verb found
- If VERBS list is empty → Bucket 1/3 (no verb at all)
- If action verb has dep=advcl/relcl → needs is_advcl_action fix
- If action verb has dep=compound → spaCy tagged verb as NOUN (Bucket 5)
""")