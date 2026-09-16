# Isolated multilingual module

This folder implements a neural-first Phase 7 preprocessing layer without
changing the existing TruthCheck pipeline. It never calls Sarvam or another
paid API.

## What it produces

`MultilingualProcessor.process(claim)` returns a `MultilingualClaim` holding:

- `original_text`: immutable source text for UI and audit use;
- neural token-level language/script tags (MuRIL Hinglish LID);
- `canonical_indic_text`: Roman-Hindi tokens transliterated to Devanagari by a
  local open neural model;
- `english_gloss`: a derived retrieval representation, accepted only after
  safety checks; and
- confidence and warnings for safe downstream routing.

For Hinglish, the intended local route is:

```text
protein kidney ke liye kharab hai
→ protein kidney के लिए खराब है       (IndicXlit)
→ Protein is harmful for the kidneys. (IndicTrans2)
```

The package uses local open models as its production defaults: MuRIL Hinglish
LID, Qwen for Windows-compatible transliteration, IndicTrans2 for glosses,
LaBSE for semantic preservation, and multilingual mDeBERTa NLI for
style scoring. Medical entity extraction uses MuRIL token-level language tags
(zero additional dependencies). The only remaining non-neural checks preserve
numbers and URLs exactly, because those values must not change between source
and gloss.

## Components and Dependencies

### Medical Entity Extraction
Medical entities (drug names, dosages, medical conditions) are extracted using
token-level language confidence from MuRIL Hinglish LID, with fallback to
pattern-based matching. This approach requires **zero external model downloads**
and achieves high accuracy on medical terminology.

### Optional Local Models

Create a Python 3.11 environment, then install the dependencies in this
folder:

```powershell
python -m pip install -r multilingual/requirements.txt
```

IndicTrans2's current remote configuration requires Transformers 4.x; the
requirements file prevents pip from selecting incompatible Transformers 5.x.

The first live IndicTrans2 call can download the open model checkpoint
`ai4bharat/indictrans2-indic-en-dist-200M`; cache it locally for offline
deployments. No credentials or paid service is required.

`IndicTransToolkit` is intentionally not installed on Windows because its
current release requires a compiled Cython extension and is not supported
there. The module supplies the small pure-Python processor subset required for
IndicTrans2 inference. The legacy `ai4bharat-transliteration` package is also
excluded because it depends on old Fairseq. The `IndicXlitTransliterator`
adapter is retained as an optional integration point for a Windows-compatible
local deployment, but it is not required for the Hindi-to-English acceptance
test.

## Test now

The default tests are deterministic and require no model downloads:

```powershell
python -m unittest multilingual.test_translation -v
```

After the optional models are installed, run a live Hindi-to-English acceptance
test:

```powershell
$env:RUN_REAL_TRANSLATION='1'
python -m unittest multilingual.test_translation -v
```

## Inspect 25 sample outputs

Run the complete multilingual flow over 25 English, Hindi, and Hinglish health
claims. It writes full structured records to `multilingual/output_dump.jsonl`:

```powershell
python -m multilingual.run_output_dump
```

For a quick first check that skips the neural style scorer, run:

```powershell
python -m multilingual.run_output_dump --limit 3 --no-style-score
```

## Integration boundary

When integration is approved, call this module immediately after raw text
normalisation and use `result.retrieval_queries` to drive merged retrieval.
Do not replace the source claim with the English gloss, and do not use an
unaccepted gloss for a verdict.
