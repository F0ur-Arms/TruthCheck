# Golden set v1 – annotation guidelines

File to fill: `evaluation/golden_set/golden_v1.csv` (UTF-8, one claim per row, header already present).
Target: 300-500 claims. Work on your own branch (`golden-set-v1`) and open a PR; never edit `golden_candidates_v0.jsonl`.

## Columns

| Column | Fill with |
|---|---|
| `id` | `g1-0001`, `g1-0002`, ... unique, never reused |
| `claim_text` | The claim exactly as a user would send it. Do not clean or translate it. Keep emojis, spelling mistakes, WhatsApp style. |
| `language` | `en`, `hi`, `hinglish`, or `other` (put the language in `notes`) |
| `script` | `latin`, `devanagari`, `other` |
| `claim_type` | `general`, `population_specific`, `treatment_cessation`, `dosage`, `food_remedy`, `vaccine`, `medical_advice_request` |
| `population` | `general`, `pregnancy`, `children`, `elderly`, `diabetes`, `hypertension`, `other:<text>` |
| `expected_verdict` | One of: `SUPPORTED`, `SUPPORTED_WITH_CAVEATS`, `MIXED_EVIDENCE`, `MISLEADING`, `MOSTLY_FALSE`, `FALSE`, `INSUFFICIENT_EVIDENCE`, `NOT_A_FACT_CHECK` |
| `gold_evidence_source` | KB filename (e.g. `who__xyz.txt`) that proves the verdict, or `NONE` if no source in `data/medical_kb/` covers it |
| `gold_evidence_quote` | The exact sentence from that source. Empty when `NONE`. |
| `evidence_url` | Public URL of the source (WHO, ICMR, NIH, Vishvas News, ...) |
| `in_kb` | `yes` if `gold_evidence_source` is in `data/medical_kb/`, else `no` |
| `claim_origin` | Where the claim came from (e.g. `vishvas_news`, `boom`, `whatsapp_forward`, `self_written`) |
| `annotator` | Your name |
| `second_annotator`, `second_verdict` | Filled by the second person on the shared subset (see below) |
| `status` | `draft` -> `labeled` -> `double_checked` -> `final` |
| `notes` | Anything unusual, ambiguity, why you chose the verdict |

## Mix to hit (counts of the total)

- Language: ~40% English, ~30% Hinglish (Roman script), ~20% Hindi (Devanagari), ~10% other (Tamil, Bengali, Marathi, ...).
- At least 15% `population_specific` (pregnancy, children, diabetes).
- At least 10% `treatment_cessation` or `dosage` (e.g. "stop taking insulin, bitter gourd cures diabetes").
- At least 15% where no KB document covers it (`in_kb = no`, verdict usually `INSUFFICIENT_EVIDENCE`).
- At least 5% `medical_advice_request` ("should I stop my BP tablets?") with verdict `NOT_A_FACT_CHECK`.
- Roughly balanced between true and false claims; do not let FALSE dominate.

## Hard rules

1. **No overlap with training/eval data.** Do not copy claims from `data/final_health_claims.csv`, `data/final_verifiedfacts.json` or `evaluation/golden_set/golden_candidates_v0.jsonl`. Search the claim text in those files before adding it.
2. **Label from evidence, not from memory.** Every non-`INSUFFICIENT_EVIDENCE` row needs a real source (quote + URL).
3. **Do not use the model to label.** You may use it to find candidate claims, but the verdict is yours.
4. **Same claim, different languages:** allowed and encouraged (separate rows, same verdict), but give them ids that are consecutive and mention the pair in `notes`.
5. Only claims, no personal data (names, phone numbers) from forwards.

## Verdict definitions

- `SUPPORTED`: the evidence directly supports the claim as stated.
- `SUPPORTED_WITH_CAVEATS`: true for some people/conditions/doses; the claim leaves out a limit that matters.
- `MIXED_EVIDENCE`: credible sources disagree, or studies conflict.
- `MISLEADING`: contains a true fact used to reach a false or dangerous conclusion.
- `MOSTLY_FALSE`: mostly wrong with a small true part.
- `FALSE`: contradicted by the evidence.
- `INSUFFICIENT_EVIDENCE`: no reliable source in the KB settles it. Use this honestly; it is a valid answer.
- `NOT_A_FACT_CHECK`: asks for personal medical advice or is not a verifiable claim.

## Double annotation

The second person independently labels at least 20% of rows (every 5th id) **without seeing** the first label: copy the `claim_text` into a separate sheet, label it, and the first annotator merges only `second_annotator` and `second_verdict` afterwards. Disagreements are discussed and the agreed verdict goes in `expected_verdict`, with the reason in `notes`. We compute Cohen's kappa on the shared rows.

## Done checklist

- [ ] 300+ rows, mix above met (a short count table in the PR description)
- [ ] No duplicate `claim_text`
- [ ] No row overlaps with the three files in rule 1
- [ ] Every row has a valid `expected_verdict` from the list
- [ ] Every row with `in_kb = yes` has a quote that actually appears in that file
- [ ] 20%+ rows have `second_verdict`
